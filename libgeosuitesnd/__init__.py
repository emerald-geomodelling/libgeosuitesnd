from os.path import splitext, basename
from itertools import count
import numpy as np
import datetime
import time
import pandas as pd
import warnings
from scipy.stats import linregress
import zipfile
import codecs
import os
import io
import logging
import pkg_resources

logger = logging.getLogger(__name__)

na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
             '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null']

def _read_csv(f):
    return pd.read_csv(f, na_values=na_values, keep_default_na=False).set_index("code")


with pkg_resources.resource_stream("libgeosuitesnd", "methods.csv") as f:
    methods = _read_csv(f)
    method_by_code = methods["name"]
with pkg_resources.resource_stream("libgeosuitesnd", "stop_reasons.csv") as f:
    stop_reasons = _read_csv(f)
    stop_reason_by_code = stop_reasons["name"]
with pkg_resources.resource_stream("libgeosuitesnd", "flags.csv") as f:
    flags = _read_csv(f)

snd_columns_by_method = {key: value.split(",") for key, value in methods["columns"].to_dict().items() if isinstance(value, str)}

#method_by_name = {v:k for k,v in method_by_code.items()}
stop_reason_by_name = {value:key for key, value in stop_reason_by_code.items()}


def parse_coordinates_asterisk_lines(data):
    # Parse coordinates
    y = float(data[0])
    x = float(data[1])
    z = float(data[2])

    # Find number and location of lines with * seperators
    asterisk_lines = [i for i, j in zip(count(), data) if j == '*']
    return x, y, z, asterisk_lines

def parse_header_information(data, asterisk, borehole_id):

    header00 = data[asterisk + 1].split()

    try:
        method_code = int(header00[0])
    except Exception:
        logger.info(borehole_id + ': method code not valid')
        method_name = None
    method_name = method_by_code.get(method_code, "geosuitesnd_%s" % method_code)

    try:
        date_components = header00[1].split('.')
        day = int(date_components[0])
        month = int(date_components[1])
        year = int(date_components[2])
        date = str(year) + '-' + str(month) + '-' + str(day)
    except Exception:
        logger.info(borehole_id + ': no date')
        day = None
        month = None
        year = None
        date = None

    try:
        stop_code = int(data[asterisk + 2].split()[1])
    except Exception:
        logger.info(borehole_id + ': Something went wrong reading stop code')
        stop_code = None
    stop_desc = stop_reason_by_code.get(stop_code, "geosuitesnd_%s" % stop_code)
    return method_code, method_name, day, month, year, date, stop_code, stop_desc

def parse_string_data_column(df_data, raw_data_nestedlist,n_data_col):
    depth_bedrock = None

    string_data = [x[n_data_col:] for x in raw_data_nestedlist]

    df_data["comments"] = ""
    
    for count, string in enumerate(string_data):
        line_flags = {}
        for flag in string:
            if flag in flags.index:
                line_flags[flags.loc[flag]["name"]] = flags.loc[flag]["value"]

        if line_flags.get("depth_bedrock", 0) and depth_bedrock is None:
            depth_bedrock = df_data.depth[count]

        for key, value in line_flags.items():
            if key != "depth_bedrock":
                if key not in df_data.columns:
                    df_data[key] = -1
                df_data.loc[count, key] = value

        df_data.loc[count, "comments"] = ' '.join(string)

    for flag in flags["name"].unique():
        df_data[flag].replace(to_replace=-1, method='ffill', inplace=True)
        df_data.loc[df_data[flag] == -1, flag] = 0
        
    return df_data, depth_bedrock

def parse_borehole_data(data, method_code, asterisk_lines,asterisk_line_idx, borehole_id):
    depth_increment = None
    depth_bedrock = None
    line_start = asterisk_lines[asterisk_line_idx] + 3
    line_end = asterisk_lines[asterisk_line_idx + 1]
    df_data = pd.DataFrame()

    try:
        raw_data_string = data[line_start:line_end]
        raw_data_nestedlist = [x.split() for x in raw_data_string]

        n_data_col = min([len(x) for x in raw_data_nestedlist])
        column_names = snd_columns_by_method[method_code]
        for i in range(0,n_data_col):
            # todo: name columns based on entries in snd_columns_by_method. CCh, 2020-03-06
            if i < len(column_names):
                df_data.loc[:, column_names[i]] = [float(x[i]) for x in raw_data_nestedlist]
            else:
                df_data.loc[:, method_by_code[method_code]+'_Col'+str(i)] = [float(x[i]) for x in raw_data_nestedlist]
        # todo: Set 0 values for resistivity in R-CPT Data to a dummy value (np.nan or -9999?). CCh, 2020-03-06

        depth_increment = df_data.depth[1] - df_data.depth[0] #todo: depth increment not being properly read for CPT data

        method_flags = methods.loc[method_code, "flags"]
        if method_flags:
            for flag in method_flags.split(","):
                df_data[flag] = -1
        df_data, depth_bedrock = parse_string_data_column(df_data, raw_data_nestedlist, n_data_col)


    except Exception:
        logger.info(borehole_id + ': No data extracted for text block ' + str(asterisk_line_idx))
    return df_data, depth_increment, depth_bedrock

def parse(input_filename, borehole_id=None):
    if borehole_id is None:
        borehole_id = input_filename.split("/")[-1].split(".", 1)[0]

    def load(f):
        f=codecs.getreader('utf8')(f, errors='ignore')
        data = f.readlines()
        return [l.strip() for l in data]
        
    if isinstance(input_filename, str):
        with open(input_filename, "rb") as f:
            data = load(f)
    else:
        data = load(input_filename)

    x, y, z, asterisk_lines = parse_coordinates_asterisk_lines(data)

    if not len(asterisk_lines) == 4:
        logger.info(borehole_id + ': number of asterisk lines in file = ' + str(len(asterisk_lines)))

    if len(asterisk_lines) < 4:
        logger.info(borehole_id + ': number of asterisk lines in file = ' + str(len(asterisk_lines)))
        logger.info('Skipping file: %s - file is missing final asterisk and may be corrupt' % borehole_id)

    # The E16 Nybakk-Slomarka project is a bit weird because old holes have separate SND files for Total and rotary
    # pressure soundings, whereas newer holes sometimes merge these into the same file. Some CPT files also are
    # missing a few header lines that contain a global ID of some sort. I think the best method here is to check if
    # the second line after an asterisk starts with a 1. If so, it means this is a block of text with data
    
    res = []
    for asterisk_line_idx, asterisk in enumerate(asterisk_lines):
        depth_increment = None
        # check that the first number two lines after the asterisk is a one.
        # If not, continue to next asterisk
        try:
            if not float(data[asterisk + 2].split(' ')[0]) - 1 < 0.0001:
                continue
        except Exception:
            continue

        # If so, parse information from the header lines before the data.
        method_code, method_name, day, month, year, date, stop_code, stop_desc = parse_header_information(data, asterisk, borehole_id)
        df_data, depth_increment, depth_bedrock = parse_borehole_data(data, method_code, asterisk_lines, asterisk_line_idx, borehole_id)

        res.append({
            "main": [{
                "method_code": method_name,
                "method_code_orig": method_code,
                "day": day,
                "month": month,
                "year": year,
                "date": date,
                "stop_code": stop_desc,
                "stop_code_orig": stop_code,
                "depth_increment": depth_increment,
                "depth_bedrock": depth_bedrock,
                "x_coordinate": x,
                "y_coordinate": y,
                "z_coordinate": z,
                "investigation_point": borehole_id
            }],
            "data": df_data,
        })

    return res
