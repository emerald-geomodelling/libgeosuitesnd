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

logger = logging.getLogger(__name__)

snd_columns_by_method = {7:['depth', 'feed_trust_force', 'pore_pressure', 'friction', 'pressure', 'resistivity'],
                25:['depth', 'feed_trust_force', 'interval', 'pumping_rate'],
                26:['depth', 'feed_trust_force', 'interval', 'pumping_rate'],
                23:['depth', 'feed_trust_force']}

method_by_code = {
    22: 'simple',
    21: 'rotary',
    23: 'rps',
    25: 'total',
     7: 'cpt',
    # rock_drilling is an older version a total sounding,
    # that doesn't have feedforce on the way down
    26: 'rock_drilling'
}
stop_reason_by_code = {
    90: 'drilling_abandoned_prematurely',
    91: 'abandoned_hit_hard_surface',
    92: 'assumed_hit_boulder',
    93: 'assumed_bedrock',
    94: 'reached_bedrock',
    95: 'broken_drill',
    96: 'other_fault',
    97: 'drilling_abandoned'
}

method_by_name = {v:k for k,v in method_by_code.items()}
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
    try:
        method_name = method_by_code[method_code]
    except Exception:
        logger.info(borehole_id + ': method code value not recognized')
        method_name = None

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
    try:
        stop_desc = stop_reason_by_code[stop_code]
    except Exception:
        logger.info(borehole_id + ': Stop code not recognized')
        stop_desc = None
    return method_code, method_name, day, month, year, date, stop_code, stop_desc

def parse_string_data_column(df_data, raw_data_nestedlist,n_data_col):
    depth_bedrock = None

    string_data = [x[n_data_col:] for x in raw_data_nestedlist]
    flushing = np.zeros(len(string_data), dtype=np.int)
    extra_spin = np.zeros(len(string_data), dtype=np.int)
    ramming = np.zeros(len(string_data), dtype=np.int)
    pumping = np.zeros(len(string_data), dtype=np.int)
    comments = []

    flushing_state = 0
    okt_rotasjon_state = 0
    ramming_state = 0
    pumping_state = 0

    for count, string in enumerate(string_data):

        #check for the letter version of the codes
        if 'R1' in string:
            okt_rotasjon_state = 1
            string.remove('R1')
        elif 'R2' in string:
            okt_rotasjon_state = 0
            string.remove('R2')

        if 'y1' in string:
            flushing_state = 1
            string.remove('y1')
        elif 'y2' in string:
            flushing_state = 0
            string.remove('y2')

        if 'S1' in string:
            ramming_state  = 1
            string.remove('S1')
        elif 'S2' in string:
            ramming_state = 0
            string.remove('S2')

        if 'D1' in string:
            ramming_state  = 1
            flushing_state = 1
            string.remove('D1')
        elif 'D2' in string:
            ramming_state = 0
            flushing_state = 0
            string.remove('D2')

        if 'P1' in string:
            pumping_state  = 1
            string.remove('P1')
        elif 'P2' in string:
            pumping_state= 0
            string.remove('P2')

        if 'F' in string:
            if depth_bedrock is None:
                depth_bedrock = df_data.depth[count]
            string.remove('F')

        # check for the number version of the codes
        if '70' in string:
            okt_rotasjon_state = 1
            string.remove('70')
        elif '71' in string:
            okt_rotasjon_state = 0
            string.remove('71')

        if '72' in string:
            flushing_state = 1
            string.remove('72')
        elif '73' in string:
            flushing_state = 0
            string.remove('73')

        if '74' in string:
            ramming_state  = 1
            string.remove('74')
        elif '75' in string:
            ramming_state = 0
            string.remove('75')

        if '76' in string:
            ramming_state  = 1
            flushing_state = 1
            string.remove('76')
        elif '77' in string:
            ramming_state = 0
            flushing_state = 0
            string.remove('77')

        if '78' in string:
            pumping_state  = 1
            string.remove('78')
        elif '79' in string:
            pumping_state= 0
            string.remove('79')

        if '43' in string:
            if depth_bedrock is None:
                depth_bedrock = df_data.depth[count]
            string.remove('43')

        flushing[count] = flushing_state
        extra_spin[count] = okt_rotasjon_state
        ramming[count] = ramming_state
        pumping[count]=pumping_state
        comments.append(' '.join(string))

    df_data.loc[:, 'flushing'] = flushing
    df_data.loc[:, 'extra_spin'] = extra_spin
    df_data.loc[:, 'ramming'] = ramming
    df_data.loc[:, 'pumping'] = pumping
    df_data.loc[:, 'comments'] = comments

    return df_data, depth_bedrock

def parse_borehole_data(data, method_code, asterisk_lines,asterisk_line_idx, input_filename):
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

        if method_code in [23, 25]:
            df_data, depth_bedrock = parse_string_data_column(df_data, raw_data_nestedlist, n_data_col)


    except Exception:
        logger.info(input_filename, 'No data extracted for text block ' + str(asterisk_line_idx))
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
        df_data, depth_increment, depth_bedrock = parse_borehole_data(data, method_code, asterisk_lines, asterisk_line_idx, input_filename)

        res.append({
            "main": [{
                "method_code": method_code,
                "method_name": method_name,
                "day": day,
                "month": month,
                "year": year,
                "date": date,
                "stop_code": stop_code,
                "stop_desc": stop_desc,
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
