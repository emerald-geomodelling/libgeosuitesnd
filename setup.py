#!/usr/bin/env python

import setuptools
import os

setuptools.setup(
    name='libgeosuitesnd',
    version='0.0.5',
    description='Parser for the GeoSuite<tm> SND export format',
    long_description="""Parser for the GeoSuite<tm> SND export format""",
    long_description_content_type="text/markdown",
    author='Egil Moeller, Craig William Christensen and others ',
    author_email='em@emeraldgeo.no, cch@emeraldgeo.no',
    url='https://github.com/emerald-geomodelling/libgeosuitesnd',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'libgeosuitesnd': ['*/*.csv']},
    install_requires=[
        "numpy",
        "pandas",
    ],
)
