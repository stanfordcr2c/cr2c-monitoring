
''' 
	Script loads data from daily log sheets and outputs to SQL database
'''

from __future__ import print_function

# Data Prep
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime as dt
from datetime import timedelta

# Utilities
import warnings
import os
from os.path import expanduser
import sys
import re

# CR2C
from dependencies import cr2c_utils as cut

def clean_varname(varname):

	# Remove all special characters
	varname = re.sub(r'[^a-zA-Z\d\s]','',varname)
	# Replace spaces with a '_'
	varname = varname.replace(' ','_')
	# Convert to upper
	varname = varname.upper()

	# Pands gbq has 128 character limit for variable names
	return varname[:128]


def process_data(pydir, table_name = 'DailyLogResponsesV3', if_exists = 'append'):

	# Get the log data from gsheets
	fielddata = cut.get_gsheet_data(table_name, pydir)
	# Eliminate special characters (':-?[]()') and replace spaces with '_'	
	colnames_raw = fielddata.columns.values
	colnames_cln = [clean_varname(colname) for colname in colnames_raw]
	# Replace columns names of dataset with clean column names
	fielddata.columns = colnames_cln
	fielddata.loc[:,'Dkey'] = fielddata['TIMESTAMP'].astype(str)
	# Load data to Google BigQuery
	cut.write_to_db(fielddata,'cr2c-monitoring','fielddata', table_name, if_exists = if_exists)


