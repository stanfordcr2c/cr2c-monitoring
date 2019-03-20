
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
import cr2c_utils as cut

def clean_varname(varname):
	for char in '-:?[]()<>.,':
		varname = varname.replace(char,'')
		varname = varname.replace(' ','_')
		varname = varname.upper()
	return varname[:128]


def process_data(pydir, table_name = 'DailyLogResponsesV2', create_table = False):

	# Get the log data from gsheets
	fielddata = cut.get_gsheet_data(table_name, pydir)
	# Eliminate special characters (':-?[]()') and replace spaces with '_'	
	colnames_raw = fielddata.columns.values
	colnames_cln = [clean_varname(colname) for colname in colnames_raw]
	# Replace columns names of dataset with clean column names
	fielddata.columns = colnames_cln
	fielddata.loc[:,'Dkey'] = fielddata['TIMESTAMP'].astype(str)
	# Load data to Google BigQuery
	cut.write_to_db(fielddata,'cr2c-monitoring','fielddata', table_name, create_mode = create_table)


