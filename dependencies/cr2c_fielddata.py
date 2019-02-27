
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


def process_data(pydir, tableName = 'DailyLogResponses', create_table = False):

	# Get the log data from gsheets
	fielddata = cut.get_gsheet_data(tableName, pydir)
	# Eliminate special characters (':-?[]()') and replace spaces with '_'	
	colnamesRaw = fielddata.columns.values
	colnamesCln = [clean_varname(colname) for colname in colnamesRaw]
	# Replace columns names of dataset with clean column names
	fielddata.columns = colnamesCln
	# Load data to Google BigQuery
	projectid = 'cr2c-monitoring'
	dataset_id = 'fielddata'
	
	# If overwriting entire table (necessary if changing column names), field_data_already = None
	if create_table:
		fielddata_new = fielddata.copy()
	# Otherwise, make sure only new records are being appended to the dataset
	else:
		fielddata_already = get_data()
		fielddata_new = fielddata.loc[~fielddata['TIMESTAMP'].isin(fielddata_already['TIMESTAMP']),:]
	# Remove duplicates and missing values
	fielddata_new.dropna(subset = ['TIMESTAMP'], inplace = True)
	fielddata_new.drop_duplicates(inplace = True)
	# Write to gbq table
	if not fielddata_new.empty:
		fielddata_new.to_gbq('{}.{}'.format(dataset_id, tableName), projectid, if_exists = 'append')


def get_data(varNames = None, start_dt_str = None, end_dt_str = None, output_csv = False, outdir = None):

	# Convert date string inputs to dt variables
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

	tableNames = ['DailyLogResponses','DailyLogResponsesV2']

	# Set "varNames" to * if none given
	if varNames:
		varNames = [varName.upper() for varName in varNames]
		varNamesAll = 'TIMESTAMP,' + ','.join(varNames)
	else:
		varNamesAll = '*'

	fielddata = pd.DataFrame([])
	projectid = 'cr2c-monitoring'
	dataset_id = 'fielddata'

	for tableName in tableNames:

		fielddata = pd.concat(
			[
				fielddata,
				# Load data from google BigQuery
				pd.read_gbq('SELECT * FROM {}.{}'.format(dataset_id, tableName), projectid)
			],
			axis = 0,
			join = 'outer'
		)

	# Dedupe data (some issue with duplicates)
	fielddata.drop_duplicates(inplace = True)
	# Convert Date_Time variable to a pd datetime and eliminate missing values
	fielddata['TIMESTAMP'] = pd.to_datetime(fielddata['TIMESTAMP'])
	fielddata.dropna(subset = ['TIMESTAMP'], inplace = True)

	if start_dt_str:
		fielddata = fielddata.loc[fielddata['Date_Time'] >= start_dt,:]
	if end_dt_str:
		fielddata = fielddata.loc[fielddata['Date_Time'] <= end_dt + timedelta(days = 1),:]

	# Output csv if desired
	if output_csv:
		if varNames:
			op_dsn = '{0}.csv'.format(','.join(varNames))
		else:
			op_dsn = 'cr2c-fieldData.csv'
		fielddata.to_csv(os.path.join(outdir, out_dsn), index = False, encoding = 'utf-8')

	return fielddata

