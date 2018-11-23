
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

# CR2Cf
import cr2c_utils as cut

def clean_varname(varname):
	for char in '-:?[]()<>.,':
		varname = varname.replace(char,'')
		varname = varname.replace(' ','_')
		varname = varname.upper()
	return varname


def process_data(tableName = 'DailyLogResponses'):

	# Get the log data from gsheets
	fielddata = cut.get_gsheet_data(tableName)
	# Eliminate special characters (':-?[]()') and replace spaces with '_'	
	colnamesRaw = fielddata.columns.values
	colnamesCln = [clean_varname(colname) for colname in colnamesRaw]
	# Replace columns names of dataset with clean column names
	fielddata.columns = colnamesCln

	# SQL command strings for sqlite3
	colnamesStr = ','.join(colnamesCln[1:])
	colInsStr = ','.join(['?']*len(colnamesCln))
	create_str = """
		CREATE TABLE IF NOT EXISTS {0} (Timestamp CHAR PRIMARY KEY, {1})
	""".format(tableName,colnamesStr)
	ins_str = """
		INSERT OR REPLACE INTO {0} (Timestamp,{1})
		VALUES ({2})
	""".format(tableName,colnamesStr,colInsStr)

	# Change to data directory
	data_dir = cut.get_dirs()[0]
	os.chdir(data_dir)
	# Create SQL object connection
	conn = sqlite3.connect('cr2c_fielddata.db')
	# Create table if it doesn't exist
	conn.execute(create_str)
	# Insert aggregated values for the sid and time period
	conn.executemany(
		ins_str,
		fielddata.to_records(index = False).tolist()
	)
	conn.commit()
	# Close Connection
	conn.close()

	# Load data to Google BigQuery
	projectid = 'cr2c-monitoring'
	dataset_id = 'fielddata'
	# Make sure only new records are being appended to the dataset
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
		varNamesAll = 'Timestamp,' + ','.join(varNames)
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
	fielddata['Timestamp'] = pd.to_datetime(fielddata['Timestamp'])
	fielddata.dropna(subset = ['Timestamp'], inplace = True)

	if start_dt_str:
		fielddata = fielddata.loc[fielddata['Date_Time'] >= start_dt,:]
	if end_dt_str:
		fielddata = fielddata.loc[fielddata['Date_Time'] <= end_dt + timedelta(days = 1),:]

	# Output csv if desired
	if output_csv:
		if varNames:
			op_fname = '{0}.csv'.format(','.join(varNames))
		else:
			op_fname = 'cr2c-fieldData.csv'
		os.chdir(outdir)
		fielddata.to_csv(op_fname, index = False, encoding = 'utf-8')

	return fielddata

