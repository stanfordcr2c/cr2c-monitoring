''' 
	Script loads data from daily log sheets and outputs to SQL database
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime as dt
from datetime import timedelta

import warnings
import os
from os.path import expanduser
import sys
import re
import cr2c_utils as cut

from tkinter.filedialog import askdirectory

def rmChars(targChars,replCar,string):
	for char in targChars:
		string = string.replace(char,replCar)
	return string

def process_data(tableName = 'DailyLogResponses'):

	# Get the log data from gsheets
	logdata = cut.get_gsheet_data([tableName])
	# Eliminate special characters (':-?[]()') and replace spaces with '_'	
	colnamesRaw = logdata.columns.values
	colnamesCln = [rmChars('-:?[]()','',colname) for colname in colnamesRaw]
	colnamesCln = [rmChars(' ','_',colname) for colname in colnamesCln]
	# Replace columns names of dataset with clean column names
	logdata.columns = colnamesCln

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
	conn = sqlite3.connect('cr2c_field_data.db')
	# Create table if it doesn't exist
	conn.execute(create_str)
	# Insert aggregated values for the elid and time period
	conn.executemany(
		ins_str,
		logdata.to_records(index = False).tolist()
	)
	conn.commit()

	# Close Connection
	conn.close()


def get_data(tableName = 'DailyLogResponses', start_dt_str = None, end_dt_str = None):

	# Convert date string inputs to dt variables
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

	# Load data from SQL
	data_dir = cut.get_dirs()[0]
	os.chdir(data_dir)
	conn = sqlite3.connect('cr2c_field_data.db')

	fielddata = pd.read_sql(
		'SELECT * FROM DailyLogResponses', 
		conn, 
		coerce_float = True
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

	return fielddata

