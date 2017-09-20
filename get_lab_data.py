'''
	Queries Lab Dataset:
	Takes list of parameter types (COD, TSS, VFA, etc) as inputs
	Outputs dictionary of pandas dataframe objects 
	(parameter types as entry names)
'''

import os
from os.path import expanduser
import sys
import sqlite3
import pandas as pd


# Manages output directories
def get_indir():
	
	# Find the CR2C.Operations folder on Box Sync on the given machine
	targetdir = os.path.join('Box Sync','CR2C.Operations')
	mondir = None
	print("Searching for Codiga Center's Operations folder on Box Sync...")
	for dirpath, dirname, filename in os.walk(expanduser('~')):
		if dirpath.find(targetdir) > 0:
			mondir = os.path.join(dirpath,'MonitoringProcedures')
			print("Found Codiga Center's Operations folder on Box Sync")
			break
			
	# Alert user if Box Sync folder not found on machine
	if mondir == None:
		print("Could not find Codiga Center's Operations folder in Box Sync.")
		print('Please make sure that Box Sync is installed and the Operations folder is synced on your machine')
		sys.exit()
	
	return os.path.join(mondir,'Data')


def get_data(mtypes):

	data_indir = get_indir()

	# Load data from SQL
	os.chdir(data_indir)
	conn = sqlite3.connect('cr2c_lab_data.db')
	mdata_all = {}
	for mtype in mtypes:
		# Clean user input wrt TSS_VSS
		if mtype.find('TSS') >= 0 or mtype.find('VSS') >= 0:
			mtype = 'TSS_VSS'

		mdata_long = pd.read_sql(
			'SELECT * FROM {}_data'.format(mtype), 
			conn, 
			coerce_float = True
		)
		mdata_all[mtype] = mdata_long

	return mdata_all
