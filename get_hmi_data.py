'''
	Queries HMI Aggregated Dataset:
	Takes a list of element IDs, aggregation time periods and aggregation types as inputs
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
	if not mondir:
		if os.path.isdir('D:/'):
			for dirpath, dirname, filename in os.walk('D:/'):
				if dirpath.find(targetdir) > 0:
					mondir = os.path.join(dirpath,'MonitoringProcedures')
					print("Found Codiga Center's Operations folder on Box Sync")
					break
		if not mondir:
			print("Could not find Codiga Center's Operations folder in Box Sync")
			print('Please make sure that Box Sync is installed and the Operations folder is synced on your machine')
			sys.exit()
	
	return os.path.join(mondir,'Data')

def get_data(elids, agg_types, tperiods):

	data_indir = get_indir()

	# Load data from SQL
	os.chdir(data_indir)
	conn = sqlite3.connect('cr2c_hmi_agg_data.db')
	hmi_data_all = {}

	for elid, agg_type, tperiod in zip(elids, agg_types, tperiods):

		hmi_data = pd.read_sql(
			'SELECT * FROM {0}_{1}s_{2}hour'.format(elid, agg_type, tperiod), 
			conn, 
			coerce_float = True
		)


		# Dedupe data (some issue with duplicates)
		hmi_data.drop_duplicates(inplace = True)

		hmi_data_all['{0}_{1}_{2}hour'.format(elid, agg_type, tperiod)] = hmi_data

	return hmi_data_all


