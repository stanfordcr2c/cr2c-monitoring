from __future__ import print_function
from tkinter.filedialog import askdirectory
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import warnings
import os
from os.path import expanduser
import sys
import sqlite3

class wide_tables:
	
	def __init__(self, table_end_dt, ndays_tables):
		self.table_end_dt = dt.strptime(table_end_dt,'%m-%d-%y')
		self.ndays_tables = ndays_tables

	# Manages output directories
	def get_dirs(self):
		
		# Find the CR2C.Operations folder on Box Sync on the given machine
		targetdir = os.path.join('Box Sync','CR2C.Operations')
		self.mondir = None
		print("Searching for Codiga Center's Operations folder on Box Sync...")
		for dirpath, dirname, filename in os.walk(expanduser('~')):
			if dirpath.find(targetdir) > 0:
				self.mondir = os.path.join(dirpath,'MonitoringProcedures')
				print("Found Codiga Center's Operations folder on Box Sync")
				break
				
		# Alert user if Box Sync folder not found on machine
		if self.mondir == None:
			print("Could not find Codiga Center's Operations folder in Box Sync.")
			print('Please make sure that Box Sync is installed and the Operations folder is synced on your machine')
			sys.exit()
		self.data_indir = os.path.join(self.mondir,'Data')

		# Request tables and charts output directory from user
		self.tables_outdir = askdirectory(title = 'Directory to output charts to')
		# Converts a long dataset to wide format


	def long_to_wide(self, df, id_vars):

	    # Create a multi-index
	    df.drop_duplicates(subset = id_vars, inplace = True)
	    df.set_index(id_vars, inplace = True)
	    
	    # Convert to wide format
	    if len(id_vars) > 2:
	        dfwide = df.unstack(id_vars[1])
	        
	        if len(id_vars) > 3:
	            dfwide = dfwide.unstack(id_vars[2])
	    
	    # Convert index to pandas native datetimeindex to allow easy date slicing
	    dfwide.reset_index(inplace = True)
	    dfwide.set_index('Sample Date & Time', inplace = True)
	    index = pd.to_datetime(dfwide.index)
	    dfwide.sort_index(inplace = True)
	    
	    return dfwide

		
	# Cleans wide dataset for output to tables
	def clean_wide_table(self, dfwide, value_vars):

		# First retrieve the stages for which there are data
		act_stages = dfwide.columns.levels[1].values
		# Reproduce stage order according to data availability
		act_st_ord = [stage for stage in self.stage_order if stage in act_stages]

		# Truncate 
		df_trunc = dfwide.Value.loc[self.table_start_dt:self.table_end_dt,(act_st_ord, value_vars)]

		# Set column order
		df_trunc = df_trunc.reindex_axis(act_st_ord, axis = 1, level = 'Stage')
		df_trunc = df_trunc.reindex_axis(value_vars, axis = 1, level = 'Type')

		# Create days since seed variable and insert as the first column
		if self.add_time_el == 1:
			days_since_seed = np.array((df_trunc.index - self.seed_dt).days)
			df_trunc.insert(0, 'Days Since Seed', days_since_seed)

		return df_trunc


	# Gets wide dataset, cleans and formats and outputs to csv
	def summarize_tables(self, add_time_el):

		self.add_time_el = 0
		if add_time_el == 1:
			self.add_time_el = 1

		self.get_dirs()
		try:
			os.chdir(self.tables_outdir)
		except OSError:
			print('Please choose a valid directory to output the tables to')
			sys.exit()

		# Specify key dates as per the length of time for table
		if not self.table_end_dt:
			self.table_end_dt = dt.now()

		table_end_dt_str  = dt.strftime(self.table_end_dt,'%m-%d-%y')
		self.table_start_dt = self.table_end_dt - timedelta(days = self.ndays_tables)
		self.seed_dt = dt.strptime('05-10-17','%m-%d-%y')

		# Load data from SQL
		os.chdir(self.data_indir)
		conn = sqlite3.connect('cr2c_lab_data.db')

		mdata_all = {}
		mtypes = ['COD','TSS_VSS','ALKALINITY','PH','VFA']

		# Loop through data typess to load into dictionary
		for mtype in mtypes: 

			mdata_long = pd.read_sql(
				'SELECT * FROM {}_data'.format(mtype), 
				conn, 
				coerce_float = True
			)
			# Set format of date variable
			mdata_long['Sample Date & Time'] = pd.to_datetime(mdata_long['Date_Time'])
			# Load resulting dataframe to dictionary
			mdata_all[mtype] = mdata_long

		# Specify id variables (same for every type since combining Alkalinity and pH)
		id_vars = ['Sample Date & Time','Stage','Type','obs_id']
		self.stage_order = \
		[
			'Raw Influent',
			'Grit Tank',
			'Microscreen',
			'AFBR',
			'Duty AFMBR MLSS',
			'Duty AFMBR Effluent',
			'Research AFMBR MLSS',
			'Research AFMBR Effluent'
		]

		# Get wide data
		CODwide = self.long_to_wide(mdata_all['COD'], id_vars)
		VFAwide = self.long_to_wide(mdata_all['VFA'], id_vars)
		TSS_VSSwide = self.long_to_wide(mdata_all['TSS_VSS'], id_vars)
		# For Alkalinity and pH, need to add Type variable back in
		ALK = mdata_all['ALKALINITY']
		ALK['Type'] = 'Alkalinity'
		PH = mdata_all['PH']
		PH['Type'] = 'pH'
		# Concatenate the two and reset index
		ALK_PH = pd.concat([PH,ALK], axis = 0, join = 'outer').reset_index(drop = True)
		# Get wide Alkalinity/pH dataset
		ALK_PHwide = self.long_to_wide(ALK_PH, id_vars)
		
		# Truncate and set column order
		CODtrunc = self.clean_wide_table(CODwide, ['Total','Soluble'])
		VFAtrunc = self.clean_wide_table(VFAwide, ['Acetate','Propionate'])
		TSS_VSStrunc = self.clean_wide_table(TSS_VSSwide,['TSS','VSS'])
		ALK_PHtrunc = self.clean_wide_table(ALK_PHwide,['pH','Alkalinity'])
		
		# Save
		os.chdir(self.tables_outdir)
		CODtrunc.to_csv('COD_table' + table_end_dt_str + '.csv')
		VFAtrunc.to_csv('VFA_table' + table_end_dt_str + '.csv')
		TSS_VSStrunc.to_csv('TSS_VSS_table' + table_end_dt_str + '.csv')
		ALK_PHtrunc.to_csv('ALK_PH_table' + table_end_dt_str + '.csv')

if __name__ == '__main__':

	# Instantiate class
	wtabs = wide_tables('9-14-17',126)

	# Create and output charts
	wtabs.summarize_tables(1)
