'''
	Creates wide tables for different water quality parameter types and outputs them
	to a directory of choice
'''

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
import get_lab_data as gld

class wide_tables:
	
	def __init__(self, table_end_dt_str, ndays_tables):
		self.table_end_dt_str = table_end_dt_str
		self.table_end_dt = dt.strptime(table_end_dt_str,'%m-%d-%y')
		self.ndays_tables = ndays_tables

	def long_to_wide(self, df, id_vars):

		# Create clean Date/Time Variable
		df['Sample Date & Time'] = pd.to_datetime(df['Date_Time'])
		all_vars = id_vars + ['Value']
		df = df[all_vars]

		# Create a multi-index
		df.drop_duplicates(subset = id_vars, inplace = True)
		df.set_index(id_vars, inplace = True)

		# Convert to wide format
		if len(id_vars) > 2:
			dfwide = df.unstack(id_vars[1])
			if len(id_vars) > 3:
				dfwide = dfwide.unstack(id_vars[2])
		elif len(id_vars) > 1:
			dfwide = df.unstack(id_vars[1])
		
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

		# Truncate (adding exception for Ammonia with no type variable)
		if value_vars == ['Value']:
			column_tuple = act_st_ord
		else:
			column_tuple = (act_st_ord, value_vars)
		df_trunc = dfwide.Value.loc[self.table_start_dt:self.table_end_dt, column_tuple]

		# Set column order (again, exception is for Ammonia with no type variable)
		if value_vars == ['Value']:
			df_trunc = df_trunc.reindex_axis(act_st_ord, axis = 1, level = None)
		else:
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

		self.tables_outdir = askdirectory(title = 'Directory to output tables to:')
		try:
			os.chdir(self.tables_outdir)
		except OSError:
			print('Please choose a valid directory to output the tables to')
			sys.exit()

		# Specify key dates as per the length of time for table
		if not self.table_end_dt:
			self.table_end_dt = dt.now()

		self.table_start_dt = self.table_end_dt - timedelta(days = self.ndays_tables)
		self.seed_dt = dt.strptime('05-10-17','%m-%d-%y')

		# Load data from SQL
		mdata_all = gld.get_data(['COD','TSS_VSS','ALKALINITY','PH','VFA','Ammonia'])

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

		# For Alkalinity and pH, need to add Type variable back in
		ALK = mdata_all['ALKALINITY']
		ALK['Type'] = 'Alkalinity'
		PH = mdata_all['PH']
		PH['Type'] = 'pH'
		NH3 = mdata_all['Ammonia']
		# Concatenate the two and reset index
		ALK_PH = pd.concat([PH,ALK], axis = 0, join = 'outer').reset_index(drop = True)

		# Get wide data
		CODwide = self.long_to_wide(mdata_all['COD'], id_vars)
		VFAwide = self.long_to_wide(mdata_all['VFA'], id_vars)
		TSS_VSSwide = self.long_to_wide(mdata_all['TSS_VSS'], id_vars)
		ALK_PHwide = self.long_to_wide(ALK_PH, id_vars)
		NH3wide = self.long_to_wide(NH3, ['Sample Date & Time','Stage'])
		
		# Truncate and set column order
		CODtrunc = self.clean_wide_table(CODwide, ['Total','Soluble'])
		VFAtrunc = self.clean_wide_table(VFAwide, ['Acetate','Propionate'])
		TSS_VSStrunc = self.clean_wide_table(TSS_VSSwide,['TSS','VSS'])
		ALK_PHtrunc = self.clean_wide_table(ALK_PHwide,['pH','Alkalinity'])
		NH3trunc = self.clean_wide_table(NH3wide,['Value'])
		
		# Save
		os.chdir(self.tables_outdir)
		CODtrunc.to_csv('COD_table' + self.table_end_dt_str + '.csv')
		VFAtrunc.to_csv('VFA_table' + self.table_end_dt_str + '.csv')
		TSS_VSStrunc.to_csv('TSS_VSS_table' + self.table_end_dt_str + '.csv')
		ALK_PHtrunc.to_csv('ALK_PH_table' + self.table_end_dt_str + '.csv')
		NH3trunc.to_csv('Ammonia_table' + self.table_end_dt_str + '.csv')

if __name__ == '__main__':

	# Instantiate class
	wtabs = wide_tables('11-10-17',175)

	# Create and output charts
	wtabs.summarize_tables(1)
