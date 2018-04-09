
'''
	Script loads data from lab tests, computes water quality parameters,
	and loads the data to an SQL database (no inputs required)
'''

from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime as dt
from datetime import timedelta

import warnings
import os
from os.path import expanduser
import sys
import cr2c_utils as cut


import seaborn as sns
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates

# Queries Lab Data SQL File 
def get_data(ltypes, start_dt_str = None, end_dt_str = None, output_csv = False, outdir = None):

	# Convert date string inputs to dt variables
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

	if output_csv and not outdir:
		print('Directory to output Lab data to...')
		outdir = askdirectory(title = 'Directory to output Lab data to...')

	# Load data from SQL
	data_dir = cut.get_dirs()[0]
	os.chdir(data_dir)
	conn = sqlite3.connect('cr2c_lab_data.db')

	# Loop through types of lab data types (ltypes)
	mdata_all = {}
	for ltype in ltypes:

		# Clean user input wrt TSS_VSS
		if ltype.find('TSS') >= 0 or ltype.find('VSS') >= 0:
			ltype = 'TSS_VSS'

		mdata_long = pd.read_sql(
			'SELECT * FROM {0}'.format(ltype), 
			conn, 
			coerce_float = True
		)

		# Dedupe data (some issue with duplicates)
		mdata_long.drop_duplicates(inplace = True)
		# Convert Date_Time variable to a pd datetime and eliminate missing values
		mdata_long['Date_Time'] = pd.to_datetime(mdata_long['Date_Time'])
		mdata_long.dropna(subset = ['Date_Time'], inplace = True)
		# Filter to desired dates
		mdata_long.drop('DKey', axis = 1, inplace = True)
		if start_dt_str:
			mdata_long = mdata_long.loc[mdata_long['Date_Time'] >= start_dt,:]
		if end_dt_str:
			mdata_long = mdata_long.loc[mdata_long['Date_Time'] <= end_dt + timedelta(days = 1),:]
		
		# Output csv if desired
		if output_csv:
			os.chdir(outdir)
			mdata_long.to_csv(ltype + '.csv', index = False, encoding = 'utf-8')

		# Write to dictionary
		mdata_all[ltype] = mdata_long

	return mdata_all


class labrun:
	
	def __init__(self, verbose = False):
		
		self.ltype_list = \
			['PH','COD','TSS_VSS','ALKALINITY','VFA','GasComp','Ammonia','Sulfate','TKN']
		self.min_feas_dt_str = '6-1-16'
		self.min_feas_dt = dt.strptime(self.min_feas_dt_str, '%m-%d-%y')
		self.file_dt = dt.now()
		self.file_dt_str = dt.strftime(self.file_dt,'%m-%d-%y')
		self.data_dir, self.pydir = cut.get_dirs()
		self.log_dir = os.path.join(self.data_dir,'Logs')
		self.verbose = verbose


	# Adds desriptive treatment stage variable to dataset for plotting
	def get_stage_descs(self):

		conditions = [
			self.mdata['Stage'] == 'DAFMBREFF',
			self.mdata['Stage'] == 'DAFMBRMLSS',
			self.mdata['Stage'] == 'RAFMBREFF',
			self.mdata['Stage'] == 'RAFMBRMLSS',
			self.mdata['Stage'] == 'RAW',
			self.mdata['Stage'] == 'GRIT',
			self.mdata['Stage'] == 'MS'
		]
		choices = [
			'Duty AFMBR Effluent','Duty AFMBR MLSS',
			'Research AFMBR Effluent','Research AFMBR MLSS',
			'Raw Influent','Grit Tank','Microscreen'
		]
		self.mdata['Stage'] = np.select(conditions, choices, default = self.mdata['Stage'])


	# Manages duplicate observations removes duplicates (with warnings)
	# gets observation id's for purposeful duplicates 
	def manage_dups(self, ltype, id_vars):

		# Set duplicates warning
		dup_warning = \
			'There are repeat entries and/or entries with no date in {0} that were removed. '+\
	        'A csv of the removed values has been saved as {1}'

		# Check for Duplicates and empties
		repeat_entries = np.where(self.mdata.duplicated())[0].tolist()
		blank_entries = np.where(pd.isnull(self.mdata.Date))[0].tolist()
		repeat_entries.extend(blank_entries)

	    # If found, remove, print warning and output csv of duplicates/empties
		if self.verbose:
			if len(repeat_entries) > 0:
				os.chdir(self.log_dir)
				dup_filename = ltype + 'duplicates' + self.file_dt_str
				warnings.warn(dup_warning.format(ltype,dup_filename + '.csv'))
				self.mdata.iloc[repeat_entries].to_csv(dup_filename + '.csv')
		
		# Eliminate duplicate data entries and reset the index
		self.mdata.drop_duplicates(keep = 'first', inplace = True)
		self.mdata.reset_index(drop = True, inplace = True)
		
		# Sort the dataset
		self.mdata.sort_values(id_vars)

		# Create a list of observation ids (counting from 0)
		obs_ids = [0]
		for obs_no in range(1,len(self.mdata)):
			row_curr = [self.mdata[id_var][obs_no] for id_var in id_vars]
			row_prev = [self.mdata[id_var][obs_no - 1] for id_var in id_vars]
			obs_id_curr = obs_ids[-1]
			if row_curr == row_prev:
				# Same date/stage/type, increment obs_id
				obs_ids.append(obs_id_curr + 1)
			else:
				# Different date/stage/type, restart obs_id count
				obs_ids.append(0)

		#Add obs_id variable to dataset
		self.mdata['obs_id'] = obs_ids

	# Tries to format a variable and outputs error message if the input data are off
	def set_var_format(self, ltype, variable, format, format_prt):

		var_typ_warn = \
			'Check {0} variable in {1}. An entry is incorrect, format should be {2}'

		self.mdata = self.mdata.apply(
			lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan
		)
		try:
			if variable == 'Date':
				self.mdata['Date'] = pd.to_datetime(self.mdata['Date'], format = '%m-%d-%y')
			else:
				self.mdata[variable] = self.mdata[variable].astype(format)
		except TypeError:
			print(var_typ_warn.format(variable, ltype, format_prt))
			sys.exit()
		except ValueError:
			print(var_typ_warn.format(variable, ltype, format_prt))
			sys.exit()


	# Cleans a processed dataset, converting it to long format for plotting, output, etc
	def clean_dataset(self, ltype, id_vars):

		self.set_var_format(ltype, 'Date', None, "m-d-yy")
		# Eliminate missing date variables
		self.mdata.dropna(subset = ['Date'], inplace = True)
		# Make sure all date variables are within a reasonable range
		date_rng_warn = \
			'A Date variable in {0} has been entered incorrectly as {1} and removed'
		if self.mdata.Date.min() < self.min_feas_dt:
			print(date_rng_warn.format(ltype,self.mdata.Date.min()))
		if self.mdata.Date.max() > self.file_dt:
			print(date_rng_warn.format(ltype,self.mdata.Date.max()))
		# Filter dates accordingly
		self.mdata = self.mdata.loc[
			(self.mdata.Date >= self.min_feas_dt) &
			(self.mdata.Date <= self.file_dt)
		]

		# Format and clean stage variable
		if ltype != 'GasComp':

			self.mdata['Stage'] = self.mdata['Stage'].astype(str)
			self.mdata['Stage'] = self.mdata['Stage'].str.upper()
			self.mdata['Stage'] = self.mdata['Stage'].str.strip()

			# Check that the stage variable has been entered correctly
			correct_stages = ['LW','RAW','GRIT','MS','AFBR','DAFMBRMLSS','DAFMBREFF','RAFMBRMLSS','RAFMBREFF']
			stage_warning = \
				'Check "Stage" entry {0} for {1} on dates: {2}. \n ' +\
				'"Stage" should be written as one of the following: \n {3}'
			stage_errors = self.mdata[ ~ self.mdata['Stage'].isin(correct_stages)]
			if len(stage_errors) > 0:
				date_err_prt = \
					[dt.strftime(stage_error,'%m-%d-%y') for stage_error in stage_errors.Date]
				print(
					stage_warning.format(
						stage_errors.Stage.values, ltype, date_err_prt, correct_stages
					)
				)
				sys.exit()
			# Get descriptive stage variable
			self.get_stage_descs()

		# Format and clean other variables
		if ltype == 'COD':

			self.mdata['Type'] = self.mdata['Type'].astype(str)
			self.mdata['Type'] = self.mdata['Type'].str.upper()
			self.mdata['Type'] = self.mdata['Type'].str.strip()
			# Check that the type variable has been entered correctly
			correct_types = ['TOTAL','SOLUBLE']
			type_warning = \
				'Check "Type" entry {0} for {1} on dates: {2}. \n' +\
				'"Type" should be written as on of the following: \n {3}'
			mdata_mod = self.mdata.reset_index(inplace = False)
			type_errors = mdata_mod[ ~ mdata_mod['Type'].isin(correct_types)]
			if len(type_errors) > 0:
				date_err_prt = \
				[dt.strftime(type_error,'%m-%d-%y') for type_error in type_errors.Date]
				print(
					type_warning.format(
						type_errors.Type.values,'COD',date_err_prt, correct_types
					)
				)
				sys.exit()

		if ltype in ['COD','Ammonia','Sulfate']:
			self.set_var_format(ltype, 'Reading (mg/L)', float, 'numeric')

		if ltype == 'PH':
			self.set_var_format(ltype, 'Reading', float, 'numeric')
		
		if ltype == 'ALKALINITY':
			self.set_var_format(ltype,'Sample Volume (mL)', float, 'numeric')
			self.set_var_format(ltype,'Acid Volume (mL, to pH 4.3)', float, 'numeric')
			self.set_var_format(ltype,'Acid Normality (N)', float, 'numeric')

		if ltype in ['COD','ALKALINITY','VFA','Ammonia','Sulfate']:
			self.set_var_format(ltype,'Dilution Factor', float, 'numeric')
		
		if ltype == 'TSS_VSS':
			self.set_var_format(ltype,'Volume (ml)', float, 'numeric')
			self.set_var_format(ltype,'Original (g)', float, 'numeric')
			self.set_var_format(ltype,'Temp105 (g)', float, 'numeric')
			self.set_var_format(ltype,'Temp550 (g)', float, 'numeric')

		if ltype == 'VFA':
			self.set_var_format(ltype,'Acetate (mgCOD/L)', float, 'numeric')
			self.set_var_format(ltype,'Propionate (mgCOD/L)', float, 'numeric')

		if ltype == 'GasComp':
			self.set_var_format(ltype,'Helium pressure (psi) +/- 50 psi', float, 'numeric')
			self.set_var_format(ltype,'Nitrogen (%)', float, 'numeric')
			self.set_var_format(ltype,'Oxygen (%)', float, 'numeric')
			self.set_var_format(ltype,'Methane (%)', float, 'numeric')
			self.set_var_format(ltype,'Carbon Dioxide (%)', float, 'numeric')

		if ltype == 'TKN':	
			varnames = \
			[
				'Sample Volume (mL)','Initial pH','Sample Volume (mL)','Initial pH','End pH',
				'Volume (mL)','Blank Initial pH','Blank End pH','Blank Volume (mL)',
				'NH4Cl Sample Volume (mL)','NH4Cl Initial pH','NH4Cl End pH','NH4Cl Volume (mL)',
				'Tryptophan Sample Volume (mL)','Tryptophan Initial pH','Tryptophan End pH','Tryptophan Volume (mL)',
				'Acid Concentration (N)','NH4Cl Concentration (mg/L)','Tryptophan Concentration (mg/L)'
			]
			for varname in varnames:
				self.set_var_format(ltype, varname, float, 'numeric')		
			
		# Get the obs_id variable (This step also removes duplicates and issues warnings)
		self.manage_dups(ltype, id_vars)
		self.mdata.reset_index(inplace = True)


	# Converts dataset to long 
	def wide_to_long(self, ltype, id_vars, value_vars):

		# Melt the data frame
		df_long = pd.melt(self.mdata, id_vars = id_vars, value_vars = value_vars)
		# Reorder columns
		col_order = ['Date_Time','Stage','variable','units','obs_id','value']
		varnames = ['Date_Time','Stage','Type','units','obs_id','Value']
		df_long = df_long[col_order]
		df_long.columns = varnames

		return df_long

	# Inputs lab testing results data and computes water quality parameters
	def process_data(self):
		
		# Start loop through the gsheets
		for ltype in self.ltype_list:

			self.mdata = cut.get_gsheet_data([ltype])

			if ltype == 'COD':
				self.clean_dataset(ltype,['Date','Stage','Type'])
			elif ltype == 'GasComp':
				self.clean_dataset(ltype,['Date','Helium pressure (psi) +/- 50 psi'])
			else:
				self.clean_dataset(ltype,['Date','Stage'])

			# ======================================= pH ======================================= #
			if ltype == 'PH':
				self.mdata[ltype] = self.mdata['Reading']
				# Get time of sample collection from PH dataset and add to date variable to get single Date + Time variable
				self.mdata['Date_str'] = self.mdata['Date'].dt.strftime('%m-%d-%y')
				self.mdata['Date-Time_str'] = self.mdata.Date_str.str.cat(self.mdata['Time'], sep = ' ')
				self.mdata['Date_Time'] = pd.to_datetime(self.mdata['Date-Time_str'])
				self.mdata['units'] = '-'
				mdata_dt = self.mdata[['Date','Date_Time']]
				mdata_dt.drop_duplicates(inplace = True)

			# ======================================= COD ======================================= #
			if ltype == 'COD':

				# Get actual cod measurement (after correcting for dilution factor)
				self.mdata['act_reading'] = self.mdata['Reading (mg/L)']*self.mdata['Dilution Factor']
				# Recast data
				# Need to dedupe again due to what seems like a bug in pandas code
				self.mdata.drop_duplicates(subset = ['Date','Stage','obs_id','Type'], inplace = True)
				self.mdata.set_index(['Date','Stage','obs_id','Type'], inplace = True)
				mdata_wide = self.mdata.unstack('Type')
				# Create "Total" and "Soluble" variables and compute "Particulate Variable"
				mdata_wide['Total'] = mdata_wide['act_reading']['TOTAL']
				mdata_wide['Soluble'] = mdata_wide['act_reading']['SOLUBLE']
				mdata_wide['Particulate'] = mdata_wide['Total'] - mdata_wide['Soluble']
				# Subset to value variables and convert index to data
				value_vars = ['Soluble','Total','Particulate']
				self.mdata = mdata_wide[value_vars]
				self.mdata.reset_index(inplace = True)
				self.mdata['units'] = 'mg/L'	
				self.mdata.columns = ['Date','Stage','obs_id'] + value_vars + ['units']

			# ======================================= TSS/VSS ======================================= #
			if ltype == 'TSS_VSS':
				# Create TSS and VSS variables
				self.mdata['TSS'] = \
					(self.mdata['Temp105 (g)'] - self.mdata['Original (g)'])/\
					self.mdata['Volume (ml)']*1E6
				self.mdata['VSS'] = \
					self.mdata['TSS'] - \
					(self.mdata['Temp550 (g)'] - self.mdata['Original (g)'])/\
					self.mdata['Volume (ml)']*1E6
				self.mdata['units'] = 'mg/L'
				# Set id and value vars for melting
				value_vars = ['TSS','VSS']

			# ======================================= ALKALINITY ======================================= #
			if ltype == 'ALKALINITY':
				# Compute alkalinity
				self.mdata['ALKALINITY'] = \
				self.mdata['Acid Volume (mL, to pH 4.3)']*self.mdata['Acid Normality (N)']/\
				self.mdata['Sample Volume (mL)']*self.mdata['Dilution Factor']*50*1000
				self.mdata['units'] = 'mg/L as CaCO3'

			# ======================================= VFA =============================================== #
			if ltype == 'VFA':
				# Compute VFA concentrations
				self.mdata['Acetate'] = self.mdata['Acetate (mgCOD/L)']*self.mdata['Dilution Factor']
				self.mdata['Propionate'] = self.mdata['Propionate (mgCOD/L)']*self.mdata['Dilution Factor']
				self.mdata['units'] = 'mgCOD/L'
				# Set value vars for melting
				value_vars = ['Acetate','Propionate']

			# ======================================= Ammonia =============================================== #
			if ltype == 'Ammonia':
				# Compute Ammonia concentration
				self.mdata['Ammonia'] = self.mdata['Reading (mg/L)']*self.mdata['Dilution Factor']
				self.mdata['units'] = 'mg/L'

			if ltype == 'TKN':
				# Compute distillation recovery
				self.mdata['Dist Recovery (%)'] = \
					((self.mdata['NH4Cl Volume (mL)'] - self.mdata['Blank Volume (mL)'])*\
					14.007*self.mdata['Acid Concentration (N)']*1000)/\
					(self.mdata['NH4Cl Concentration (mg/L)']*self.mdata['NH4Cl Sample Volume (mL)'])
				# Compute digestion efficiency
				self.mdata['Digest Eff (%)'] = \
					((self.mdata['Tryptophan Volume (mL)'] - self.mdata['Blank Volume (mL)'])*\
					14.007*self.mdata['Acid Concentration (N)']*1000)/\
					(self.mdata['Tryptophan Concentration (mg/L)']*self.mdata['Tryptophan Sample Volume (mL)'])	
				# Compute corrected TKN value (corrected for distillation recovery and digestion efficiency)
				self.mdata['TKN'] = \
					(((self.mdata['Volume (mL)'] - self.mdata['Blank Volume (mL)'])*\
					14.007*self.mdata['Acid Concentration (N)']*1000)/\
					(self.mdata['Sample Volume (mL)']))/\
					(self.mdata['Dist Recovery (%)']*self.mdata['Digest Eff (%)'])	
				# Set value vars for melting
				self.mdata['units'] = 'mgTKN/L'

			# ======================================= Sulfate =============================================== #
			if ltype == 'Sulfate':
				# Compute Sulfate concentration
				self.mdata['Sulfate'] = self.mdata['Reading (mg/L)']*self.mdata['Dilution Factor']
				self.mdata['units'] = 'mg/L'

			# ======================================= GasComp ============================================ #
			if ltype == 'GasComp':
				self.mdata['Hel_Pressure'] = self.mdata['Helium pressure (psi) +/- 50 psi']
				self.mdata['Stage'] = 'NA'
				self.mdata['units'] = '(see Type)'
				# Set value vars for melting
				value_vars = ['Hel_Pressure (psi)','Nitrogen (%)','Oxygen (%)','Methane (%)','Carbon Dioxide (%)']

			# Add Sample Date-Time variable from PH
			if ltype != 'PH':
				self.mdata = self.mdata.merge(mdata_dt, on = 'Date')

			# Convert to long format
			id_vars = ['Date_Time','Stage','obs_id','units']
			if ltype in ['PH','ALKALINITY','Ammonia','TKN','Sulfate']:
				value_vars = [ltype]
			mdata_long = self.wide_to_long(ltype, id_vars, value_vars)
			# Create key unique by Date_Time, Stage, Type, and obs_id
			mdata_long['Dkey'] = \
				mdata_long['Date_Time'].astype(str) + mdata_long['Stage'] + mdata_long['Type'] + mdata_long['obs_id'].astype(str)
			# Reorder columns to put DKey as first column
			colnames = list(mdata_long.columns.values)
			mdata_long = mdata_long[colnames[-1:] + colnames[0:-1]]

			# Load data to SQL
			# SQL command strings for sqlite3
			colNTypeStr = 'Date_Time INT, Stage TEXT, Type TEXT, units TEXT, obs_id INT, Value REAL'
			colNStr = ','.join(mdata_long.columns.values)
			colIns = ','.join(['?']*len(mdata_long.columns))
			create_str = """
				CREATE TABLE IF NOT EXISTS {0} (DKey INT PRIMARY KEY, {1})
			""".format(ltype,colNTypeStr)
			ins_str = """
				INSERT OR REPLACE INTO {0} ({1})
				VALUES ({2})
			""".format(ltype,colNStr,colIns)
			# Set connection to SQL database (pertaining to given year)
			os.chdir(self.data_dir)
			conn = sqlite3.connect('cr2c_lab_data.db')
			# Load data to SQL
			# Create the table if it doesn't exist
			conn.execute(create_str)
			# Insert aggregated values for the elid and time period
			conn.executemany(
				ins_str,
				mdata_long.to_records(index = False).tolist()
			)
			conn.commit()
			# Close Connection
			conn.close()

		# Sets the start and end dates for the charts, depending on user input
	def manage_chart_dates(self, start_dt_str, end_dt_str):

		if start_dt_str == None:
			start_dt = self.min_feas_dt	
		else:
			start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
		
		if end_dt_str == None:
			end_dt = self.file_dt
		else:
			end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

		return start_dt, end_dt
	

	def get_lab_plots(
		self,
		start_dt_str,
		end_dt_str,
		mplot_list,
		wrap_var,
		stage_sub = None,
		type_sub = None,
		outdir = None,
		opfile_suff = None
	):

		if not outdir:
			# Request tables and charts output directory from user
			tkTitle = 'Directory to output charts to...'
			print(tkTitle)
			outdir = askdirectory(title = tkTitle)
		try:
			os.chdir(outdir)
		except OSError:
			print('Please choose a valid directory to output the charts to')
			sys.exit()

		if opfile_suff:
			opfile_suff = '_' + opfile_suff
		else:
			opfile_suff = ''

		# Clean case of mplot_list and wrap var inputs
		mplot_list = [element.upper() for element in mplot_list]
		wrap_var = wrap_var[0].upper() + wrap_var[1:].lower()

		# Order of treatment stages in plots
		stage_order = [
			'Raw Influent',
			'Grit Tank',
			'Microscreen',
			'AFBR',
			'Duty AFMBR MLSS',
			'Duty AFMBR Effluent',
			'Research AFMBR MLSS',
			'Research AFMBR Effluent'
		]	

		# Manage dates given by user
		start_dt, end_dt = self.manage_chart_dates(start_dt_str, end_dt_str)

		# Get all of the lab data requested
		mdata_all = get_data(mplot_list, start_dt_str = start_dt_str, end_dt_str = end_dt_str)

		# Loop through the lab data types
		for ltype in mplot_list:

			if ltype.find('TSS') >= 0 or ltype.find('VSS') >= 0:
				ltype = 'TSS_VSS'

			mdata_long = mdata_all[ltype]
			# ID variables for grouping by day 
			# (for monitoring types that might have multiple observations in a day)
			id_vars_chrt = ['Date_Time','Stage','Type']
			
			if ltype == 'COD':
				# Set plotting variables
				ylabel = 'COD Reading (mg/L)'
				type_list = ['Total','Soluble','Particulate']
				share_yax = False

			if ltype == 'TSS_VSS':

				# Set plotting variables
				ylabel = 'Suspended Solids (mg/L)'
				type_list = ['TSS','VSS']
				share_yax = True

			if ltype == 'PH':

				# Set plotting variables
				ylabel = 'pH'
				mdata_long['Type'] = 'pH'
				type_list = ['pH']
				share_yax = True

			if ltype == 'ALKALINITY':

				# Set plotting variables
				ylabel = 'Alkalinity (mg/L as ' + r'$CaCO_3$)'
				mdata_long['Type'] = 'Alkalinity'
				type_list = ['Alkalinity']
				share_yax = True

			if ltype == 'VFA':

				# Set plotting variables
				ylabel = 'VFAs as mgCOD/L'
				type_list = ['Acetate','Propionate']
				share_yax = False

			if ltype == 'AMMONIA':

				#Set plotting variables
				ylabel = r'$NH_3$' + ' (mg/L as N)'
				mdata_long['Type'] = 'Ammonia'
				type_list = ['Ammonia']
				share_yax = True

			if ltype == 'TKN':

				# Set plotting variables
				ylabel = 'mgTKN/L'
				mdata_long['Type'] = 'TKN'
				type_list = ['TKN']
				share_yax = True

			if ltype == 'SULFATE':

				# Set plotting variables
				ylabel = 'mg/L ' + r'$SO_4$'
				mdata_long['Type'] = 'Sulfate'
				type_list = ['Sulfate']
				share_yax = True

			# Filter to the dates desired for the plots
			mdata_chart = mdata_long.loc[
				(mdata_long.Date_Time >= start_dt) &
				(mdata_long.Date_Time < end_dt + timedelta(days = 1)) 
			]

			# Filter to stages and types being subset to
			if stage_sub:
				mdata_chart = mdata_chart.loc[mdata_chart.Stage.isin(stage_sub)]
			if type_sub:
				mdata_chart = mdata_chart.loc[mdata_chart.Type.isin(type_sub)]

			# Get the stages for which there are data
			act_stages = mdata_chart.Stage.values
			# Reproduce stage order according to data availability
			stage_list = [stage for stage in stage_order if stage in act_stages]

			if wrap_var == 'Stage':
				wrap_list = stage_list
				hue_list  = type_list
				hue_var = 'Type'
			elif wrap_var == 'Type':
				wrap_list = type_list
				hue_list  = stage_list
				hue_var = 'Stage'
			else:
				print('wrap_var can only be "Stage" or "Type"')
				sys.exit()

			# Set plot width and length according to the wrapping variable	
			plot_wid = 5*min(3,len(wrap_list))
			wrap_wid = min(3,len(wrap_list))
			plot_len = 6*np.ceil(len(wrap_list)/3) + 5

			# Average all observations (by type and stage) taken on a day
			mdata_chart = mdata_chart.groupby(id_vars_chrt).mean()

			# Remove index!
			mdata_chart.reset_index(inplace = True)

			# Set plot facetting and layout
			mplot = sns.FacetGrid(
				mdata_chart,
				col = wrap_var,
				col_order = wrap_list,
				col_wrap = wrap_wid,
				hue = hue_var,
				hue_order = hue_list,
				sharey = share_yax
			)

			# Set date format
			dfmt = dates.DateFormatter('%m/%d/%y')
			# Set tickmarks for days of the month
			dlocator = dates.DayLocator(bymonthday = [1,15])		
			# Format the axes in the plot panel
			for ax in mplot.axes.flatten():
			    ax.xaxis.set_major_locator(dlocator)
			    ax.xaxis.set_major_formatter(dfmt)
			    # Different format for PH vs other y-axes
			    if ltype == 'PH':
			    	tkr.FormatStrFormatter('%0.2f')
			    else:
				    ax.yaxis.set_major_formatter(
				    	tkr.FuncFormatter(lambda x, p: format(int(x), ','))
				    )

			# Plot values and set axis labels/formatting
			mplot.map(plt.plot,'Date_Time','Value', linestyle = '-', marker = "o", ms = 4)
			mplot.set_titles('{col_name}')
			mplot.set_ylabels(ylabel)
			mplot.set_xlabels('')
			mplot.set_xticklabels(rotation = 45)

			# Output plot to given directory
			plot_filename = "{0}{1}.png"
			os.chdir(outdir)

			# Add and position the legend
			if ltype in ['PH','ALKALINITY'] and wrap_var == 'Stage':
				plt.savefig(
					plot_filename.format(ltype, opfile_suff), 
					bbox_inches = 'tight',
					width = plot_wid, 
					height = plot_len
				)
				plt.close()
			else:
				handles, labels = ax.get_legend_handles_labels()
				lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor = (1,0.75))
				plt.savefig(
					plot_filename.format(ltype, opfile_suff), 
					bbox_extra_artists = (lgd,),
					bbox_inches = 'tight',
					width = plot_wid, 
					height = plot_len
				)
				plt.close()


	def long_to_wide(self, df, id_vars):

		# Create descriptive Date/Time Variable
		df.rename(columns = {'Date_Time' : 'Sample Date & Time'}, inplace = True)
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
	def clean_wide_table(self, dfwide, value_vars, start_dt, end_dt, add_time_el):

		stage_order = \
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
		

		# First retrieve the stages for which there are data
		act_stages = dfwide.columns.levels[1].values
		# Reproduce stage order according to data availability
		act_st_ord = [stage for stage in stage_order if stage in act_stages]

		# Truncate (adding exception for Ammonia with no type variable)
		if value_vars == ['Value']:
			column_tuple = act_st_ord
		else:
			column_tuple = (act_st_ord, value_vars)

		df_trunc = dfwide.Value.loc[start_dt:end_dt, column_tuple]

		# Set column order (again, exception is for Ammonia with no type variable)
		if value_vars == ['Value']:
			df_trunc = df_trunc.reindex_axis(act_st_ord, axis = 1, level = None)
		else:
			df_trunc = df_trunc.reindex_axis(act_st_ord, axis = 1, level = 'Stage')
			df_trunc = df_trunc.reindex_axis(value_vars, axis = 1, level = 'Type')

		# Create days since seed variable and insert as the first column
		if add_time_el == 1:
			seed_dt = dt.strptime('5-11-17','%m-%d-%y')
			days_since_seed = np.array((df_trunc.index - seed_dt).days)
			df_trunc.insert(0, 'Days Since Seed', days_since_seed)

		return df_trunc


	# Gets wide dataset, cleans and formats and outputs to csv
	def summarize_tables(self, end_dt_str, ndays, add_time_el = True, outdir = None, opfile_suff = None):
		
		if not outdir:
			tkTitle = 'Directory to output tables to...'
			outdir = askdirectory(title = tkTitle)
		try:
			os.chdir(outdir)
		except OSError:
			print('Please choose a valid directory to output the tables to')
			sys.exit()

		if opfile_suff:
			opfile_suff = '_' + opfile_suff
		else:
			opfile_suff = ''

		# Get start and end dates
		end_dt = dt.strptime(end_dt_str,'%m-%d-%y') + timedelta(days = 1)
		start_dt = end_dt - timedelta(days = ndays)
		seed_dt = dt.strptime('05-10-17','%m-%d-%y')

		# Load data from SQL
		mdata_all = get_data(['COD','TSS_VSS','ALKALINITY','PH','VFA','Ammonia','Sulfate'])

		# Specify id variables (same for every type since combining Alkalinity and pH)
		id_vars = ['Sample Date & Time','Stage','Type','obs_id']

		# For Alkalinity, pH, NH3, and SO4, need to add Type variable back in
		ALK = mdata_all['ALKALINITY']
		ALK['Type'] = 'Alkalinity'
		PH = mdata_all['PH']
		PH['Type'] = 'pH'
		NH3 = mdata_all['Ammonia']
		SO4 = mdata_all['Sulfate']

		# Concatenate Alkaliity and pH and reset index
		ALK_PH = pd.concat([PH,ALK], axis = 0, join = 'outer').reset_index(drop = True)

		# Get wide data
		CODwide = self.long_to_wide(mdata_all['COD'], id_vars)
		VFAwide = self.long_to_wide(mdata_all['VFA'], id_vars)
		TSS_VSSwide = self.long_to_wide(mdata_all['TSS_VSS'], id_vars)
		ALK_PHwide = self.long_to_wide(ALK_PH, id_vars)
		NH3wide = self.long_to_wide(NH3, ['Sample Date & Time','Stage'])
		SO4wide = self.long_to_wide(SO4, ['Sample Date & Time','Stage'])
		
		# Truncate and set column order
		CODtrunc = self.clean_wide_table(CODwide, ['Total','Soluble'], start_dt, end_dt, add_time_el)
		VFAtrunc = self.clean_wide_table(VFAwide, ['Acetate','Propionate'], start_dt, end_dt, add_time_el)
		TSS_VSStrunc = self.clean_wide_table(TSS_VSSwide,['TSS','VSS'], start_dt, end_dt, add_time_el)
		ALK_PHtrunc = self.clean_wide_table(ALK_PHwide,['pH','Alkalinity'], start_dt, end_dt, add_time_el)
		NH3trunc = self.clean_wide_table(NH3wide,['Value'], start_dt, end_dt, add_time_el)
		SO4trunc = self.clean_wide_table(SO4wide,['Value'], start_dt, end_dt, add_time_el)
		
		# Save
		os.chdir(outdir)
		CODtrunc.to_csv('COD_table' + end_dt_str + opfile_suff + '.csv')
		VFAtrunc.to_csv('VFA_table' + end_dt_str + opfile_suff + '.csv')
		TSS_VSStrunc.to_csv('TSS_VSS_table' + end_dt_str + opfile_suff + '.csv')
		ALK_PHtrunc.to_csv('ALK_PH_table' + end_dt_str + opfile_suff + '.csv')
		NH3trunc.to_csv('Ammonia_table' + end_dt_str + opfile_suff + '.csv')
		SO4trunc.to_csv('Sulfate_table' + end_dt_str + opfile_suff + '.csv')

