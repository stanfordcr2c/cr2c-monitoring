
'''
	Script loads data from lab tests, computes water quality parameters,
	and loads the data to an SQL database (no inputs required)
'''

# Data Prep
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime as dt
from datetime import timedelta

#Google API
from google.cloud import bigquery

# Utilities
import functools
import warnings
import os
from os.path import expanduser
import sys

# CR2C
import cr2c_utils as cut


# Wrapper for querying data from the cr2c_labdata.db data store
def get_data(
	ltypes, 
	start_dt_str = None, end_dt_str = None, output_csv = False, outdir = None
):

	# Convert date string inputs to dt variables
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

	# Load data from SQL
	data_dir = cut.get_dirs()[0]
	os.chdir(data_dir)
	conn = sqlite3.connect('cr2c_labdata.db')

	# Loop through types of lab data types (ltypes)
	ldata_all = {}
	for ltype in ltypes:

		# Clean user input wrt TSS_VSS
		if ltype.find('TSS') >= 0 or ltype.find('VSS') >= 0:
			ltype = 'TSS_VSS'

		# ldata_long = pd.read_sql(
		# 	'SELECT * FROM {0}'.format(ltype), 
		# 	conn, 
		# 	coerce_float = True
		# )
		projectid = 'cr2c-monitoring'
		dataset_id = 'test_dataset'
		ldata_long = pd.read_gbq('SELECT * FROM {}.{}'.format(dataset_id, ltype), projectid)

		# Dedupe data (just in case, pandas can be glitchy with duplicates)
		ldata_long.drop_duplicates(inplace = True)
		# Convert Date_Time variable to a pd datetime and eliminate missing values
		ldata_long.loc[:,'Date_Time'] = pd.to_datetime(ldata_long['Date_Time'])
		ldata_long.dropna(subset = ['Date_Time'], inplace = True)
		# Filter to desired dates
		# ldata_long.drop('Dkey', axis = 1, inplace = True)
		if start_dt_str:
			ldata_long = ldata_long.loc[ldata_long['Date_Time'] >= start_dt,:]
		if end_dt_str:
			ldata_long = ldata_long.loc[ldata_long['Date_Time'] <= end_dt + timedelta(days = 1),:]
		
		# Output csv if desired
		if output_csv:
			os.chdir(outdir)
			ldata_long.to_csv(ltype + '.csv', index = False, encoding = 'utf-8')

		# Write to dictionary
		ldata_all[ltype] = ldata_long

	return ldata_all


# Main lab data class (where all processing/plotting occurs)
class labrun:
	
	def __init__(self, verbose = False):
		
		self.ltype_list = \
			['PH','COD','TSS_VSS','ALKALINITY','VFA','GASCOMP','AMMONIA','SULFATE','TKN','BOD']
		self.min_feas_dt = dt.strptime('6-1-16', '%m-%d-%y')
		self.file_dt = dt.now()
		self.file_dt_str = dt.strftime(self.file_dt,'%m-%d-%y')
		self.data_dir, self.pydir = cut.get_dirs()
		self.log_dir = os.path.join(self.data_dir,'Logs')
		self.verbose = verbose


	# Gets a more descriptive value for each of the treatment stages entered by operators into the lab data gsheet (as the variable "Stage").
	def get_stage_descs(self):

		conditions = [
			self.ldata['Stage'] == 'DAFMBREFF',
			self.ldata['Stage'] == 'DAFMBRMLSS',
			self.ldata['Stage'] == 'RAFMBREFF',
			self.ldata['Stage'] == 'RAFMBRMLSS',
			self.ldata['Stage'] == 'RAW',
			self.ldata['Stage'] == 'GRIT',
			self.ldata['Stage'] == 'MS',
			self.ldata['Stage'] == 'MESH',
			self.ldata['Stage'] == 'LW',
			self.ldata['Stage'] == 'BLANK',
			self.ldata['Stage'] == 'STD',
		]
		choices = [
			'Duty AFMBR Effluent','Duty AFMBR MLSS',
			'Research AFMBR Effluent','Research AFMBR MLSS',
			'Raw Influent','Grit Tank','Microscreen','MESH','Lake Water','Blank','Standard'
		]
		self.ldata.loc[:,'Stage'] = np.select(conditions, choices, default = self.ldata['Stage'])


	# Manages duplicate observations by removing duplicates (with warnings)
	# gets observation id's for readings with duplicate Date-Stage values (if intentionally taking multiple readings)  
	def manage_dups(self, ltype, id_vars):

		# Set duplicates warning
		dup_warning = \
			'There are repeat entries and/or entries with no date in {0} that were removed. '+\
	        'A csv of the removed values has been saved as {1}'

		# Check for Duplicates and empties
		repeat_entries = np.where(self.ldata.duplicated())[0].tolist()
		blank_entries = np.where(pd.isnull(self.ldata.Date))[0].tolist()
		repeat_entries.extend(blank_entries)

	    # If found, remove, print warning and output csv of duplicates/empties
		if self.verbose and len(repeat_entries) > 0:
			os.chdir(self.log_dir)
			dup_filename = ltype + 'duplicates' + self.file_dt_str
			warnings.warn(dup_warning.format(ltype,dup_filename + '.csv'))
			self.ldata.iloc[repeat_entries].to_csv(dup_filename + '.csv')
		
		# Eliminate duplicate data entries and reset the index
		self.ldata.drop_duplicates(keep = 'first', inplace = True)
		self.ldata.reset_index(drop = True, inplace = True)
		
		# Sort the dataset
		self.ldata.sort_values(id_vars, inplace = True)

		# Create a list of observation ids (counting from 0)
		obs_ids = [0]
		for obs_no in range(1,len(self.ldata)):
			row_curr = [self.ldata[id_var][obs_no] for id_var in id_vars]
			row_prev = [self.ldata[id_var][obs_no - 1] for id_var in id_vars]
			obs_id_curr = obs_ids[-1]
			if row_curr == row_prev:
				# Same date/stage/type, increment obs_id
				obs_ids.append(obs_id_curr + 1)
			else:
				# Different date/stage/type, restart obs_id count
				obs_ids.append(0)

		#Add obs_id variable to dataset
		self.ldata.loc[:,'obs_id'] = obs_ids


	# Tries to format a raw input variable from gsheets data as a datetime or a floating number; 
	# outputs error message if the input data cannot be re-formatted
	def set_var_format(self, ltype, variable, var_format):

		if variable == 'Date':
			format_desc = 'm-d-yy'
		else:
			format_desc = 'numeric'

		var_typ_warn = \
			'cr2c_labdata: set_var_format: Check {0} variable in {1}. An entry is incorrect, format should be {2}'

		self.ldata = self.ldata.apply(
			lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan
		)
		try:
			if variable == 'Date':
				self.ldata.loc[:,'Date'] = pd.to_datetime(self.ldata['Date'], format = '%m-%d-%y')
			else:
				self.ldata.loc[:,variable] = self.ldata[variable].astype(var_format)
		except TypeError:
			print(var_typ_warn.format(variable, ltype, format_desc))
			sys.exit()
		except ValueError:
			print(var_typ_warn.format(variable, ltype, format_desc))
			sys.exit()


	# Cleans the raw gsheets lab data by reformatting variables and removing duplicates (or tagging intentional duplicates)
	def clean_dataset(self, ltype, id_vars):

		# First sort the data frame by its id variables
		self.ldata.sort_values(id_vars, inplace = True)
		self.set_var_format(ltype, 'Date', None)
		# Eliminate missing date variables
		self.ldata.dropna(subset = ['Date'], inplace = True)
		# Make sure all date variables are within a reasonable range
		date_rng_warn = \
			'cr2c_labdata: clean_dataset: A Date variable in {0} has been entered incorrectly as {1} and removed'
		if self.ldata.Date.min() < self.min_feas_dt:
			print(date_rng_warn.format(ltype,self.ldata.Date.min()))
		if self.ldata.Date.max() > self.file_dt:
			print(date_rng_warn.format(ltype,self.ldata.Date.max()))
		# Filter dates accordingly
		self.ldata = self.ldata.loc[
			(self.ldata.Date >= self.min_feas_dt) &
			(self.ldata.Date <= self.file_dt)
		].copy()

		# Format and clean stage variable
		if ltype != 'GASCOMP':

			self.ldata.loc[:,'Stage'] = self.ldata['Stage'].astype(str)
			self.ldata.loc[:,'Stage'] = self.ldata['Stage'].str.upper()
			self.ldata.loc[:,'Stage'] = self.ldata['Stage'].str.strip()

			# Check that the stage variable has been entered correctly
			# Number of composite samplers on site
			CSNos = [str(el + 1) for el in list(range(24))]
			# Correct stage abbreviations
			correct_stages = \
				['BLANK','STD','LW','RAW','GRIT','MS','AFBR','DAFMBRMLSS','DAFMBREFF','RAFMBRMLSS','RAFMBREFF','MESH'] 
			# Add stages particular to a composite sampler
			correct_stages_all = \
				correct_stages +\
				[stage + ' CS' + CSNo for CSNo in CSNos for stage in correct_stages]
			# Warning to print if an incorrect stage is found
			stage_warning = \
				'cr2c_labdata: clean_dataset: Check "Stage" entry {0} for {1} on dates: {2}. \n ' +\
				'"Stage" should be written as one of the following: \n {3} (or "[STAGE]" + " CS[#]" if logging a sample from a specific composite sampler)'
			stage_errors = self.ldata[ ~ self.ldata['Stage'].isin(correct_stages_all)]
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

			self.ldata.loc[:,'Type'] = self.ldata['Type'].astype(str)
			self.ldata.loc[:,'Type'] = self.ldata['Type'].str.upper()
			self.ldata.loc[:,'Type'] = self.ldata['Type'].str.strip()
			# Check that the type variable has been entered correctly
			correct_types = ['TOTAL','SOLUBLE']
			type_warning = \
				'cr2c_labdata: clean_dataset: Check "Type" entry {0} for {1} on dates: {2}. \n' +\
				'"Type" should be written as on of the following: \n {3}'
			ldata_mod = self.ldata.reset_index(inplace = False).copy()
			type_errors = ldata_mod.loc[ ~ ldata_mod['Type'].isin(correct_types),:]
			if len(type_errors) > 0:
				date_err_prt = \
				[dt.strftime(type_error,'%m-%d-%y') for type_error in type_errors.Date]
				print(
					type_warning.format(
						type_errors.Type.values,'COD',date_err_prt, correct_types
					)
				)
				sys.exit()

		if ltype in ['COD','AMMONIA','SULFATE']:
			self.set_var_format(ltype, 'Reading (mg/L)', float)

		if ltype == 'BOD':
			self.set_var_format(ltype, 'Sample Volume (mL)', float)
			self.set_var_format(ltype, 'Initial DO (mg/L)'	, float)
			self.set_var_format(ltype, 'Day 5 DO (mg/L)', float)

		if ltype == 'PH':
			self.set_var_format(ltype, 'Reading', float)
		
		if ltype == 'ALKALINITY':
			self.set_var_format(ltype,'Sample Volume (mL)', float)
			self.set_var_format(ltype,'Acid Volume (mL, to pH 4.3)', float)
			self.set_var_format(ltype,'Acid Normality (N)', float)

		if ltype in ['COD','ALKALINITY','VFA','AMMONIA','SULFATE']:
			self.set_var_format(ltype,'Dilution Factor', float)
		
		if ltype == 'TSS_VSS':
			self.set_var_format(ltype,'Volume (ml)', float)
			self.set_var_format(ltype,'Original (g)', float)
			self.set_var_format(ltype,'Temp105 (g)', float)
			self.set_var_format(ltype,'Temp550 (g)', float)

		if ltype == 'VFA':
			self.set_var_format(ltype,'Acetate (mgCOD/L)', float)
			self.set_var_format(ltype,'Propionate (mgCOD/L)', float)

		if ltype == 'GASCOMP':
			self.set_var_format(ltype,'Helium pressure (psi) +/- 50 psi', float)
			self.set_var_format(ltype,'Nitrogen (%)', float)
			self.set_var_format(ltype,'Oxygen (%)', float)
			self.set_var_format(ltype,'Methane (%)', float)
			self.set_var_format(ltype,'Carbon Dioxide (%)', float)

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
				self.set_var_format(ltype, varname, float)		
			
		# Get the obs_id variable (This step also removes duplicates and issues warnings)
		self.manage_dups(ltype, id_vars)
		self.ldata.reset_index(inplace = True)


	# Performs a pandas melt procedure on the lab data with the right column ordering
	def wide_to_long(self, ltype, id_vars, value_vars):

		# Melt the data frame
		df_long = pd.melt(self.ldata, id_vars = id_vars, value_vars = value_vars)

		# Add Type to "variable" variable if BOD (OD days, specifically)
		if ltype == 'BOD':
			df_long.reset_index(inplace = True)
			df_long.loc[:,'variable'] = df_long['Type'] + ': ' + df_long['variable']
		# Reorder columns
		col_order = ['Date_Time','Stage','variable','units','obs_id','value']
		varnames = ['Date_Time','Stage','Type','units','obs_id','Value']
		df_long = df_long[col_order]
		df_long.columns = varnames

		return df_long

	# Performs sequential pandas unstack procedures to convert a long dataframe to a wide dataframe
	def long_to_wide(self, df, id_vars):

		# Create descriptive Date/Time Variable
		df.rename(columns = {'Date_Time' : 'Sample Date & Time'}, inplace = True)
		all_vars = id_vars + ['Value']
		df = df[all_vars].copy()

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

	# Counts the characters in a string that appear more than once and outputs a string of these characters	
	def count_multichars(self,string):
		chars = list(set([char for char in string if string.count(char) > 1]))
		return ''.join(chars)


	# Cleans the wide table of lab data results
	def clean_wide_table(self, dfwide, value_vars, start_dt, end_dt, add_time_el):

		stage_order = \
			[
				'Raw Influent',
				'Grit Tank',
				'Microscreen',
				'MESH',
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
			df_trunc = df_trunc.reindex(act_st_ord, axis = 1, level = None)
		else:
			df_trunc = df_trunc.reindex(act_st_ord, axis = 1, level = 'Stage')
			df_trunc = df_trunc.reindex(value_vars, axis = 1, level = 'Type')

		# Create days since seed variable and insert as the first column
		if add_time_el:
			seed_dt = dt.strptime('5-11-17','%m-%d-%y')
			days_since_seed = np.array((df_trunc.index - seed_dt).days)
			df_trunc.insert(0, 'Days Since Seed', days_since_seed)

		return df_trunc


	# Combines and filters a set of wide tables and outputs the resulting table as a csv file
	def summarize_tables(self, end_dt_str, ndays, add_time_el = True, outdir = None, opfile_suff = None):

		if opfile_suff:
			opfile_suff = '_' + opfile_suff
		else:
			opfile_suff = ''

		# Get start and end dates
		end_dt = dt.strptime(end_dt_str,'%m-%d-%y') + timedelta(days = 1)
		start_dt = end_dt - timedelta(days = ndays)
		seed_dt = dt.strptime('05-10-17','%m-%d-%y')

		# Load data from SQL
		ldata_all = get_data(['COD','TSS_VSS','ALKALINITY','PH','VFA','AMMONIA','SULFATE'])

		# Specify id variables (same for every type since combining Alkalinity and pH)
		id_vars = ['Sample Date & Time','Stage','Type','obs_id']

		# For Alkalinity, pH, NH3, and SO4, need to add Type variable back in
		ALK = ldata_all['ALKALINITY'].copy()
		ALK.loc[:,'Type'] = 'Alkalinity'
		PH = ldata_all['PH'].copy()
		PH.loc[:,'Type'] = 'pH'
		NH3 = ldata_all['AMMONIA'].copy()
		SO4 = ldata_all['SULFATE'].copy()

		# Concatenate Alkaliity and pH and reset index
		ALK_PH = pd.concat([PH,ALK], axis = 0, join = 'outer').reset_index(drop = True)

		# Get wide data
		CODwide = self.long_to_wide(ldata_all['COD'].copy(), id_vars)
		VFAwide = self.long_to_wide(ldata_all['VFA'].copy(), id_vars)
		TSS_VSSwide = self.long_to_wide(ldata_all['TSS_VSS'].copy(), id_vars)
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


	# The main caller that executes all methods to read data from Google Sheets, clean and reformat it. 
	# Performs necessary computations on laboratory results, converts all data to a long format, and outputs the result to the cr2c_labdata.db data store.
	def process_data(self):
		
		# Start loop through the gsheets
		for ltype in self.ltype_list:

			self.ldata = cut.get_gsheet_data(ltype)

			if ltype == 'COD':
				self.clean_dataset(ltype,['Date','Stage','Type'])
			elif ltype == 'BOD':
				self.clean_dataset(ltype,['Date','Stage'])
			elif ltype == 'GASCOMP':
				self.clean_dataset(ltype,['Date','Helium pressure (psi) +/- 50 psi'])
			else:
				self.clean_dataset(ltype,['Date','Stage'])

			# ID variables for reshaping data
			id_vars = ['Date_Time','Stage','obs_id','units']

			# ======================================= pH ======================================= #
			if ltype == 'PH':
				self.ldata.loc[:,ltype] = self.ldata['Reading']
				# Get time of sample collection from PH dataset and add to date variable to get single Date + Time variable
				self.ldata.loc[:,'Date_str'] = self.ldata['Date'].dt.strftime('%m-%d-%y')
				self.ldata.loc[:,'Date-Time_str'] = self.ldata.Date_str.str.cat(self.ldata['Time'], sep = ' ')
				self.ldata.loc[:,'Date_Time'] = pd.to_datetime(self.ldata['Date-Time_str'])
				self.ldata.loc[:,'units'] = '-'
				ldata_dt = self.ldata.loc[:,['Date','Date_Time']].copy()
				ldata_dt.drop_duplicates(inplace = True)

			# ======================================= COD ======================================= #
			if ltype == 'COD':

				# Get actual cod measurement (after correcting for dilution factor)
				self.ldata.loc[:,'act_reading'] = self.ldata['Reading (mg/L)']*self.ldata['Dilution Factor']
				# Recast data
				# Need to dedupe again due to what seems like a bug in pandas code
				self.ldata.drop_duplicates(subset = ['Date','Stage','obs_id','Type'], inplace = True)
				self.ldata.set_index(['Date','Stage','obs_id','Type'], inplace = True)
				ldata_wide = self.ldata.unstack('Type')
				# Create "Total" and "Soluble" variables and compute "Particulate Variable"
				ldata_wide.loc[:,'Total'] = ldata_wide['act_reading']['TOTAL']
				ldata_wide.loc[:,'Soluble'] = ldata_wide['act_reading']['SOLUBLE']
				ldata_wide.loc[:,'Particulate'] = ldata_wide['Total'] - ldata_wide['Soluble']
				# Subset to value variables and convert index to data
				value_vars = ['Soluble','Total','Particulate']
				self.ldata = ldata_wide[value_vars].copy()
				self.ldata.reset_index(inplace = True)
				self.ldata.loc[:,'units'] = 'mg/L'	
				self.ldata = self.ldata[['Date','Stage','obs_id'] + value_vars + ['units']].copy()
				# Rename columns
				self.ldata.columns = ['Date','Stage','obs_id'] + value_vars + ['units']

			if ltype == 'BOD':

				self.ldata.drop_duplicates(subset = ['Date','Stage','obs_id'], inplace = True)
				self.ldata.set_index(['Date','Stage','obs_id'], inplace = True)

				# Get the blank Oxygen Demand used for the given day
				ldata_wide = self.ldata.unstack('Stage')
				ldata_wide.loc[:,'Blank OD'] = \
					ldata_wide['Initial DO (mg/L)']['Blank'] - \
					ldata_wide['Day 5 DO (mg/L)']['Blank']
				ldata_wide.reset_index(inplace = True)
				blank_od = pd.DataFrame(
					{
						'Date': ldata_wide['Date'].values,
						'obs_id': ldata_wide['obs_id'].values,
						'Blank OD': ldata_wide['Blank OD'].values
					}
				)
				blank_od.dropna(inplace = True)

				# Remove outlying blank OD values
				blank_od_exp = blank_od.merge(blank_od, on = 'Date', how = 'outer')
				blank_od_exp = blank_od_exp.loc[blank_od_exp['obs_id_x'] != blank_od_exp['obs_id_y'],:]
				blank_od_exp.sort_index(axis = 1, ascending = True, inplace = True)
				# Set the threshold beyond which the -log of the ratio of two OD readings is considered abnormal
				diffLim = abs(np.log(1/2))
				blank_od_exp.loc[:,'diff'] = abs(np.log(blank_od_exp['Blank OD_x']/blank_od_exp['Blank OD_y']))
				# Filter to ids that have a pairwise difference greater than the threshold
				diff_ids = blank_od_exp.loc[blank_od_exp['diff'] > diffLim,['Date','obs_id_x']].copy()
				# Retrieve the ids associated with the outlier
				diff_ids_udate = \
					diff_ids.groupby('Date').apply(lambda group: ''.join([str(row) for row in group['obs_id_x'].values]))
				# Get the ID that is the common culprit
				outlying_ids = \
					diff_ids_udate.apply(lambda row: self.count_multichars(row))
				# Create data frame of culprit IDs and their corresponding dates (convoluted, but pandas is too complicated...)
				outlying_ids = pd.DataFrame({'Date': diff_ids_udate.index.values, 'outlying_id': outlying_ids.values})
				# Merge back onto the original data
				blank_od = blank_od.merge(outlying_ids, on = 'Date', how = 'outer')
				# Filter out outlying values!
				blank_od.loc[:,'keep'] = blank_od.apply(lambda row: str(row['obs_id']) not in str(row['outlying_id']), axis = 1)
				blank_od_means = blank_od.loc[blank_od['keep'],['Date','Blank OD']].groupby('Date').mean()
				blank_od_means.columns = ['Blank OD']
				blank_od_means.reset_index(inplace = True)

				# Calculate BOD values by comparing to blank
				self.ldata.reset_index(inplace = True)
				self.ldata = self.ldata.merge(blank_od_means, on = 'Date', how = 'outer')
				self.ldata = self.ldata.loc[self.ldata['Stage'] != 'Blank',:]
				
				self.ldata.loc[:,'Adjustment Factor'] = 1 - self.ldata['Sample Volume (mL)']/300
				self.ldata.loc[:,'BOD 5'] = \
					(
						(self.ldata['Initial DO (mg/L)'] - self.ldata['Day 5 DO (mg/L)']) -\
						self.ldata['Blank OD']*self.ldata['Adjustment Factor']
					)/\
					(self.ldata['Sample Volume (mL)']/300)
				self.ldata.loc[:,'BOD U'] = self.ldata['BOD 5']/(1 - np.exp(-0.23*5))

				BODMeans = \
					self.ldata.groupby(['Date','Stage']).mean()[['BOD 5','BOD U']]
				BODSDs = \
					self.ldata.groupby(['Date','Stage']).std()[['BOD 5','BOD U']]

				BODMeans.reset_index(inplace = True)
				BODSDs.reset_index(inplace = True)
				BODSDs = pd.melt(BODSDs, id_vars = ['Date','Stage'], value_vars = ['BOD 5','BOD U'])

				self.ldata = pd.melt(BODMeans, id_vars = ['Date','Stage'], value_vars = ['BOD 5','BOD U'])
				self.ldata = self.ldata.merge(BODSDs, on = ['Date','Stage','variable'], how = 'outer')


				self.ldata.loc[:,'Type'] = self.ldata['variable']
				self.ldata.loc[:,'Value'] = self.ldata['value_x']
				self.ldata.loc[:,'Min Value'] = self.ldata['Value'] - 1.96*self.ldata['value_y']
				self.ldata.loc[:,'Max Value'] = self.ldata['Value'] + 1.96*self.ldata['value_y']
				self.ldata.loc[:,'units'] = 'mg/L'
				self.ldata = self.ldata.loc[:,['Date','Stage','Type','Value','Min Value','Max Value','units']].copy()

				id_vars = ['Date_Time','Stage','obs_id','Type','units']
				value_vars = ['Value','Min Value','Max Value']


			# ======================================= TSS/VSS ======================================= #
			if ltype == 'TSS_VSS':
				# Create TSS and VSS variables
				self.ldata.loc[:,'TSS'] = \
					(self.ldata['Temp105 (g)'] - self.ldata['Original (g)'])/\
					self.ldata['Volume (ml)']*1E6
				self.ldata.loc[:,'VSS'] = \
					self.ldata['TSS'] - \
					(self.ldata['Temp550 (g)'] - self.ldata['Original (g)'])/\
					self.ldata['Volume (ml)']*1E6
				self.ldata.loc[:,'units'] = 'mg/L'
				# Set id and value vars for melting
				value_vars = ['TSS','VSS']

			# ======================================= ALKALINITY ======================================= #
			if ltype == 'ALKALINITY':
				# Compute alkalinity
				self.ldata.loc[:,'ALKALINITY'] = \
					self.ldata['Acid Volume (mL, to pH 4.3)']*self.ldata['Acid Normality (N)']/\
					self.ldata['Sample Volume (mL)']*self.ldata['Dilution Factor']*50*1000	
				self.ldata.loc[:,'units'] = 'mg/L as CaCO3'

			# ======================================= VFA =============================================== #
			if ltype == 'VFA':
				# Compute VFA concentrations
				self.ldata.loc[:,'Acetate'] = self.ldata['Acetate (mgCOD/L)']*self.ldata['Dilution Factor']
				self.ldata.loc[:,'Propionate'] = self.ldata['Propionate (mgCOD/L)']*self.ldata['Dilution Factor']
				self.ldata.loc[:,'units'] = 'mgCOD/L'
				# Set value vars for melting
				value_vars = ['Acetate','Propionate']

			# ======================================= AMMONIA =============================================== #
			if ltype == 'AMMONIA':
				# Compute Ammonia concentration
				self.ldata.loc[:,'Ammonia'] = self.ldata['Reading (mg/L)']*self.ldata['Dilution Factor']
				self.ldata.loc[:,'units'] = 'mg/L'

			if ltype == 'TKN':
				# Compute distillation recovery
				self.ldata.loc[:,'Dist Recovery (%)'] = \
					((self.ldata['NH4Cl Volume (mL)'] - self.ldata['Blank Volume (mL)'])*\
					14.007*self.ldata['Acid Concentration (N)']*1000)/\
					(self.ldata['NH4Cl Concentration (mg/L)']*self.ldata['NH4Cl Sample Volume (mL)'])
				# Compute digestion efficiency
				self.ldata.loc[:,'Digest Eff (%)'] = \
					((self.ldata['Tryptophan Volume (mL)'] - self.ldata['Blank Volume (mL)'])*\
					14.007*self.ldata['Acid Concentration (N)']*1000)/\
					(self.ldata['Tryptophan Concentration (mg/L)']*self.ldata['Tryptophan Sample Volume (mL)'])	
				# Compute corrected TKN value (corrected for distillation recovery and digestion efficiency)
				self.ldata.loc[:,'TKN'] = \
					(((self.ldata['Volume (mL)'] - self.ldata['Blank Volume (mL)'])*\
					14.007*self.ldata['Acid Concentration (N)']*1000)/\
					(self.ldata['Sample Volume (mL)']))/\
					(self.ldata['Dist Recovery (%)']*self.ldata['Digest Eff (%)'])	
				# Set value vars for melting
				self.ldata.loc[:,'units'] = 'mgTKN/L'

			# ======================================= SULFATE =============================================== #
			if ltype == 'SULFATE':
				# Compute Sulfate concentration
				self.ldata.loc[:,'Sulfate'] = self.ldata['Reading (mg/L)']*self.ldata['Dilution Factor']
				self.ldata.loc[:,'units'] = 'mg/L S'

			# ======================================= GASCOMP ============================================ #
			if ltype == 'GASCOMP':
				self.ldata.loc[:,'Hel_Pressure'] = self.ldata['Helium pressure (psi) +/- 50 psi']
				self.ldata.loc[:,'Stage'] = 'NA'
				self.ldata.loc[:,'units'] = '(see Type)'
				# Set value vars for melting
				value_vars = ['Hel_Pressure (psi)','Nitrogen (%)','Oxygen (%)','Methane (%)','Carbon Dioxide (%)']

			# Add Sample Date-Time variable from PH
			if ltype != 'PH':
				self.ldata = self.ldata.merge(ldata_dt, on = 'Date')

			# Convert to long format
			if ltype in ['PH','ALKALINITY','AMMONIA','TKN','SULFATE']:
				value_vars = [ltype]

			# Convert to long format 
			ldata_long = self.wide_to_long(ltype, id_vars, value_vars)

			# Create key unique by Date_Time, Stage, Type, and obs_id
			ldata_long.loc[:,'Dkey'] = \
				ldata_long['Date_Time'].astype(str) + ldata_long['Stage'] + ldata_long['Type'] + ldata_long['obs_id'].astype(str)
			# Reorder columns to put DKey as first column
			colnames = list(ldata_long.columns.values)
			ldata_long = ldata_long[colnames[-1:] + colnames[0:-1]]

			# Load data to SQL
			# SQL command strings for sqlite3
			colNTypeStr = 'Date_Time INT, Stage TEXT, Type TEXT, units TEXT, obs_id INT, Value REAL'
			colNStr = ','.join(ldata_long.columns.values)
			colIns = ','.join(['?']*len(ldata_long.columns))

			create_str = """
				CREATE TABLE IF NOT EXISTS {0} (DKey INT PRIMARY KEY, {1})
			""".format(ltype,colNTypeStr)
			ins_str = """
				INSERT OR REPLACE INTO {0} ({1})
				VALUES ({2})
			""".format(ltype, colNStr, colIns)
			# Set connection to SQL database (pertaining to given year)
			os.chdir(self.data_dir)
			conn = sqlite3.connect('cr2c_labdata.db')
			# Load data to SQL
			# Create the table if it doesn't exist
			conn.execute(create_str)
			# Insert aggregated values for the sid and time period
			conn.executemany(
				ins_str,
				ldata_long.to_records(index = False).tolist()
			)
			conn.commit()
			# Close Connection
			conn.close()

			# Load data to Google BigQuery
			projectid = 'cr2c-monitoring'
			dataset_id = 'labdata'
			# Make sure only new records are being appended to the dataset
			ldata_long_already = get_data([ltype])[ltype]
			ldata_long_new = ldata_long.loc[~ldata_long['Dkey'].isin(ldata_long_already['Dkey']),:]
			# Remove duplicates and missing values
			ldata_long_new.dropna(inplace = True)
			ldata_long_new.drop_duplicates(inplace = True)
			# Write to gbq table
			if not ldata_long_new.empty:
				ldata_long_new.to_gbq('{}.{}'.format(dataset_id, ltype), projectid, if_exists = 'append')

