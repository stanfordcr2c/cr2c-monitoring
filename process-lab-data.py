# Script loads data from lab tests, computes water quality parameters,
# and if desired, saves csv files with most up-to-date testing data,
# outputs plots, and wide tables for monitoring reports

from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import seaborn as sns
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import warnings
import os
import sys
import httplib2
import sqlite3
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage


try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

class cr2c_monitor_run:
	
	def __init__(self):
		
		self.mtype_list = ['COD','TSS_VSS','PH','ALKALINITY','VFA']
		self.min_feas_dt_str = '6-1-16'
		self.min_feas_dt = dt.strptime(self.min_feas_dt_str, '%m-%d-%y')
		self.file_dt = dt.now()
		self.file_dt_str = dt.strftime(self.file_dt,'%m-%d-%y')
    
    # Gets valid user credentials from storage.
    # If nothing has been stored, or if the stored credentials are invalid,
    # the OAuth2 flow is completed to obtain the new credentials.
	def get_credentials(self):

		SCOPES = 'https://www.googleapis.com/auth/spreadsheets.readonly'
		CLIENT_SECRET_FILE = 'client_secret.json'
		APPLICATION_NAME = 'cr2c-monitoring'

		home_dir = os.path.expanduser('~')
		credential_dir = os.path.join(home_dir, '.credentials')

		if not os.path.exists(credential_dir):
			os.makedirs(credential_dir)
		credential_path = os.path.join(
			credential_dir,
			'sheets.googleapis.com-cr2c-monitoring.json'
		)
		store = Storage(credential_path)
		credentials = store.get()

		os.chdir(os.path.join(self.pydir,'GoogleProjectsAdmin'))
		spreadsheetId = open('spreadsheetId.txt').read()

		if not credentials or credentials.invalid:	
			flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
			flow.user_agent = APPLICATION_NAME
			credentials = tools.run_flow(flow, store, flags)

		return credentials, spreadsheetId

	# Retrieves all data from a gsheets file given list of sheet names
	def get_gsheet_data(self, sheet_names):

		credentials, spreadsheetId = self.get_credentials()
		http = credentials.authorize(httplib2.Http())
		discoveryUrl = (
			'https://sheets.googleapis.com/$discovery/rest?'
			'version=v4'
		)
		service = discovery.build(
			'sheets', 
			'v4', 
			http = http,
			discoveryServiceUrl = discoveryUrl
		)
		range_names = [sheet_name + '!A:G' for sheet_name in sheet_names]
		gsheet_result = service.spreadsheets().values().batchGet(
			spreadsheetId = spreadsheetId, 
			ranges = range_names
		).execute()	

		gsheet_values = gsheet_result['valueRanges']

		return gsheet_values

	# Manages output directories
	def get_outdirs(self):
		
		# Save parent directories
		self.cwd = os.getcwd()
		self.pydir = os.path.abspath(os.path.join(__file__,*([".."] * 2)))
		self.mondir = os.path.abspath(os.path.join(__file__ ,*([".."] * 3)))
		self.data_outdir = os.path.join(self.mondir,'Data')


	# Sets the start and end dates for the charts, depending on user input
	def manage_chart_dates(self, chart_start_dt, chart_end_dt):

		if chart_start_dt == None:
			self.chart_start_dt = self.min_feas_dt	
		else:
			self.chart_start_dt = dt.strptime(chart_start_dt, '%m-%d-%y')
		
		if chart_end_dt == None:
			self.chart_end_dt = self.file_dt
		else:
			self.chart_end_dt = dt.strptime(chart_end_dt, '%m-%d-%y')

		self.chart_start_dt_str = dt.strftime(self.chart_start_dt, '%m-%d-%y')
		self.chart_end_dt_str = dt.strftime(self.chart_end_dt, '%m-%d-%y')


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
	def manage_dups(self, mtype, id_vars):

		# Set duplicates warning
		dup_warning = \
			'There are repeat entries and/or entries with no date in {0} that were removed. '+\
	        'A csv of the removed values has been saved as {1}'

		# Check for Duplicates and empties
		repeat_entries = np.where(self.mdata.duplicated())[0].tolist()
		blank_entries = np.where(pd.isnull(self.mdata.Date))[0].tolist()
		repeat_entries.extend(blank_entries)

	    # If found, remove, print warning and output csv of duplicates/empties
		if len(repeat_entries) > 0:
			os.chdir(self.data_outdir)
			dup_filename = mtype + 'duplicates' + self.file_dt_str
			warnings.warn(dup_warning.format(mtype,dup_filename + '.csv'))
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
	def set_var_format(self, mtype, variable, format, format_prt):

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
			print(var_typ_warn.format(variable, mtype, format_prt))
			sys.exit()
		except ValueError:
			print(var_typ_warn.format(variable, mtype, format_prt))
			sys.exit()


	# Cleans a processed dataset, converting it to long format for plotting, output, etc
	def clean_dataset(self, mtype, id_vars):

		self.set_var_format(mtype, 'Date', None, "m-d-yy")
		# Make sure all date variables are within a reasonable range
		date_rng_warn = \
			'A Date variable in {0} has been entered incorrectly as {1} and removed'
		if self.mdata.Date.min() < self.min_feas_dt:
			print(date_rng_warn.format(mtype,self.mdata.Date.min()))
		if self.mdata.Date.max() > self.file_dt:
			print(date_rng_warn.format(mtype,self.mdata.Date.max()))
		# Filter dates accordingly
		self.mdata = self.mdata.loc[
			(self.mdata.Date >= self.min_feas_dt) &
			(self.mdata.Date <= self.file_dt)
		]

		# Format and clean stage variable
		self.mdata['Stage'] = self.mdata['Stage'].astype(str)
		self.mdata['Stage'] = self.mdata['Stage'].str.upper()
		self.mdata['Stage'] = self.mdata['Stage'].str.strip()

		# Check that the stage variable has been entered correctly
		correct_stages = ['RAW','GRIT','MS','AFBR','DAFMBRMLSS','DAFMBREFF','RAFMBRMLSS','RAFMBREFF']
		stage_warning = \
			'Check "Stage" entry {0} for {1} on dates: {2}. \n ' +\
			'"Stage" should be written as one of the following: \n {3}'
		stage_errors = self.mdata[ ~ self.mdata['Stage'].isin(correct_stages)]
		if len(stage_errors) > 0:
			date_err_prt = \
				[dt.strftime(stage_error,'%m-%d-%y') for stage_error in stage_errors.Date]
			print(
				stage_warning.format(
					stage_errors.Stage.values, mtype, date_err_prt, correct_stages
				)
			)
			sys.exit()

		# Format and clean other variables
		if mtype == 'COD':
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
			self.set_var_format(mtype, 'Reading (mg/L)', float, 'numeric')

		if mtype == 'PH':
			self.set_var_format(mtype, 'Reading', float, 'numeric')
		
		if mtype == 'ALKALINITY':
			self.set_var_format(mtype,'Sample Volume (ml)', float, 'numeric')
			self.set_var_format(mtype,'Acid Volume (ml, to pH 4.3)', float, 'numeric')
			self.set_var_format(mtype,'Acid Normality (N)', float, 'numeric')

		if mtype in ['COD','ALKALINITY','VFA']:
			self.set_var_format(mtype,'Dilution Factor', float, 'numeric')
		
		if mtype == 'TSS_VSS':
			self.set_var_format(mtype,'Volume (ml)', float, 'numeric')
			self.set_var_format(mtype,'Original (g)', float, 'numeric')
			self.set_var_format(mtype,'Temp105 (g)', float, 'numeric')
			self.set_var_format(mtype,'Temp550 (g)', float, 'numeric')

		if mtype == 'VFA':
			self.set_var_format(mtype,'Acetate (mgCOD/L)', float, 'numeric')
			self.set_var_format(mtype,'Propionate (mgCOD/L)', float, 'numeric')

		# Get the obs_id variable (This step also removes duplicates and issues warnings)
		self.manage_dups(mtype, id_vars)
		self.mdata.reset_index(inplace = True)

		# Get descriptive stage variable
		self.get_stage_descs()


	# Converts dataset to long 
	def wide_to_long(self, mtype, id_vars, value_vars):

		# Melt the data frame
		df_long = pd.melt(self.mdata, id_vars = id_vars, value_vars = value_vars)

		# Reorder columns
		df_long = df_long[['Date','Stage','variable','obs_id','value']]
		
		# Rename columns 
		varnames = ['Date','Stage','Type','obs_id','Value']
		if mtype in ['PH','ALKALINITY']:
			# Get rid of additional column in PH and Alkalinity data
			df_long = df_long[['Date','Stage','obs_id','value']]
			varnames = ['Date','Stage','obs_id','Value']
		df_long.columns = varnames

		return df_long

	# Inputs lab testing results data and computes water quality parameters
	def process_data(self):
		
		# Set output directories according to user input
		self.get_outdirs()

		# Load data from gsheets
		all_sheets = self.get_gsheet_data(self.mtype_list)

		# Start loop through the gsheets
		for sheet in all_sheets:

			# Retrieve the monitoring data type from the range name
			mtype = sheet['range'].split('!')[0]

			# Get data and header, convert to pandas data frame and clean
			mdata_list = sheet['values']
			headers = mdata_list.pop(0) 
			self.mdata = pd.DataFrame(mdata_list, columns = headers)
			if mtype == 'COD':
				self.clean_dataset(mtype,['Date','Stage','Type'])
			else:
				self.clean_dataset(mtype,['Date','Stage'])

			# ======================================= COD ======================================= #
			if mtype == 'COD':

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
				
				# Set id and value vars for recasting
				id_vars = ['Date','Stage','obs_id']
				value_vars = ['Soluble','Total','Particulate']

				# Subset to value variables and convert index to data
				self.mdata = mdata_wide[value_vars]
				self.mdata.reset_index(inplace = True)
				# Rename the columns
				self.mdata.columns = ['Date','Stage','obs_id'] + value_vars

			# ======================================= TSS/VSS ======================================= #
			if mtype == 'TSS_VSS':

				# Create TSS and VSS variables
				self.mdata['TSS'] = \
					(self.mdata['Temp105 (g)'] - self.mdata['Original (g)'])/\
					self.mdata['Volume (ml)']*1E6
				self.mdata['VSS'] = \
					self.mdata['TSS'] - \
					(self.mdata['Temp550 (g)'] - self.mdata['Original (g)'])/\
					self.mdata['Volume (ml)']*1E6

				# Set id and value vars for cleaning
				id_vars = ['Date','Stage','obs_id']
				value_vars = ['TSS','VSS']

			# ======================================= pH ======================================= #
			if mtype == 'PH':

				# Set id and value vars for cleaning
				id_vars = ['Date','Stage','obs_id']
				value_vars = 'Reading'		
			
			# ======================================= ALKALINITY ======================================= #
			if mtype == 'ALKALINITY':
				
				# Compute alkalinity
				self.mdata['ALKALINITY'] = \
				self.mdata['Acid Volume (ml, to pH 4.3)']*self.mdata['Acid Normality (N)']/\
				self.mdata['Sample Volume (ml)']*self.mdata['Dilution Factor']*50*1000

				# Set id and value vars for cleaning	
				id_vars = ['Date','Stage','obs_id']			
				value_vars = 'ALKALINITY'

			if mtype == 'VFA':

				# Compute VFA concentrations
				self.mdata['Acetate'] = self.mdata['Acetate (mgCOD/L)']*self.mdata['Dilution Factor']
				self.mdata['Propionate'] = self.mdata['Propionate (mgCOD/L)']*self.mdata['Dilution Factor']

				# Set id and value vars for recasting
				id_vars = ['Date','Stage','obs_id']
				value_vars = ['Acetate','Propionate']

			# Convert to long format
			mdata_long = self.wide_to_long(mtype, id_vars, value_vars)

			# Load data to SQL
			os.chdir(self.data_outdir)
			conn = sqlite3.connect('cr2c_lab_data.db')
			mdata_long.to_sql(mtype + '_data', conn, if_exists = 'replace', index = False)

# Execute script
if __name__ == "__main__":

	# Instantiate class
	cr2c_mr = cr2c_monitor_run()

	# Run data processing 
	cr2c_mr.process_data()
