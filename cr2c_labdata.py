'''
	Script loads data from lab tests, computes water quality parameters,
	and loads the data to an SQL database (no inputs required, fully automated!)
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


class labrun:
	
	def __init__(self):
		
		self.mtype_list = ['PH','COD','TSS_VSS','ALKALINITY','VFA','GasComp','Ammonia','Sulfate']
		self.min_feas_dt_str = '6-1-16'
		self.min_feas_dt = dt.strptime(self.min_feas_dt_str, '%m-%d-%y')
		self.file_dt = dt.now()
		self.file_dt_str = dt.strftime(self.file_dt,'%m-%d-%y')
		self.data_dir, self.pydir = cut.get_dirs()


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
			os.chdir(self.data_dir)
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
		# Eliminate missing date variables
		self.mdata.dropna(subset = ['Date'], inplace = True)
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
		if mtype != 'GasComp':

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
			# Get descriptive stage variable
			self.get_stage_descs()

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

		if mtype in ['COD','Ammonia','Sulfate']:
			self.set_var_format(mtype, 'Reading (mg/L)', float, 'numeric')

		if mtype == 'PH':
			self.set_var_format(mtype, 'Reading', float, 'numeric')
		
		if mtype == 'ALKALINITY':
			self.set_var_format(mtype,'Sample Volume (ml)', float, 'numeric')
			self.set_var_format(mtype,'Acid Volume (ml, to pH 4.3)', float, 'numeric')
			self.set_var_format(mtype,'Acid Normality (N)', float, 'numeric')

		if mtype in ['COD','ALKALINITY','VFA','Ammonia','Sulfate']:
			self.set_var_format(mtype,'Dilution Factor', float, 'numeric')
		
		if mtype == 'TSS_VSS':
			self.set_var_format(mtype,'Volume (ml)', float, 'numeric')
			self.set_var_format(mtype,'Original (g)', float, 'numeric')
			self.set_var_format(mtype,'Temp105 (g)', float, 'numeric')
			self.set_var_format(mtype,'Temp550 (g)', float, 'numeric')

		if mtype == 'VFA':
			self.set_var_format(mtype,'Acetate (mgCOD/L)', float, 'numeric')
			self.set_var_format(mtype,'Propionate (mgCOD/L)', float, 'numeric')

		if mtype == 'GasComp':
			self.set_var_format(mtype,'Helium pressure (psi) +/- 50 psi', float, 'numeric')
			self.set_var_format(mtype,'Nitrogen (%)', float, 'numeric')
			self.set_var_format(mtype,'Oxygen (%)', float, 'numeric')
			self.set_var_format(mtype,'Methane (%)', float, 'numeric')
			self.set_var_format(mtype,'Carbon Dioxide (%)', float, 'numeric')
			
		# Get the obs_id variable (This step also removes duplicates and issues warnings)
		self.manage_dups(mtype, id_vars)
		self.mdata.reset_index(inplace = True)


	# Converts dataset to long 
	def wide_to_long(self, mtype, id_vars, value_vars):

		# Melt the data frame
		df_long = pd.melt(self.mdata, id_vars = id_vars, value_vars = value_vars)
		# Reorder columns
		if mtype == 'GasComp':
			col_order = ['Date_Time','Hel_Pressure','variable','obs_id','value']
			varnames = ['Date_Time','Hel_Pressure','Type','obs_id','Value']
		elif mtype in ['PH','ALKALINITY']:
			col_order = ['Date_Time','Stage','obs_id','value']
			varnames = ['Date_Time','Stage','obs_id','Value']
		elif mtype in ['Ammonia','Sulfate']:
			col_order = ['Date_Time','Stage','value']
			varnames = ['Date_Time','Stage','Value']
		else:	
			col_order = ['Date_Time','Stage','variable','obs_id','value']
			varnames = ['Date_Time','Stage','Type','obs_id','Value']

		df_long = df_long[col_order]
		df_long.columns = varnames

		return df_long

	# Inputs lab testing results data and computes water quality parameters
	def process_data(self):
		
		# Load data from gsheets
		all_sheets = cut.get_gsheet_data(self.mtype_list)

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
			elif mtype == 'GasComp':
				self.clean_dataset(mtype,['Date','Helium pressure (psi) +/- 50 psi'])
			else:
				self.clean_dataset(mtype,['Date','Stage'])

			# ======================================= pH ======================================= #
			if mtype == 'PH':

				# Set id and value vars for cleaning
				id_vars = ['Date_Time','Stage','obs_id']
				value_vars = 'Reading'	

				# Get time of sample collection from PH dataset and add to date variable to get single Date + Time variable
				self.mdata['Date_str'] = self.mdata['Date'].dt.strftime('%m-%d-%y')
				self.mdata['Date-Time_str'] = self.mdata.Date_str.str.cat(self.mdata['Time'], sep = ' ')
				self.mdata['Date_Time'] = pd.to_datetime(self.mdata['Date-Time_str'])
				mdata_dt = self.mdata[['Date','Date_Time']]

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
				id_vars = ['Date_Time','Stage','obs_id']
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
				id_vars = ['Date_Time','Stage','obs_id']
				value_vars = ['TSS','VSS']

			# ======================================= ALKALINITY ======================================= #
			if mtype == 'ALKALINITY':
				
				# Compute alkalinity
				self.mdata['ALKALINITY'] = \
				self.mdata['Acid Volume (ml, to pH 4.3)']*self.mdata['Acid Normality (N)']/\
				self.mdata['Sample Volume (ml)']*self.mdata['Dilution Factor']*50*1000

				# Set id and value vars for cleaning	
				id_vars = ['Date_Time','Stage','obs_id']			
				value_vars = 'ALKALINITY'

			# ======================================= VFA =============================================== #
			if mtype == 'VFA':

				# Compute VFA concentrations
				self.mdata['Acetate'] = self.mdata['Acetate (mgCOD/L)']*self.mdata['Dilution Factor']
				self.mdata['Propionate'] = self.mdata['Propionate (mgCOD/L)']*self.mdata['Dilution Factor']

				# Set id and value vars for recasting
				id_vars = ['Date_Time','Stage','obs_id']
				value_vars = ['Acetate','Propionate']

			# ======================================= Ammonia =============================================== #
			if mtype == 'Ammonia':

				# Compute Ammonia concentration
				self.mdata['Ammonia'] = self.mdata['Reading (mg/L)']*self.mdata['Dilution Factor']

				# Set id and value vars for recasting
				id_vars = ['Date_Time','Stage']
				value_vars = 'Ammonia'

			# ======================================= Sulfate =============================================== #
			if mtype == 'Sulfate':

				# Compute Sulfate concentration
				self.mdata['Sulfate'] = self.mdata['Reading (mg/L)']*self.mdata['Dilution Factor']

				# Set id and value vars for recasting
				id_vars = ['Date_Time','Stage']
				value_vars = 'Sulfate'

			# ======================================= GasComp ============================================ #
			if mtype == 'GasComp':

				self.mdata['Hel_Pressure'] = self.mdata['Helium pressure (psi) +/- 50 psi']
				# Set id and value vars for recasting
				id_vars = ['Date_Time','Hel_Pressure','obs_id']
				value_vars = ['Nitrogen (%)','Oxygen (%)','Methane (%)','Carbon Dioxide (%)']


			# Add Sample Date-Time variable from PH
			if mtype != 'PH':
				self.mdata = self.mdata.merge(mdata_dt, on = 'Date')

			# Convert to long format
			mdata_long = self.wide_to_long(mtype, id_vars, value_vars)

			# Load data to SQL
			os.chdir(self.data_dir)
			conn = sqlite3.connect('cr2c_lab_data.db')
			mdata_long.to_sql(mtype + '_data', conn, if_exists = 'replace', index = False)


	# Queries Lab Data SQL File 
	def get_data(self, mtypes, start_dt_str = None, end_dt_str = None):

		# Convert date string inputs to dt variables
		if start_dt_str:
			start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
		if end_dt_str:
			end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

		# Load data from SQL
		os.chdir(self.data_dir)
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
			# Dedupe data (some issue with duplicates)
			mdata_long.drop_duplicates(inplace = True)
			# Convert Date_Time variable to a pd datetime and eliminate missing values
			mdata_long['Date_Time'] = pd.to_datetime(mdata_long['Date_Time'])
			mdata_long.dropna(subset = ['Date_Time'], inplace = True)

			if start_dt_str:
				mdata_long = mdata_long.loc[mdata_long['Date_Time'] >= start_dt,:]
			if end_dt_str:
				mdata_long = mdata_long.loc[mdata_long['Date_Time'] <= end_dt + timedelta(days = 1),:]

			mdata_all[mtype] = mdata_long

		return mdata_all


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
		mdata_all = self.get_data(mplot_list)

		# Loop through the lab data types
		for mtype in mplot_list:

			if mtype.find('TSS') >= 0 or mtype.find('VSS') >= 0:
				mtype = 'TSS_VSS'

			mdata_long = mdata_all[mtype]

			# Set format of date variable
			mdata_long['Date_Time'] = pd.to_datetime(mdata_long['Date_Time'])
			
			if mtype == 'COD':

				# Set plotting variables
				id_vars_chrt = ['Date_Time','Stage','Type']
				ylabel = 'COD Reading (mg/L)'
				type_list = ['Total','Soluble','Particulate']
				share_yax = False

			if mtype == 'TSS_VSS':

				# Set plotting variables
				id_vars_chrt = ['Date_Time','Stage','Type']
				ylabel = 'Suspended Solids (mg/L)'
				type_list = ['TSS','VSS']
				share_yax = True

			if mtype == 'PH':

				# Set plotting variables
				id_vars_chrt = ['Date_Time','Stage','Type']
				ylabel = 'pH'
				mdata_long['Type'] = 'pH'
				type_list = ['pH']
				share_yax = True

			if mtype == 'ALKALINITY':

				# Set plotting variables
				id_vars_chrt = ['Date_Time','Stage','Type']
				ylabel = 'Alkalinity (mg/L as ' + r'$CaCO_3$)'
				mdata_long['Type'] = 'Alkalinity'
				type_list = ['Alkalinity']
				share_yax = True

			if mtype == 'VFA':

				# Set plotting variables
				id_vars_chrt = ['Date_Time','Stage','Type']
				ylabel = 'VFAs as mgCOD/L'
				type_list = ['Acetate','Propionate']
				share_yax = False

			if mtype == 'AMMONIA':

				#Set plotting variables
				id_vars_chrt = ['Date_Time','Stage','Type']
				ylabel = r'$NH_3$' + ' (mg/L as N)'
				mdata_long['Type'] = 'Ammonia'
				type_list = ['Ammonia']
				share_yax = True

			if mtype == 'SULFATE':

				# Set plotting variables
				id_vars_chrt = ['Date_Time','Stage','Type']
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
			    if mtype == 'PH':
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

			# Add and position legend
			if mtype in ['PH','ALKALINITY'] and wrap_var == 'Stage':
				plt.savefig(
					plot_filename.format(mtype, opfile_suff), 
					bbox_inches = 'tight',
					width = plot_wid, 
					height = plot_len
				)
			else:
				handles, labels = ax.get_legend_handles_labels()
				lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor = (1,0.75))
				plt.savefig(
					plot_filename.format(mtype, opfile_suff), 
					bbox_extra_artists = (lgd,),
					bbox_inches = 'tight',
					width = plot_wid, 
					height = plot_len
				)


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
		mdata_all = self.get_data(['COD','TSS_VSS','ALKALINITY','PH','VFA','Ammonia','Sulfate'])

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

