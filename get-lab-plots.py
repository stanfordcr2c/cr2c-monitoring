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
from os.path import expanduser
import sys
import sqlite3
	

class lab_plots:

	def __init__(self, mplot_list, start_dt, end_dt):

		self.mplot_list = mplot_list
		self.start_dt = start_dt
		self.end_dt = end_dt


	# Manages output directories
	def get_outdirs(self):
		
		# Find the CR2C.Operations folder on Box Sync on the given machine
		targetdir = os.path.join('Box Sync','CR2C.Operations')
		self.mondir = None
		print("Searching for Codiga Center's Operations folder on Box Sync")
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
		self.charts_outdir = askdirectory(title = 'Directory to output charts to')


	# Sets the start and end dates for the charts, depending on user input
	def manage_chart_dates(self):

		if self.start_dt == None:
			self.start_dt = self.min_feas_dt	
		else:
			self.start_dt = dt.strptime(self.start_dt, '%m-%d-%y')
		
		if self.end_dt == None:
			self.end_dt = dt.now()
		else:
			self.end_dt = dt.strptime(self.end_dt, '%m-%d-%y')

		self.start_dt_str = dt.strftime(self.start_dt, '%m-%d-%y')
		self.end_dt_str = dt.strftime(self.end_dt, '%m-%d-%y')


	def get_lab_plots(self):

		# Get data input and chart output directories
		self.get_outdirs()
		try:
			os.chdir(self.charts_outdir)
		except OSError:
			print('Please choose a valid directory to output the charts to')
			sys.exit()
		try:
			os.chdir(self.data_indir)
		except OSError:
			print('Please choose the directory with the lab data sql file')
			sys.exit()

		# Capitalize all input chart data types
		self.mplot_list = [element.upper() for element in self.mplot_list]

		# Manage dates given by user
		self.manage_chart_dates()

		# Loop through the lab data types
		for mtype in self.mplot_list:

			# Load data from SQL
			os.chdir(self.data_indir)
			conn = sqlite3.connect('cr2c_lab_data.db')
			# Clean user input wrt TSS_VSS
			if mtype.find('TSS') >= 0 or mtype.find('VSS') >= 0:
				mtype = 'TSS_VSS'

			mdata_long = pd.read_sql(
				'SELECT * FROM {}_data'.format(mtype), 
				conn, 
				coerce_float = True
			)

			# Set format of date variable
			mdata_long['Date'] = pd.to_datetime(mdata_long['Date'])

			# Set 'hue' variable for seaborn
			hue = 'Type'
			if mtype in ['PH','ALKALINITY']:
				hue = None
			
			if mtype == 'COD':

				# Set plotting variables
				id_vars_chrt = ['Date','Stage','Type']
				ylabel = 'COD Reading (mg/L)'
				hue_order_list = ['Total','Soluble','Particulate']
				col_order_list = [
					'Raw Influent',
					'Grit Tank',
					'Microscreen',
					'AFBR',
					'Duty AFMBR MLSS',
					'Duty AFMBR Effluent'
				]	

			if mtype == 'TSS_VSS':

				# Set plotting variables
				id_vars_chrt = ['Date','Stage','Type']
				ylabel = 'Suspended Solids (mg/L)'
				hue_order_list = ['TSS','VSS']
				col_order_list = [
					'Raw Influent',
					'Grit Tank',
					'Microscreen',
					'AFBR',
					'Duty AFMBR MLSS',
					'Duty AFMBR Effluent'
				]

			if mtype == 'PH':

				# Set plotting variables
				id_vars_chrt = ['Date','Stage']
				ylabel = 'pH'
				hue_order_list = 'Value'	
				col_order_list = [
					'Raw Influent',
					'Grit Tank',
					'Microscreen',
					'AFBR',
					'Duty AFMBR MLSS',
					'Duty AFMBR Effluent'
				]		

			if mtype == 'ALKALINITY':

				# Set plotting variables
				id_vars_chrt = ['Date','Stage']
				ylabel = 'Alkalinity (mg/L as ' + r'$CaCO_3$)'
				hue_order_list = 'Value'
				col_order_list = [
					'Raw Influent',
					'Grit Tank',
					'Microscreen',
					'AFBR',
					'Duty AFMBR MLSS',
					'Duty AFMBR Effluent'
				]

			if mtype == 'VFA':

				# Set plotting variables
				id_vars_chrt = ['Date','Stage','Type']
				ylabel = 'VFAs as mgCOD/L'
				hue_order_list = ['Acetate','Propionate']
				col_order_list = [
					'AFBR',
					'Duty AFMBR MLSS',
					'Duty AFMBR Effluent'
				]

			# Filter to the dates desired for the plots
			mdata_chart = mdata_long.loc[
				(mdata_long.Date >= self.start_dt) &
				(mdata_long.Date <= self.end_dt)
			]

			# Average all observations (by type and stage) taken on a day
			mdata_chart = mdata_chart.groupby(id_vars_chrt).mean()

			# Remove index!
			mdata_chart.reset_index(inplace = True)

			# Set plot facetting and layout
			mplot = sns.FacetGrid(
				mdata_chart,
				col = 'Stage',
				col_order = col_order_list,
				col_wrap = 3,
				hue = hue,
				hue_order = hue_order_list,
				legend_out = False
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
			mplot.map(plt.plot,'Date','Value', linestyle = '-', marker = "o", ms = 4)
			mplot.set_titles('{col_name}')
			mplot.set_ylabels(ylabel)
			mplot.set_xlabels('Date')
			mplot.set_xticklabels(rotation = 45)
			mplot.add_legend(frameon = True)

			# Output plot to given directory
			plot_filename = "{0}_{1}_to_{2}.png"
			os.chdir(self.charts_outdir)
			plt.savefig(
				plot_filename.format(mtype, self.start_dt_str, self.end_dt_str), 
				width = 15, 
				height = 18
			)

if __name__ == '__main__':

	# Instantiate class
	lplots = lab_plots(
		# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
		['COD','TSS','PH','ALKALINITY','VFA'], 
		# Start of chart date range (default is June 1st 2016)
		'07-01-17', 
		# End of date range (default is today's date)
		None 
	)

	# Create and output charts
	lplots.get_lab_plots()