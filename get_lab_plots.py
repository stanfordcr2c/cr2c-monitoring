# Creates plots customizable by date range and type from cr2c's lab testing data

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
import math
import sys
import get_lab_data as gld
	

class lab_plots:

	def __init__(self, start_dt, end_dt):

		self.start_dt = start_dt
		self.end_dt = end_dt

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


	def get_lab_plots(
		self,
		mplot_list,
		wrap_var,
		stage_sub = None,
		type_sub = None
	):

		# Request tables and charts output directory from user
		self.charts_outdir = askdirectory(title = 'Directory to output charts to')
		try:
			os.chdir(self.charts_outdir)
		except OSError:
			print('Please choose a valid directory to output the charts to')
			sys.exit()

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
		self.manage_chart_dates()

		# Get all of the lab data requested
		mdata_all = gld.get_ldata(mplot_list)

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

			# Filter to the dates desired for the plots
			mdata_chart = mdata_long.loc[
				(mdata_long.Date_Time >= self.start_dt) &
				(mdata_long.Date_Time <= self.end_dt) 
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
			plot_len = 6*math.ceil(len(wrap_list)/3) + 5

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

			# Add and position legend
			handles, labels = ax.get_legend_handles_labels()
			lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor = (1,0.75))

			# Output plot to given directory
			plot_filename = "{0}_{1}_to_{2}.png"
			os.chdir(self.charts_outdir)
			plt.savefig(
				plot_filename.format(mtype, self.start_dt_str, self.end_dt_str), 
				bbox_extra_artists = (lgd,),
				bbox_inches = 'tight',
				width = plot_wid, 
				height = plot_len
			)

if __name__ == '__main__':

	# Instantiate class
	lplots = lab_plots(
		# Start of chart date range (default is June 1st 2016)
		'07-01-17', 
		# End of date range (default is today's date)
		None
	)

	# Create and output charts
	lplots.get_lab_plots(
		# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
		['COD','TSS','VFA','PH'], 
		# Variable to break down into panels according to
		'Type',
		# Stages to Subset to
		['Microscreen','AFBR','Duty AFMBR Effluent','Duty AFMBR MLSS']
	)