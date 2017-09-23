''' 
	This script calculates totals and averages for any given HMI data point(s), 
	time period, and date range for which a raw eDNA query has been run (and a csv file
	for that query obtained)
	If desired, also outputs plots and summary tables
'''

from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import pylab as pl
import numpy as np
import pandas as pd
import datetime as datetime
from datetime import datetime as dt
from datetime import timedelta as tdelt
from pandas import read_excel
import os
import sys
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

class hmi_data_agg:

	def __init__(self, qtype, stype):

		self.qtype = qtype.upper()
		self.stype = stype.upper()


	def prep_data(self, elid):

		# Set high and low limits for sensors based on type (water, gas, ph, conductivity, temp)
		if self.stype == 'WATER':
			hi_limit = 200
			lo_limit = 0.2
		elif self.stype == 'GAS':
			hi_limit = 10
			lo_limit = 0.005
		elif self.stype == 'PH':
			hi_limit = 10
			lo_limit = 4

		# Load data
		try:
			self.hmi_data = pd.read_csv(self.hmi_path)
		except FileNotFoundError:
			print('Please choose an existing input file with the HMI data')
			sys.exit()

		# Load variables and set output variable names
		varname = 'CR2C.CODIGA.{0}.SCALEDVALUE {1} [{2}]'
		self.xvar = elid + '_ts'
		self.yvar = elid + '_value'	

		# Rename variable
		self.hmi_data[self.yvar] = \
			self.hmi_data[varname.format(elid,'Value', self.qtype)]
		# Set low/negative values to 0 and remove unreasonably high values
		self.hmi_data.loc[self.hmi_data[self.yvar] < lo_limit, self.yvar] = 0
		self.hmi_data.loc[self.hmi_data[self.yvar] > hi_limit, self.yvar] = np.NaN			
		# Rename and format corresponding timestamp variable 
		self.hmi_data[self.xvar ] = \
			self.hmi_data[varname.format(elid, 'Time', self.qtype)]
		self.hmi_data[self.xvar ] = \
			pd.to_datetime(self.hmi_data[self.xvar])

		# Filter dataset to clean values, time period and variable selected
		self.hmi_data = self.hmi_data.loc[
			(self.hmi_data[self.xvar] >= self.start_dt - datetime.timedelta(days = 1)) &
			(self.hmi_data[self.xvar] <= self.end_dt + datetime.timedelta(days = 1))
			, 
			[self.xvar, self.yvar]
		]
		# Eliminate missing values and reset index
		self.hmi_data.dropna(axis = 0, how = 'any', inplace = True)
		self.hmi_data.reset_index(inplace = True)

		# Get numeric time elapsed
		self.first_ts = self.hmi_data[self.xvar][0]
		self.last_ts  = self.hmi_data[self.xvar][len(self.hmi_data) - 1]

		# Check to make sure that the totals/averages do not include the first
		# and last days for which data are available (just to ensure accuracy)
		if self.first_ts >= self.start_dt or self.last_ts <= self.end_dt:
			start_dt_warn = self.first_ts + np.timedelta64(1,'D')
			end_dt_warn   =  self.last_ts - np.timedelta64(1,'D')
			start_dt_warn = dt.strftime(start_dt_warn, '%m-%d-%y')
			end_dt_warn = dt.strftime(end_dt_warn, '%m-%d-%y')
			warn_msg = \
				'Given the range of data available for {0}, accurate aggregate values can only be obtained for: {1} to {2}'
			print(warn_msg.format(elid, start_dt_warn, end_dt_warn))


	def get_tot_var(
		self, 
		tperiod,
		elid, 
		agg_type
	):

		# Calculate time elapsed in minutes (since highest resolution is ~30s)
		self.hmi_data['tel'] = \
			(self.hmi_data[self.xvar] - self.first_ts)/\
			np.timedelta64(60,'s')
		self.hmi_data['hour'] = self.hmi_data[self.xvar].values.astype('datetime64[h]')
		# Calculate time elapsed in minutes at the beginning of the given hour
		self.hmi_data['tel_hstrt'] = \
			(self.hmi_data['hour'] - self.first_ts)/\
			np.timedelta64(60,'s')
		
		# Create a variable giving the totalized component for the given section (tel to tel_next)
		self.hmi_data['tot'] =\
			(self.hmi_data['tel'].shift(-1) - self.hmi_data['tel'])*\
			(self.hmi_data[self.yvar].shift(-1) + self.hmi_data[self.yvar])/2

		# Adjust the totalized component at the beginning of each hour (add the levtover time since 0:00)
		self.hmi_data.loc[self.hmi_data['tel_hstrt'] != self.hmi_data['tel_hstrt'].shift(1),'tot'] = \
			self.hmi_data['tot'] +\
			(self.hmi_data['tel'] - self.hmi_data['tel_hstrt'])*\
			0.5*(
				self.hmi_data[self.yvar] +\
				self.hmi_data[self.yvar].shift(1) +\
				(self.hmi_data[self.yvar] - self.hmi_data[self.yvar].shift(1))/\
				(self.hmi_data['tel'] - self.hmi_data['tel'].shift(1))*\
				(self.hmi_data['tel_hstrt'] - self.hmi_data['tel'].shift(1))
			)
		
		# Adjust the totalized component at the end of each hour (subtract the time after 0:00)
		self.hmi_data.loc[self.hmi_data['tel_hstrt'] != self.hmi_data['tel_hstrt'].shift(-1),'tot'] = \
			self.hmi_data['tot'] -\
			(self.hmi_data['tel'].shift(-1) - self.hmi_data['tel_hstrt'] - 60)*\
			0.5*(
				self.hmi_data[self.yvar] + \
				self.hmi_data[self.yvar].shift(-1) + \
				(self.hmi_data[self.yvar].shift(-1) - self.hmi_data[self.yvar])/\
				(self.hmi_data['tel'].shift(-1) - self.hmi_data['tel'])*\
				(self.hmi_data['tel_hstrt'] + 60 - self.hmi_data['tel'])
			)

		# Compute the area under the curve for each time period
		nperiods = (self.end_dt - self.start_dt).days*24/tperiod
		nperiods = int(nperiods)
		tots_res = []
		for period in range(nperiods):
			start_tel = (self.start_dt - self.first_ts) / np.timedelta64(1,'m') + period*60*tperiod
			end_tel = start_tel + 60*tperiod
			start_ts = self.start_dt + datetime.timedelta(hours = period*tperiod)
			ip_tot = self.hmi_data.loc[
				(self.hmi_data['tel'] >= start_tel) & 
				(self.hmi_data['tel'] <= end_tel),
				'tot'
			].sum()
			if agg_type == 'AVERAGE':
				ip_tot = ip_tot/(60*tperiod)
			tots_row = [start_ts, ip_tot]
			tots_res.append(tots_row)

		return tots_res


	def run_report(
		self,
		tperiod,
		elids,
		agg_types,
		start_dt,
		end_dt,
		hmi_path = None,
		output_csv = None
	):

		# Select input data file
		if hmi_path:
			self.hmi_path = hmi_path
		else:
			self.hmi_path = askopenfilename(title = 'Select HMI data input file')
		
		# Get dates and date strings for output filenames
		self.start_dt = dt.strptime(start_dt,'%m-%d-%y')
		self.end_dt = dt.strptime(end_dt,'%m-%d-%y')
		start_dt_str = dt.strftime(self.start_dt, '%m-%d-%y')
		end_dt_str = dt.strftime(self.end_dt, '%m-%d-%y')

		# Get string of all element ids and clean agg_types input
		self.all_elids = '_'.join(elids) 
		agg_types = [agg_type.upper() for agg_type in agg_types]

		for elid, agg_type in zip(elids, agg_types):
			# Get prepped data
			self.prep_data(elid)
			# Get totalized values'
			report_dat = self.get_tot_var(tperiod, elid, agg_type)
			if elid == elids[0]:
				self.res_df = pd.DataFrame([row[0] for row in report_dat], columns = ['Time'])
			# Skip time variable for all other elements we are getting data for
			self.res_df[elid + '_' + agg_type] = [row[1] for row in report_dat]

		# Output to directory given
		if output_csv:
			op_path = askdirectory(title = 'Directory to save HMI {0} output_file_to:'.format(self.stype))
			agg_filename = "HMI{0}_{1}_{2}_{3}.csv".format(self.stype, self.all_elids, start_dt_str, end_dt_str)
			self.res_df.to_csv(
				os.path.join(op_path, agg_filename), 
				index = False, 
				encoding = 'utf-8'
			)

		return self.res_df


	def get_agg_sumst(
		self, 
		output_types,
		start_dt = None,
		end_dt = None,
		sum_period = 'DAY', 
		plt_type = None, 
		plt_colors = None,
		ylabel = None,
		get_nhours = None
	):
		

		if start_dt == None:
			start_dt = self.start_dt
		else:
			start_dt = dt.strptime(start_dt,'%m-%d-%y')
		if end_dt == None:
			end_dt = self.end_dt
		else:
			end_dt = dt.strptime(end_dt,'%m-%d-%y')

		start_dt_str = dt.strftime(start_dt,'%m-%d-%y')
		end_dt_str = dt.strftime(end_dt,'%m-%d-%y')

		# Clean case of input arguments
		sum_period = sum_period.upper()
		plt_type = plt_type.upper()
		if type(output_types) == list:
			output_types = [output_type.upper() for output_type in output_types]
		else:
			output_types = output_types.upper()

		# Input aggregated data from file if a report isn't being run at the same time
		try:
			self.res_df
		except AttributeError:
			hmi_path = askopenfilename(title = 'Select file with HMI aggregated data')	 
			self.res_df = pd.read_csv(hmi_path)

		# Get output directory and string with all element ids from report
		agg_outdir = askdirectory(title = 'Directory to output to')

		# Retrieve element ids from aggregated data
		elids = self.res_df.columns[1:].values

		# Convert Time variable to pd.datetime variable
		self.res_df['Time'] = pd.to_datetime(self.res_df['Time'])
		self.res_df['Date'] = self.res_df['Time'].dt.date

		# Filter to the dates desired for the plots
		self.res_df = self.res_df.loc[
			(self.res_df.Time >= start_dt) &
			(self.res_df.Time <= end_dt)
		]

		# Get dataset aggregated by Day, Week or Month
		if sum_period == 'HOUR':
			xlabel = 'Time'
		else:
			self.res_df['Date'] = self.res_df['Time'].dt.date

		if sum_period == 'DAY':
			xlabel = 'Date'

		if sum_period == 'WEEK':
			xlabel = 'Weeks (since {0})'.format(start_dt_str)
			self.res_df[xlabel] = np.floor((self.res_df['Time'] - self.start_dt)/np.timedelta64(7,'D'))
		
		if sum_period == 'MONTH':
			xlabel = 'Months (since {0}, as 30 days)'.format(start_dt_str)
			self.res_df[xlabel] = np.floor((self.res_df['Time'] - self.start_dt)/np.timedelta64(30,'D'))

		if get_nhours == 1:
			for elid in elids:
				self.res_df['Number Hours {0}'.format(elid)] = \
					np.where(self.res_df[elid].values > 0, 1, 0)

		agg_sumst = self.res_df.groupby(xlabel).sum()

		# Plot!
		if 'PLOT' in output_types:

			# Set the maximum number of tick labels
			nobs  = len(agg_sumst.index.values)
			nlims = nobs
			if sum_period == 'DAY':
				nlims = 12
			# Get the indices of the x-axis values according to these tick labels
			lim_len  = int(np.floor(nobs/nlims))
			tic_idxs = [lim*lim_len for lim in range(nlims)]
			tic_vals = [agg_sumst.index.values[tic_idx] for tic_idx in tic_idxs]
			
			if sum_period != 'DAY':
				tic_vals = ['{0} - {1}'.format(int(tic_val), int(tic_val + 1)) for tic_val in tic_vals]

			if plt_type == 'BAR':
				ax = agg_sumst[elids].plot.bar(stacked = False, width = 0.8, color = plt_colors)
				plt.xticks(tic_idxs,tic_vals)
			else:
				ax = agg_sumst[elids].plot(color = plt_colors)

			plt.ylabel(ylabel)
			plt.legend()

			ax.yaxis.set_major_formatter(
				tkr.FuncFormatter(lambda y, p: format(int(y), ','))
			)
			
			plt.xticks(rotation = 45)
			plt.tight_layout()

			# Output plots and/or sumstats csv files to directory of choice
			plot_filename  = "HMI{0}_{1}_{2}_{3}.png".format(self.stype, self.all_elids, start_dt_str, end_dt_str)
			plt.savefig(
				os.path.join(agg_outdir, plot_filename), 
				width = 20, 
				height = 50
			)

		if 'TABLE' in output_types:

			sumst_filename = "HMI{0}_{1}_{2}_{3}.csv".format(self.stype, self.all_elids, start_dt_str, end_dt_str)
			agg_sumst.reset_index(inplace = True)
			agg_sumst.to_csv(
				os.path.join(agg_outdir, sumst_filename), 
				index = False,
				encoding = 'utf-8'
			)


if __name__ == '__main__':
	hmi_dat = hmi_data_agg(
		'raw', # Type of eDNA query (case insensitive, can be raw, 1 min, 1 hour)
		'water' # Type of sensor (case insensitive, can be water, gas, pH, conductivity or temperature
	)
	hmi_dat.run_report(
		1, # Number of hours you want to sum/average over
		['FT202','FT305'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
		['total','total'], # Type of aggregate function you want (can be total or average)
		'9-10-17', # Start of date range you want summary data for
		'9-22-17', # End of date range you want summary data for)
		output_csv = 1
	)
	# hmi_dat.get_agg_sumst(
	# 	output_types = ['PLOT','TABLE'],
	# 	sum_period = 'day',
	# 	plt_type = 'bar',
	# 	# plt_colors = ['#90775a','#eeae10'],
	# 	ylabel = 'Reactor I/O Volumes (Gal/day)'
	# )
