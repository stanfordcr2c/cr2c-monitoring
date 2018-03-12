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
import seaborn as sns
import pylab as pl
import numpy as np
import pandas as pd
import datetime as datetime
from datetime import datetime as dt
from datetime import timedelta
from pandas import read_excel
import sqlite3
import cr2c_utils as cut
import os
from os.path import expanduser
import sys
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory


def get_data(
	elids, 
	tperiods, 
	ttypes,
	year = None, 
	month_sub = None, 
	start_dt_str = None, 
	end_dt_str = None, 
	output_csv = False, 
	outdir = None
):

	# Clean user inputs
	ttypes = [ttype.upper() for ttype in ttypes]

	# Convert date string inputs to dt variables
	start_dt = dt.strptime('5-10-17','%m-%d-%y')
	end_dt = dt.now()
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')
	# Get list of years for which data are desired
	years = np.arange(start_dt.year,end_dt.year + 1)

	# Create connection to SQL database
	data_dir = cut.get_dirs()[0]
	os.chdir(data_dir)
	hmi_data_all = pd.DataFrame()

	for elid, tperiod, ttype in zip(elids, tperiods, ttypes):

		# month_sub insert
		if month_sub:
			msub_ins = 'WHERE Month == {0}'.format(month_sub)
		else:
			msub_ins = ''

		sql_str = """
			SELECT distinct * FROM {0}_{1}{2}_AVERAGES
			{3}
			order by Time 
		""".format(elid, tperiod, ttype, msub_ins)

		# Loop through years for which data are desired
		# (concatenate datasets)
		hmi_data = pd.DataFrame()
		for year in years:
		
			conn = sqlite3.connect('cr2c_hmi_agg_data_{}.db'.format(year))
			hmi_data_yr = pd.read_sql(
				sql_str,
				conn,
				coerce_float = True
			)
			hmi_data = pd.concat([hmi_data, hmi_data_yr], axis = 0)

		# Format the time variable
		hmi_data['Time'] = pd.to_datetime(hmi_data['Time'])
		# Set time variable
		if ttype == 'HOUR':
			hmi_data.loc[:,'Time'] = hmi_data['Time'].values.astype('datetime64[h]')
		elif ttype == 'MINUTE':
			hmi_data.loc[:,'Time'] = hmi_data['Time'].values.astype('datetime64[m]')

		# Rename Value variable to its corresponding element id
		hmi_data.rename(columns = {'Value': elid}, inplace = True)
		# Drop duplicates (happens with hourly aggregates sometimes...)
		hmi_data.drop_duplicates(['Time'], inplace = True)

		if start_dt_str:
			hmi_data = hmi_data.loc[hmi_data['Time'] >= start_dt,]
		if end_dt_str:
			hmi_data = hmi_data.loc[hmi_data['Time'] < end_dt + timedelta(days = 1),]

		if not len(hmi_data_all):
			hmi_data_all = hmi_data
		else:
			hmi_data_all = hmi_data_all.merge(hmi_data[['Time', elid]], on = 'Time', how = 'outer')

	if output_csv:

		if not outdir:
			print('Directory to output HMI data to...')
			outdir = askdirectory(title = 'Directory to output HMI data to...')

		os.chdir(outdir)
		op_fname = '_'.join(elids + [str(tperiod) for tperiod in tperiods]) + '.csv'
		hmi_data_all.to_csv(op_fname, index = False, encoding = 'utf-8')


	return hmi_data_all

# Function to eliminate misaligned time readings from dataset (can happen from bugs in pandas + timedelta)
def clean_data(elids, tperiods, ttypes, year):

	# Clean user inputs
	ttypes = [ttype.upper() for ttype in ttypes]

	# Read in all the data that will be cleaned
	hmi_data_all = get_data(elids, tperiods, ttypes, year)

	for elid, tperiod, ttype in zip(elids, tperiods, ttypes):

		# First read in the data
		hmi_data = hmi_data_all[['Tkey','Time','Month',elid]]
		hmi_data.rename(columns = {elid: 'Value'}, inplace = True)
		# Eliminate timesteps that are out of sync and dedupe
		hmi_data['Time'] = hmi_data['Time'].values.astype('datetime64[m]')
		# Leaving out Tkey because otherwise duplicate entries will still be unique
		hmi_data.drop_duplicates(['Time','Month','Value'], inplace = True)
 
 		# Delete time periods that are out of sync (such as 20:05:00 when its an hourly dataset)
		del_str = """
			DELETE FROM {0}_{1}{2}_AVERAGES
			WHERE Tkey % 1e+10 > 0
		""".format(elid, tperiod, ttype)
		ins_str = """
			INSERT OR REPLACE INTO {0}_{1}{2}_AVERAGES (Tkey, Time, Month, Value)
			VALUES (?,?,?,?)
		""".format(elid, tperiod, ttype)

		# Create connection to SQL database
		conn = sqlite3.connect('cr2c_hmi_agg_data_{0}.db'.format(year))
		# Execute the delete statement
		conn.execute(del_str)
		# Load cleaned data back to the database
		conn.executemany(
			ins_str,
			hmi_data.to_records(index = False).tolist()
		)
		conn.commit()
		# Close connection to sql file
		conn.close()

	return


# Primary HMI data aggregation class
class hmi_data_agg:

	def __init__(self, start_dt_str, end_dt_str, hmi_path = None):

		self.start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
		self.end_dt = dt.strptime(end_dt_str,'%m-%d-%y')
		self.data_dir = cut.get_dirs()[0]

		# Select input data file and load data for run
		if hmi_path:
			self.hmi_dir = os.path.dirname(hmi_path)
		else:
			tkTitle = 'Select HMI data input file...'
			print(tkTitle)
			hmi_path = askopenfilename(title = tkTitle)
			self.hmi_dir = os.path.dirname(hmi_path)
		try:
			self.hmi_data_all = pd.read_csv(hmi_path)
		except FileNotFoundError:
			print('Please choose an existing input file with the HMI data')
			sys.exit()

	def prep_data(self, elid, stype):

		# Set high and low limits for sensors based on type (water, gas, ph, conductivity, temp)
		if stype == 'WATER':
			hi_limit = 200
			lo_limit = 0.2
		elif stype == 'GAS':
			hi_limit = 10
			lo_limit = 0.005
		elif stype == 'PH':
			hi_limit = 10
			lo_limit = 4
		elif stype == 'TEMP':
			hi_limit = 50
			lo_limit = 0
		elif stype == 'PRESSURE':
			hi_limit = 16
			lo_limit = 13.4
		elif stype == 'TMP':
			hi_limit = 20
			lo_limit = -20

		# Load variables and set output variable names
		varname = 'CR2C.CODIGA.{0}.SCALEDVALUE {1} [{2}]'

		# Rename variable
		qtype = 'RAW'
		self.hmi_data = self.hmi_data_all
		self.hmi_data['Value'] = \
			self.hmi_data[varname.format(elid,'Value', qtype)]
		# Set low/negative values to 0 (if a flow, otherwise remove) and remove unreasonably high values
		if stype in ['GAS','WATER']:
			self.hmi_data.loc[self.hmi_data['Value'] < lo_limit, 'Value'] = 0
		else:
			self.hmi_data.loc[self.hmi_data['Value'] < lo_limit, 'Value'] = np.NaN
		self.hmi_data.loc[self.hmi_data['Value'] > hi_limit, 'Value'] = np.NaN

		# Rename and format corresponding timestamp variable
		self.hmi_data['Time' ] = \
			self.hmi_data[varname.format(elid, 'Time', qtype)]
		# Set as datetime variable at second resolution (uses less memory than nanosecond!)
		self.hmi_data['Time' ] = \
			pd.to_datetime(self.hmi_data['Time']).values.astype('datetime64[s]')

		# Filter dataset to clean values, time period and variable selected
		self.hmi_data = self.hmi_data.loc[
			(self.hmi_data['Time'] >= self.start_dt - datetime.timedelta(days = 1)) &
			(self.hmi_data['Time'] < self.end_dt + datetime.timedelta(days = 1))
			,
			['Time', 'Value']
		]
		# Eliminate missing values and reset index
		self.hmi_data.dropna(axis = 0, how = 'any', inplace = True)
		self.hmi_data.reset_index(inplace = True)

		# Get the first and last time
		self.first_ts = self.hmi_data['Time'][0]
		self.last_ts  = self.hmi_data['Time'][len(self.hmi_data) - 1]

		# Check to make sure that the totals/averages do not include the first
		# and last days for which data are available (just to ensure accuracy)
		if self.first_ts >= self.start_dt or self.last_ts <= self.end_dt:
			start_dt_warn = self.first_ts + np.timedelta64(1,'D')
			end_dt_warn   =  self.last_ts - np.timedelta64(1,'D')
			start_dt_warn_str = dt.strftime(start_dt_warn, '%m-%d-%y')
			end_dt_warn_str = dt.strftime(end_dt_warn, '%m-%d-%y')
			warn_msg = \
				'Given the range of data available for {0}, accurate aggregate values can only be obtained for: {1} to {2}'
			print(warn_msg.format(elid, start_dt_warn_str, end_dt_warn_str))
			# Change start_dt and end_dt of system to avoid overwriting sql file with empty data
			self.start_dt = start_dt_warn
			self.end_dt = end_dt_warn


	def get_tot_var(
		self,
		tperiod,
		ttype,
		elid
	):

		# Get minute-level dataframe of timesteps for the time period requested
		ts_array = np.arange(
			self.start_dt,
			self.end_dt + datetime.timedelta(days = 1),
			np.timedelta64(1,'m')
		)
		empty_df = pd.DataFrame(ts_array, columns = ['Time'])

		# Merge this with the HMI data and fill in NaNs by interpolating
		hmi_data_all = self.hmi_data.merge(empty_df, on = 'Time', how = 'outer')
		# Sort the dataset by Time (important for TimeEL below)
		hmi_data_all.sort_values('Time', inplace = True)
		# ... need to set Time as an index to do this
		hmi_data_all.set_index('Time')
		hmi_data_all['Value'] = hmi_data_all['Value'].interpolate()
		# ... reset index so we can work with Time in a normal way again
		hmi_data_all.reset_index(inplace = True)

		# Get the time elapsed between adjacent Values (dividing by np.timedelta64 converts to floating number)
		hmi_data_all['TimeEl'] = (hmi_data_all['Time'].shift(-1) - hmi_data_all['Time'])/np.timedelta64(1,'m')
		# Compute the area under the curve for each timestep (relative to the next time step)
		hmi_data_all['TotValue'] = hmi_data_all['Value']*hmi_data_all['TimeEl']

		# Get the timedelta/datetime64 string from the ttype input argument (either 'h' or 'm')
		ttype_d = ttype[0].lower()

		# Calculate the "Time Category" variable which indicates the time range for the observation
		hmi_data_all['TimeCat'] = \
			np.floor(
				(hmi_data_all['Time'] - self.start_dt)/\
				np.timedelta64(tperiod, ttype_d)
			)

		# Group by time range and sum the TotValue variable!
		tots_res = hmi_data_all.groupby('TimeCat').sum()
		tots_res.reset_index(inplace = True)


		# Retrieve the timestep from the TimeCat Variable
		tots_res['TimeCat'] = pd.to_timedelta(tots_res['TimeCat']*tperiod, ttype_d)
		tots_res['Time'] = self.start_dt + tots_res['TimeCat']
		# Get average value for the time period (want to correct for whether the tperiod is 1 minute vs 1 hour (i.e. 60 minutes))
		tperiod_hrs = tperiod
		if ttype == 'MINUTE':
			tperiod_hrs = tperiod/60
		tots_res['Value'] = tots_res['TotValue']/(tperiod_hrs*60)

		# Set data to minute-level resolution (bug in datetime or pandas can offset start_dt + TimeCat by a couple seconds)
		tots_res['Time'] = tots_res['Time'].values.astype('datetime64[m]')

		# Output
		return tots_res[['Time','Value']]


	def run_report(
		self,
		tperiods,
		ttypes,
		elids,
		stypes,
		output_csv = False,
		output_sql = True
	):

		# Retrieve sql table directory
		os.chdir(self.data_dir)

		# Clean inputs
		ttypes = [ttype.upper() for ttype in ttypes]
		stypes = [stype.upper() for stype in stypes]

		for tperiod, ttype, elid, stype in zip(tperiods, ttypes, elids, stypes):

			print('Getting aggregated data for {0} ({1}{2})...'.format(elid, tperiod, ttype))

			# Get prepped data
			self.prep_data(elid, stype)
			# Get totalized values
			tots_res = self.get_tot_var(tperiod, ttype, elid)
			# Get month integer allows data partitioning
			tots_res['Month'] = tots_res['Time'].dt.month
			# Get years for which report is run 
			# (only matters if the time period is spread over more than one calendar year)
			years = np.unique(tots_res['Time'].dt.year.values)

			# Reorder columns
			tots_res = tots_res[['Time','Month','Value']]

			# Output data as desired
			if output_sql:

				# SQL command strings for sqlite3
				create_str = """
					CREATE TABLE IF NOT EXISTS {0}_{1}{2}_AVERAGES (Tkey INT PRIMARY KEY, Time , Month, Value)
				""".format(elid, tperiod, ttype)
				ins_str = """
					INSERT OR REPLACE INTO {0}_{1}{2}_AVERAGES (Tkey, Time, Month, Value)
					VALUES (?,?,?,?)
				""".format(elid, tperiod, ttype)

				# Output data to sql database pertaining to its year
				for year in years:

					# Set connection to SQL database (pertaining to given year)
					conn = sqlite3.connect('cr2c_hmi_agg_data_{0}.db'.format(year))

					# Subset the data to its year
					tots_res_yr = tots_res.loc[tots_res['Time'].dt.year == year,:]

					# Create key from "Time" variable to use when updating/inserting entry into sql table
					tots_res_yr['Tkey'] = tots_res_yr['Time']
					tots_res_yr = tots_res_yr[['Tkey','Time','Month','Value']]

					# Load data to SQL
					# Create the table if it doesn't exist
					conn.execute(create_str)
					# Insert aggregated values for the elid and time period
					conn.executemany(
						ins_str,
						tots_res_yr.to_records(index = False).tolist()
					)
					conn.commit()

					# Close Connection
					conn.close()

			if output_csv:
				os.chdir(self.hmi_dir)
				tots_res.to_csv('{0}_{1}{2}_AVERAGES.csv'.format(elid, tperiod, ttype), index = False, encoding = 'utf-8')


	def get_tmp_plots(
		self,
		start_dt_str,
		end_dt_str,
		outdir = None,
		opfile_suff = None
	):

		start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
		end_dt = dt.strptime(end_dt_str,'%m-%d-%y')

		if not outdir:
			tkTitle = 'Directory to output charts/tables to...'
			print(tkTitle)
			outdir = askdirectory(title = tkTitle)

		if opfile_suff:
			opfile_suff = '_' + opfile_suff
		else:
			opfile_suff = ''

		# Get feeding data
		feeding_dat_zm = get_data(['FT305'],[5],['minute'], start_dt.year, start_dt_str = start_dt_str, end_dt_str = end_dt_str)
		feeding_dat = get_data(['FT305'],[1],['hour'], start_dt.year, start_dt_str = start_dt_str, end_dt_str = end_dt_str)

		# Get tmp data
		tmp_dat_zm = get_data(['AIT302'],[5],['minute'], start_dt.year)
		tmp_dat = get_data(['AIT302'],[1],['hour'], start_dt.year)

		# Merge the two files
		tmp_feed_dat = feeding_dat.merge(tmp_dat, on = 'Time')
		tmp_feed_dat_zm = feeding_dat_zm.merge(tmp_dat_zm, on = 'Time')

		# Remove index!
		tmp_feed_dat.reset_index(inplace = True)
		tmp_feed_dat_zm.reset_index(inplace = True)

		# Group the dataset into days and weeks
		tmp_feed_dat['Week'] = tmp_feed_dat['Time'].dt.week
		tmp_feed_dat['tel_day'] = tmp_feed_dat['Time'].dt.hour*60 + tmp_feed_dat['Time'].dt.minute
		tmp_feed_dat['Day']  = tmp_feed_dat['Time'].dt.weekday
		tmp_feed_dat['Hour'] = tmp_feed_dat['Time'].dt.hour + tmp_feed_dat['Time'].dt.weekday*24
		tmp_feed_dat['Date'] = tmp_feed_dat['Time'].dt.date

		# Do the same for the "zoom" dataset
		tmp_feed_dat_zm['Week'] = tmp_feed_dat_zm['Time'].dt.week
		tmp_feed_dat_zm['tel_day'] = tmp_feed_dat_zm['Time'].dt.hour*60 + tmp_feed_dat_zm['Time'].dt.minute
		tmp_feed_dat_zm['Day']  = tmp_feed_dat_zm['Time'].dt.weekday
		tmp_feed_dat_zm['Hour'] = tmp_feed_dat_zm['Time'].dt.hour + tmp_feed_dat_zm['Time'].dt.weekday*24

		# Get data for last week
		tmp_feed_week = tmp_feed_dat.loc[
			tmp_feed_dat['Time'].dt.date - end_dt.date() >= \
			np.timedelta64(-6,'D'),
		]
		# For last week, get daily membrane flux (L/m2-hr)
		tmp_feed_week = tmp_feed_week.groupby('Date').sum()
		tmp_feed_week.reset_index(inplace = True)
		l_p_gal = 3.78541 # Liters/Gallon
		tmp_feed_week.loc[:,'Net Flux'] = tmp_feed_week['FT305']*60/(39.5*24)*l_p_gal

		# Get data for last week
		tmp_feed_day = tmp_feed_dat_zm.loc[
			tmp_feed_dat_zm['Time'].dt.date - end_dt.date() > \
			np.timedelta64(-1,'D'),
		]

		# Plot!
		sns.set_style('white')
		# Last 6 months (or entire date range)
		# TMP
		ax1 = plt.subplot2grid((16,1),(0,0), rowspan = 2)
		ax1.plot(tmp_feed_dat['Time'],tmp_feed_dat['AIT302'], 'g-', linewidth = 0.5)
		ax1.set_title(
			'Hourly Average TMP and Permeate Flow ({0} to {1})'.format(start_dt_str, end_dt_str),
			fontweight = 'bold'
		)
		ax1.set_ylabel('TMP (psia)')
		ax1.xaxis.set_ticklabels([])
		# Flow
		ax2 = plt.subplot2grid((16,1),(2,0), rowspan = 2)
		ax2.plot(tmp_feed_dat['Time'],tmp_feed_dat['FT305'], 'b-', linewidth = 0.5)
		ax2.set_ylabel('Flow (gpm)')
		labels = ax2.get_xticklabels()
		plt.setp(labels, rotation=45, fontsize=10)
		# Last day
		# TMP
		ax3 = plt.subplot2grid((16,1),(6,0), rowspan = 2)
		ax3.plot(tmp_feed_day['Time'],tmp_feed_day['AIT302'], 'g-', linewidth = 0.5)
		ax3.set_title(
			'Hourly Average TMP and Permeate Flow (last 24 hours)',
			fontweight = 'bold'
		)
		ax3.set_ylabel('TMP (psia)')
		ax3.xaxis.set_ticklabels([])
		# Flow
		ax4 = plt.subplot2grid((16,1),(8,0), rowspan = 2)
		ax4.plot(tmp_feed_day['Time'],tmp_feed_day['FT305'], 'b-', linewidth = 0.5)
		ax4.set_ylabel('Flow (gpm)')
		labels = ax4.get_xticklabels()
		plt.setp(labels, rotation=45, fontsize=10)
		# Average Daily flux for the last week
		ax5 = plt.subplot2grid((16,1),(12,0), rowspan = 4)
		ax5.plot(tmp_feed_week['Date'],tmp_feed_week['Net Flux'], 'b-', linewidth = 0.5)
		ax5.set_ylim((0,max(tmp_feed_week['Net Flux'].values)*1.1))
		ax5.set_ylabel('Net Flux (' + r'$L/m^2-hr$)')
		ax5.set_title(
			'Average Daily Net Membrane Flux (last 7 days)',
			fontweight = 'bold'
		)		
		labels = ax5.get_xticklabels()
		plt.setp(labels, rotation=45, fontsize=10)
		# Output plots and/or sumstats csv files to directory of choice
		plot_filename  = "FLOW_TMP{0}.png".format(opfile_suff)
		fig = matplotlib.pyplot.gcf()
		fig.set_size_inches(7, 12)

		plt.savefig(
			os.path.join(outdir, plot_filename),
			width = 20,
			height = 160
		)
		plt.close()


	def get_feed_sumst(
		self,
		stype,
		output_types,
		start_dt_str,
		end_dt_str,
		sum_period = 'DAY', 
		plt_type = None,
		plt_colors = None,
		ylabel = None,
		get_nhours = None,
		outdir = None,
		opfile_suff = None
	):


		start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
		end_dt = dt.strptime(end_dt_str,'%m-%d-%y')

		# Clean case of input arguments
		sum_period = sum_period.upper()
		if opfile_suff:
			opfile_suff = '_' + opfile_suff
		else:
			opfile_suff = ''

		plt_type = plt_type.upper()
		if type(output_types) == list:
			output_types = [output_type.upper() for output_type in output_types]
		else:
			output_types = output_types.upper()

		# Define HMI element ids according to query type (water or biogas)
		stype = stype.upper()
		if stype == 'GAS':
			elids = ['FT700','FT704']
		if stype == 'WATER':
			elids = ['FT202','FT305']
		if stype == 'TEMP':
			elids = ['AT304','AT310']

		# Get output directory and string with all element ids from report
		if not outdir:
			tkTitle = 'Directory to output charts/tables to...'
			print(tkTitle)
			outdir = askdirectory(title = tkTitle)

		feeding_dat = get_data(elids, [1,1],['hour','hour'], start_dt_str = start_dt_str, end_dt_str = end_dt_str)

		# Retrieve element ids from aggregated data
		all_elids = '_'.join(elids)

		# Get hourly flow totals for each elid
		for elid in elids:
			feeding_dat[elid] = feeding_dat[elid]*60

		# Convert Time variable to pd.datetime variable
		feeding_dat['Time'] = pd.to_datetime(feeding_dat['Time'])
		feeding_dat['Date'] = feeding_dat['Time'].dt.date

		# Filter to the dates desired for the plots
		feeding_dat = feeding_dat.loc[
			(feeding_dat.Time >= start_dt) &
			(feeding_dat.Time < end_dt + timedelta(days = 1))
		]

		# Get dataset aggregated by Day, Week or Month
		# Based on aggregation period, get the number of hours we are summing averages over (averages are in minutes)
		if sum_period == 'HOUR':
			xlabel = 'Time'
			nhours = 1
		else:
			feeding_dat['Date'] = feeding_dat['Time'].dt.date

		if sum_period == 'DAY':
			xlabel = 'Date'
			nhours = 24

		if sum_period == 'WEEK':
			xlabel = 'Weeks (since {0})'.format(start_dt_str)
			feeding_dat[xlabel] = np.floor((feeding_dat['Time'] - start_dt)/np.timedelta64(7,'D'))
			nhours = 24*7

		if sum_period == 'MONTH':
			xlabel = 'Months (since {0}, as 30 days)'.format(start_dt_str)
			feeding_dat[xlabel] = np.floor((feeding_dat['Time'] - start_dt)/np.timedelta64(30,'D'))
			nhours = 24*7*30

		if get_nhours == 1:
			for elid in elids:
				feeding_dat['Number Hours {0}'.format(elid)] = \
					np.where(feeding_dat[elid].values > 0, 1, 0)

		agg_sumst = feeding_dat.groupby(xlabel).sum()

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
			plot_filename  = "HMI{0}_{1}{2}.png".format(stype, all_elids, opfile_suff)
			plt.savefig(
				os.path.join(outdir, plot_filename),
				width = 20,
				height = 50
			)
			plt.close()

		if 'TABLE' in output_types:

			sumst_filename = "HMI{0}_{1}{2}.csv".format(stype, all_elids, opfile_suff)
			agg_sumst.reset_index(inplace = True)
			agg_sumst = agg_sumst[[xlabel] + elids]
			agg_sumst.to_csv(
				os.path.join(outdir, sumst_filename),
				index = False,
				encoding = 'utf-8'
			)

	def get_temp_plots(self, end_dt_str, outdir = None, opfile_suff = None, plt_colors = None):

		elids = ['AT304','AT310']

		end_dt = dt.strptime(end_dt_str,'%m-%d-%y')
		start_dt = end_dt - timedelta(days = 180)
		start_dt_str = dt.strftime(start_dt,'%m-%d-%y')

		if not outdir:
			tkTitle = 'Directory to output charts/tables to...'
			print(tkTitle)
			outdir = askdirectory(title = tkTitle)

		if opfile_suff:
			opfile_suff = '_' + opfile_suff
		else:
			opfile_suff = ''

		# Get temperature data
		temp_dat = get_data(elids,[1,1],['hour','hour'], start_dt_str = start_dt_str, end_dt_str = end_dt_str)
		temp_dat.loc[:,'Date'] = temp_dat['Time'].dt.date
		
		# Daily average for the last 6 months
		temp_dat_dly = temp_dat.groupby('Date').mean()
		temp_dat_dly.reset_index(inplace = True)

		# Hourly average for the last week
		temp_dat_week = temp_dat.loc[
			temp_dat['Date'] - end_dt.date() >= \
			np.timedelta64(-6,'D'),
		]

		# Plot daily average
		ax1 = plt.subplot2grid((8,1),(0,0), rowspan = 3)
		ax1.plot(temp_dat_dly['Date'],temp_dat_dly['AT304'], 'g-', linewidth = 0.5, color = plt_colors[0])
		ax1.plot(temp_dat_dly['Date'],temp_dat_dly['AT310'], 'g-', linewidth = 0.5, color = plt_colors[1])
		plt.set_title(
			'Mean Daily Temperature ({0} to {1})'.format(start_dt_str, end_dt_str),
			fontweight = 'bold'
		)
		ax1.set_ylabel('Temperature (°C)')
		labels = ax1.get_xticklabels()
		plt.setp(labels, rotation = 45, fontsize = 10)
		# Plot hourly average
		ax2 = plt.subplot2grid((8,1),(4,0), rowspan = 3)
		afbrPlt = ax2.plot(temp_dat_week['Time'],temp_dat_week['AT304'], 'b-', linewidth = 0.5, color = plt_colors[0])
		afmbrPlt = ax2.plot(temp_dat_week['Time'],temp_dat_week['AT310'], 'b-', linewidth = 0.5, color = plt_colors[1])
		ax2.set_title(
			'Mean Hourly Temperature (last 7 days)'.format(start_dt_str, end_dt_str),
			fontweight = 'bold'
		)
		ax2.set_ylabel('Temperature (°C)')
		labels = ax2.get_xticklabels()
		plt.setp(labels, rotation = 45, fontsize = 10)
		lgd = ax2.legend(
			(afbrPlt[0],afmbrPlt[0]),
			('AFBR','AFMBR'),
			loc = 'center',
			bbox_to_anchor = (0.5, -0.5), 
			fancybox = True, 
			shadow = True, 
			ncol = 2
		)
		# Output plot to directory of choice
		plot_filename  = "Temperature{0}.png".format(opfile_suff)
		fig = matplotlib.pyplot.gcf()
		fig.set_size_inches(7, 8)

		plt.savefig(
			os.path.join(outdir, plot_filename),
			width = 20,
			height = 80
		)
		plt.close()



