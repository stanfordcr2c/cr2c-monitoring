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
import get_lab_data as gld
import sqlite3
import os
from os.path import expanduser
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
		elif self.stype == 'TEMP':
			hi_limit = 50
			lo_limit = 0
		elif self.stype == 'PRESSURE':
			hi_limit = 16
			lo_limit = 13.4
		elif self.stype == 'TMP':
			hi_limit = 20
			lo_limit = -20

		# Load data
		try:
			self.hmi_data = pd.read_csv(self.hmi_path)
		except FileNotFoundError:
			print('Please choose an existing input file with the HMI data')
			sys.exit()

		# Load variables and set output variable names
		varname = 'CR2C.CODIGA.{0}.SCALEDVALUE {1} [{2}]'

		# Rename variable
		self.hmi_data['Value'] = \
			self.hmi_data[varname.format(elid,'Value', self.qtype)]
		# Set low/negative values to 0 (if a flow, otherwise remove) and remove unreasonably high values
		if self.stype in ['GAS','WATER']:
			self.hmi_data.loc[self.hmi_data['Value'] < lo_limit, 'Value'] = 0
		else:
			self.hmi_data.loc[self.hmi_data['Value'] < lo_limit, 'Value'] = np.NaN	
		self.hmi_data.loc[self.hmi_data['Value'] > hi_limit, 'Value'] = np.NaN	

		# Rename and format corresponding timestamp variable 
		self.hmi_data['Time' ] = \
			self.hmi_data[varname.format(elid, 'Time', self.qtype)]
		self.hmi_data['Time' ] = \
			pd.to_datetime(self.hmi_data['Time'])

		# Filter dataset to clean values, time period and variable selected
		self.hmi_data = self.hmi_data.loc[
			(self.hmi_data['Time'] >= self.start_dt - datetime.timedelta(days = 1)) &
			(self.hmi_data['Time'] <= self.end_dt + datetime.timedelta(days = 1))
			, 
			['Time', 'Value']
		]
		# Eliminate missing values and reset index
		self.hmi_data.dropna(axis = 0, how = 'any', inplace = True)
		self.hmi_data.reset_index(inplace = True)

		# Get numeric time elapsed
		self.first_ts = self.hmi_data['Time'][0]
		self.last_ts  = self.hmi_data['Time'][len(self.hmi_data) - 1]

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
		# ... need to set Time as an index to do this
		hmi_data_all.set_index('Time')
		hmi_data_all.loc[:,'Value'] = hmi_data_all['Value'].interpolate()
		hmi_data_all.sort_values('Time',inplace = True)
		# ... reset index so we can work with Time in a normal way again
		hmi_data_all.reset_index(inplace = True)

		# Get the time elapsed between adjacent Values (dividing by np.timedelta64 converts to floating number)
		hmi_data_all['TimeEl'] = (hmi_data_all['Time'].shift(-1) - hmi_data_all['Time'])/np.timedelta64(1,'m')
		# Compute the area under the curve for each timestep (relative to the next time step)
		hmi_data_all['TotValue'] = hmi_data_all['Value']*hmi_data_all['TimeEl']
		
		# Extract the timedelta/datetime64 string from the ttype input argument (either 'h' or 'm')
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
		# Get average value for the time period (converting tdelta to minutes since observations)
		tperiod_hrs = tperiod
		if ttype == 'MINUTE':
			tperiod_hrs = tperiod/60
		tots_res['Value'] = tots_res['TotValue']/(tperiod_hrs*60)

		# Output
		return tots_res[['Time','Value']]


	def run_report(
		self,
		tperiods,
		ttypes,
		elids,
		start_dt_str,
		end_dt_str,
		hmi_path = None,
		output_csv = False,
		output_sql = True
	):

		# Select input data file
		if hmi_path:
			self.hmi_path = hmi_path
			self.hmi_dir = os.path.dirname(hmi_path)
		else:
			self.hmi_path = askopenfilename(title = 'Select HMI data input file')
			self.hmi_dir = os.path.dirname(self.hmi_path)

		# Get dates and date strings for output filenames
		self.start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
		self.end_dt = dt.strptime(end_dt_str,'%m-%d-%y')

		# Retrieve sql table directory
		table_dir = gld.get_indir()
		os.chdir(table_dir)

		# Open connection to sql database file
		if output_sql:
			conn = sqlite3.connect('cr2c_hmi_agg_data_{0}.db'.format(self.start_dt.year))

		ttypes = [ttype.upper() for ttype in ttypes]

		for elid, tperiod, ttype in zip(elids, tperiods, ttypes):

			# Get prepped data
			self.prep_data(elid)
			# Get totalized values'

			tots_res = self.get_tot_var(tperiod, ttype, elid)
			# Get month integer (for possible partitioning later on)
			tots_res['Month'] = tots_res['Time'].dt.month
			
			# Reorder columns and set time index
			# tots_res.set_index(tots_res['Time'], inplace = True)
			tots_res = tots_res[['Time','Month','Value']]

			# Output data as desired
			if output_sql:

				# SQL command strings for sqlite3
				create_str = """
					CREATE TABLE IF NOT EXISTS {0}_{1}{2}_{3}S (Time , Month, Value)
				"""

				insert_str = """
					INSERT OR REPLACE INTO {0}_{1}{2}_{3}S (Time, Month, Value)
					VALUES (?,?,?)
				"""

				# Load data to SQL
				# Create the table if it doesn't exist
				conn.execute(create_str.format(elid, tperiod, ttype, 'AVERAGE'))
				# Insert aggregated values for the elid and time period
				conn.executemany(
					insert_str.format(elid, tperiod, ttype, 'AVERAGE'),
					tots_res.to_records(index = False).tolist()
				)
				conn.commit()

			if output_csv:
				os.chdir(self.hmi_dir)
				tots_res.to_csv('{0}_{1}{2}_{3}S.csv'.format(elid, tperiod, ttype, 'AVERAGE'), index = False, encoding = 'utf-8')

		# Close connection to sql database file
		if output_sql:
			conn.close()

# Manages output directories
def get_indir():
	
	# Find the CR2C.Operations folder on Box Sync on the given machine
	targetdir = os.path.join('Box Sync','CR2C.Operations')
	mondir = None
	print("Searching for Codiga Center's Operations folder on Box Sync...")
	for dirpath, dirname, filename in os.walk(expanduser('~')):
		if dirpath.find(targetdir) > 0:
			mondir = os.path.join(dirpath,'MonitoringProcedures')
			print("Found Codiga Center's Operations folder on Box Sync")
			break
			
	# Alert user if Box Sync folder not found on machine
	if not mondir:
		if os.path.isdir('D:/'):
			for dirpath, dirname, filename in os.walk('D:/'):
				if dirpath.find(targetdir) > 0:
					mondir = os.path.join(dirpath,'MonitoringProcedures')
					print("Found Codiga Center's Operations folder on Box Sync")
					break
		if not mondir:
			print("Could not find Codiga Center's Operations folder in Box Sync")
			print('Please make sure that Box Sync is installed and the Operations folder is synced on your machine')
			sys.exit()
	
	return os.path.join(mondir,'Data')

def get_data(elids, tperiods, ttypes, year, month_sub = None, start_dt_str = None, end_dt_str = None):

	data_indir = get_indir()

	# Clean user inputs
	ttypes = [ttype.upper() for ttype in ttypes]

	# Convert date string inputs to dt variables
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

	# Load data from SQL
	os.chdir(data_indir)
	conn = sqlite3.connect('cr2c_hmi_agg_data_{}.db'.format(year))
	hmi_data_all = {}

	for elid, tperiod, ttype in zip(elids, tperiods, ttypes):

		if month_sub:

			sql_str = """
				SELECT * FROM {0}_{1}{2}_AVERAGES
				WHERE Month = {4}
			""".format(elid, tperiod, ttype, month_sub)

		else:

			sql_str = "SELECT * FROM {0}_{1}{2}_AVERAGES".format(elid, tperiod, ttype)

		hmi_data = pd.read_sql(
			sql_str, 
			conn, 
			coerce_float = True
		)

		# Dedupe data (some issue with duplicates)
		hmi_data.drop_duplicates(inplace = True)
		hmi_data.sort_values('Time', inplace = True)
		# Format the time variable
		hmi_data['Time'] = pd.to_datetime(hmi_data['Time'])

		if start_dt_str:
			hmi_data = hmi_data.loc[hmi_data['Time'] >= start_dt,]
		if end_dt_str:
			hmi_data = hmi_data.loc[hmi_data['Time'] <= end_dt,]

		hmi_data_all['{0}_{1}{2}_AVERAGES'.format(elid, tperiod, ttype, month_sub)] = hmi_data

	return hmi_data_all


if __name__ == '__main__':

	hmi_dat = hmi_data_agg(
		'raw', # Type of eDNA query (case insensitive, can be raw, 1 min, 1 hour)
		'gas' # Type of sensor (case insensitive, can be water, gas, pH, conductivity or temperature
	)
	hmi_dat.run_report(
		[1,1], # Number of hours you want to average over
		['hour','hour'], # Type of time period (can be "hour" or "minute")
		['FT700','FT704'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
		'5-11-17', # Start of date range you want summary data for
		'8-20-17' # End of date range you want summary data for)
	)
	hmi_dat = hmi_data_agg(
		'raw', # Type of eDNA query (case insensitive, can be raw, 1 min, 1 hour)
		'water' # Type of sensor (case insensitive, can be water, gas, pH, conductivity or temperature
	)
	hmi_dat.run_report(
		[1,1,5], # Number of time periods you want to average over
		['hour','hour','minute'], # Type of time period (can be "hour" or "minute")
		['FT202','FT305','FT305'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
		'5-11-17', # Start of date range you want summary data for
		'8-20-17' # End of date range you want summary data for)
	)
	hmi_dat = hmi_data_agg(
		'raw', # Type of eDNA query (case insensitive, can be raw, 1 min, 1 hour)
		'tmp' # Type of sensor (case insensitive, can be water, gas, pH, conductivity or temperature
	)
	hmi_dat.run_report(
		[5], # Number of hours you want to average over
		['minute'],
		['AIT302'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
		'5-11-17', # Start of date range you want summary data for
		'8-20-17' # End of date range you want summary data for)
	)


