'''
	This script calculates totals and averages for any given op data point(s),
	time period, and date range for which a raw eDNA query has been run (and a csv file
	for that query obtained)
	If desired, also outputs plots and summary tables
'''

from __future__ import print_function

# Data Prep
import numpy as np
import pandas as pd
import datetime as datetime
from datetime import datetime as dt
from datetime import timedelta
from pandas import read_excel
import sqlite3

# Utilities
import os
from os.path import expanduser
import sys
import traceback as tb
import warnings as wn

# CR2C
import cr2c_utils as cut


def get_data(
	stypes,
	sids, 
	tperiods, 
	ttypes,
	combine_all = True,
	year_sub = None, 
	month_sub = None, 
	start_dt_str = None, 
	end_dt_str = None, 
	output_csv = False, 
	outdir = None
):

	# Convert date string inputs to dt variables
	start_dt = dt.strptime('5-10-17','%m-%d-%y')
	end_dt = dt.now()
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

	# Find Operational Data directory and change working directory
	data_dir = cut.get_dirs()[0]
	os.chdir(data_dir)

	# Initialize output data variable
	if combine_all:
		opdata_all = pd.DataFrame()
	else:
		opdata_all = {}

	# Manage data selection input 
	nsids = len(sids)
	if nsids != len(stypes) or nsids != len(tperiods) or nsids != len(ttypes):
		print('Error in cr2c_opdata: get_data: The lengths of the sids, stypes, tperiods and ttypes arguments must be equal')
		sys.exit()

	# Manage month and year subset input (will be added to sqlite3 query string)
	sub_ins = ''
	if year_sub and month_sub:
		sub_ins = 'WHERE YEAR == {} AND Month == {}'.format(year_sub, month_sub)
	elif month_sub:
		sub_ins = 'WHERE Month == {}'.format(month_sub)
	elif year_sub:
		sub_ins = 'WHERE Year == {}'.format(year_sub)

	for sid, stype, tperiod, ttype in zip(sids, stypes, tperiods, ttypes):

		sql_str = """
			SELECT distinct * FROM {0}_{1}_{2}_{3}_AVERAGES {4}
			order by Time 
		""".format(stype, sid, tperiod, ttype, sub_ins)

		# Open connection and read to pandas dataframe
		conn = sqlite3.connect('cr2c_opdata.db')
		opdata = pd.read_sql(
			sql_str,
			conn,
			coerce_float = True
		)

		# Format the time variable
		opdata['Time'] = pd.to_datetime(opdata['Time'])
		# Set time variable
		if ttype == 'HOUR':
			opdata.loc[:,'Time'] = opdata['Time'].values.astype('datetime64[h]')
		elif ttype == 'MINUTE':
			opdata.loc[:,'Time'] = opdata['Time'].values.astype('datetime64[m]')

		# Drop duplicates (happens with hourly aggregates sometimes...)
		opdata.drop_duplicates(['Time'], inplace = True)

		if start_dt_str:
			opdata = opdata.loc[opdata['Time'] >= start_dt,]
		if end_dt_str:
			opdata = opdata.loc[opdata['Time'] < end_dt + timedelta(days = 1),]

		# If returning all as a single dataframe, merge the result in loop (or initialize dataframe)
		if combine_all:

			# Rename Value variable to its corresponding Sensor ID
			opdata.rename(columns = {'Value': sid}, inplace = True)
			if not len(opdata_all):
				opdata_all = opdata
			else:
				opdata_all = opdata_all.merge(opdata[['Time', sid]], on = 'Time', how = 'outer')
		
		# Otherwise, load output to dictionary
		else:
			opdata_all['{0}_{1}_{2}_{3}_AVERAGES'.format(stype, sid, tperiod, ttype)] = opdata

	if combine_all and output_csv:

		os.chdir(outdir)
		op_fname = '_'.join(sids + [str(tperiod) for tperiod in tperiods]) + '.csv'
		opdata_all.to_csv(op_fname, index = False, encoding = 'utf-8')

	return opdata_all


# Returns a list of the tables in the op SQL database
def	get_table_names():

	# Create connection to SQL database
	data_dir = cut.get_dirs()[0]
	os.chdir(data_dir)
	conn = sqlite3.connect('cr2c_opdata.db')
	cursor = conn.cursor()
	# Execute
	cursor.execute(""" SELECT name FROM sqlite_master WHERE type ='table'""")

	return [names[0] for names in cursor.fetchall()]


# Takes a list of file paths and concatenates all of the files
def cat_dfs(ip_paths, idx_var = None, output_csv = False, outdir = None, output_dsn = None):
	
	concat_dlist = []
	for ip_path in ip_paths:
		concat_dlist.append(pd.read_csv(ip_path, low_memory = False))
	concat_data = pd.concat([df for df in concat_dlist], ignore_index = True)
	# Remove duplicates (may be some overlap)
	concat_data.drop_duplicates(keep = 'first', inplace = True)
	
	# Sort by index (if given)
	if idx_var:
		concat_data.sort_values(idx_var, inplace = True)

	if output:

		concat_data.to_csv(
			os.path.join(outdir, output_dsn), 
			index = False, 
			encoding = 'utf-8'
		)
	
	return concat_data


# Primary op data aggregation class
class opdata_agg:

	def __init__(self, start_dt_str, end_dt_str, ip_path):

		self.start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
		self.end_dt = dt.strptime(end_dt_str,'%m-%d-%y') + timedelta(days = 1)
		self.data_dir = cut.get_dirs()[0]
		self.ip_path = ip_path


	def prep_opdata(self, stype, sid):

		# This is the type of query (unlikely to change)
		qtype = 'RAW'

		# Read in raw op data
		try:
			self.opdata = pd.read_csv(self.ip_path, low_memory = False)
		except Exception as e:
			print('\nThere was an error reading in the op data:\n')
			tb.print_exc(file = sys.stdout)
			tb.print_exc(limit = 1, file = sys.stdout)
			sys.exit()
		
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
		elif stype == 'DPI':
			hi_limit = 40
			lo_limit = -40
		elif stype == 'LEVEL':
			hi_limit = 100
			lo_limit = 0
			
		# Load variables and set output variable names
		varname = 'CR2C.CODIGA.{0}.SCALEDVALUE {1} [{2}]'
		# Rename value variable
		self.opdata.loc[:,'Value'] = \
			self.opdata[varname.format(sid,'Value', qtype)]

		# Set low/negative values to 0 (if a flow, otherwise remove) and remove unreasonably high values
		if stype in ['GAS','WATER','LEVEL']:
			self.opdata.loc[self.opdata['Value'] < lo_limit, 'Value'] = 0
		else:
			self.opdata.loc[self.opdata['Value'] < lo_limit, 'Value'] = np.NaN

		self.opdata.loc[self.opdata['Value'] > hi_limit, 'Value'] = np.NaN

		# Rename and format corresponding timestamp variable
		self.opdata.loc[:,'Time' ] = \
			self.opdata[varname.format(sid, 'Time', qtype)]
		# Subset to "Time" and "Value" variables
		self.opdata = self.opdata.loc[:,['Time','Value']]
		# Eliminate missing values and reset index
		self.opdata.dropna(axis = 0, how = 'any', inplace = True)

		# Set Time as datetime variable at second resolution (uses less memory than nanosecond!)
		self.opdata.loc[:,'Time' ] = \
			pd.to_datetime(self.opdata['Time']).values.astype('datetime64[s]')
		# Create datetime index
		self.opdata.set_index(pd.DatetimeIndex(self.opdata['Time']), inplace = True)
		# Remove Time variable from dataset
		self.opdata.drop('Time', axis = 1, inplace = True)
		# Get first and last available time stamps in index
		self.first_ts, self.last_ts = self.opdata.index[0], self.opdata.index[-1]

		# Check to make sure that the totals/averages do not include the first
		# and last days for which data are available (just to ensure accuracy)
		if self.first_ts >= self.start_dt or self.last_ts <= self.end_dt:

			# Set dates for warning message (set to 0:00 of the given day)
			self.start_dt_warn = self.first_ts + timedelta(days = 1)
			self.start_dt_warn = datetime.datetime(self.start_dt_warn.year, self.start_dt_warn.month, self.start_dt_warn.day)
			self.end_dt_warn   = self.last_ts - timedelta(days = 1)

			# Issue warning
			msg = \
				'Given the range of data available for {0}, accurate aggregate values can only be obtained for: {1} to {2}'
			wn.warn(msg.format(sid, dt.strftime(self.start_dt_warn, '%m-%d-%y'), dt.strftime(self.end_dt_warn, '%m-%d-%y')))
			# Change start_dt and end_dt of system to avoid overwriting sql file with empty data
			self.start_dt = datetime.datetime(self.first_ts.year, self.first_ts.month, self.first_ts.day) + timedelta(days = 1)
			# Need to set the self.end_dt to midnight of the NEXT day
			self.end_dt = datetime.datetime(self.last_ts.year, self.last_ts.month, self.last_ts.day) 

		return self.opdata


	def get_average(self, opdata, tperiod, ttype):

		# Get minute-level dataframe of timesteps for the time period requested
		ts_array = np.arange(
			self.start_dt,
			self.end_dt,
			np.timedelta64(1,'m')
		)
		empty_df = pd.DataFrame(ts_array, columns = ['Time'])
		empty_df.set_index(pd.DatetimeIndex(ts_array), inplace = True)
		# Merge this with the op data and fill in NaNs by interpolating
		opdata_all = opdata.merge(empty_df, how = 'outer', left_index = True, right_index = True)
		opdata_all.loc[:,'Value'] = opdata_all['Value'].interpolate()	
		# Create time variable from index values
		opdata_all.loc[:,'Time'] = opdata_all.index.values
		# Get the time elapsed between adjacent Values (in minutes, dividing by np.timedelta64 converts to floating number)
		opdata_all['TimeEl'] = (opdata_all['Time'].shift(-1) - opdata_all['Time'])/np.timedelta64(1,'m')
		# Subset to the time period desired (AFTER interpolating and computing the TimeEl variable)
		opdata_all = opdata_all.loc[self.start_dt:self.end_dt]

		# Get the timedelta/datetime64 string from the ttype input argument (either 'h' or 'm')
		ttype_d = ttype[0].lower()
		# Calculate the "Time Category" variable which indicates the time range for the observation
		opdata_all['TimeCat'] = \
			np.floor(
				(opdata_all['Time'] - self.start_dt)/\
				np.timedelta64(tperiod, ttype_d)
			)
		# Group by time range and compute a weighted average with "TimeEl" as the weight
		tots_res = \
			opdata_all.groupby('TimeCat').\
			apply(lambda x: np.average(x.Value, weights = x.TimeEl))
		tots_res = pd.DataFrame(tots_res, columns = ['Value'])
		tots_res.reset_index(inplace = True)

		# Retrieve the timestep from the TimeCat Variable
		tots_res['TimeCat'] = pd.to_timedelta(tots_res['TimeCat']*tperiod, ttype_d)
		tots_res['Time'] = self.start_dt + tots_res['TimeCat']
		# Set data to minute-level resolution (bug in datetime or pandas can offset start_dt + TimeCat by a couple seconds)
		tots_res['Time'] = tots_res['Time'].values.astype('datetime64[m]')
		# Subset to Time, Value and time range for which reliable aggregated values can be obtained
		tots_res = tots_res.loc[:,['Time','Value']]

		# Output
		return tots_res


	def run_agg(self, stypes, sids, tperiods, ttypes, output_csv = False, output_sql = True, outdir = None):

		# Get sql table directory
		os.chdir(self.data_dir)

		# Clean inputs
		ttypes, stypes = [ttype.upper() for ttype in ttypes], [stype.upper() for stype in stypes]

		for tperiod, ttype, sid, stype in zip(tperiods, ttypes, sids, stypes):

			print('Getting aggregated data for {0} ({1}{2})...'.format(sid, tperiod, ttype))

			# Get prepped data
			self.prep_opdata(stype, sid)
			# Get totalized values
			tots_res = self.get_average(self.opdata, tperiod, ttype)
			# Get year and month (for partitioning purposes)
			tots_res.loc[:,'Year'] = tots_res['Time'].dt.year
			tots_res.loc[:,'Month'] = tots_res['Time'].dt.month
			# Reorder columns
			tots_res = tots_res[['Time','Year','Month','Value']].copy()

			# Output data as desired
			if output_sql:

				# SQL command strings for sqlite3
				create_str = """
					CREATE TABLE IF NOT EXISTS {0}_{1}_{2}_{3}_AVERAGES (Time INT PRIMARY KEY, Year , Month, Value)
				""".format(stype, sid, tperiod, ttype)
				ins_str = """
					INSERT OR REPLACE INTO {0}_{1}_{2}_{3}_AVERAGES (Time, Year, Month, Value)
					VALUES (?,?,?,?)
				""".format(stype, sid, tperiod, ttype)
				# Set connection to SQL database (pertaining to given year)
				conn = sqlite3.connect('cr2c_opdata.db')
				# Load data to SQL
				# Create the table if it doesn't exist
				conn.execute(create_str)
				# Insert aggregated values for the sid and time period
				conn.executemany(
					ins_str,
					tots_res.to_records(index = False).tolist()
				)
				conn.commit()
				# Close Connection
				conn.close()

			if output_csv:
				if not outdir:
					outdir = askdirectory()
				os.chdir(outdir)
				tots_res.to_csv('{0}_{1}_{2}_{3}_AVERAGES.csv'.\
					format(stype, sid, tperiod, ttype), index = False, encoding = 'utf-8')

