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

	# Initialize output data variable
	opdata_all = {}
	for stype, sid, tperiod, ttype in zip(stypes, sids, tperiods, ttypes):

		sql_str = """
			SELECT distinct * FROM {}_{}_{}_{}_AVERAGES {}
			order by Time 
		""".format(stype, sid, tperiod, ttype, sub_ins)

		# Load data from google BigQuery
		projectid = 'cr2c-monitoring'
		dataset_id = 'opdata'
		opdata = pd.read_gbq(
			'SELECT * FROM {}.{}_{}_{}_{}_AVERAGES'.format(dataset_id, stype, sid, tperiod, ttype), 
			projectid
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
		opdata.dropna(subset = ['Time'], inplace = True)

		if start_dt_str:
			opdata = opdata.loc[opdata['Time'] >= start_dt,]
		if end_dt_str:
			opdata = opdata.loc[opdata['Time'] < end_dt + timedelta(days = 1),]

		# If returning all as a single dataframe, rename 'Value' to the Sensor ID
		if combine_all:
			# Rename Value variable to its corresponding Sensor ID
			opdata.rename(columns = {'Value': sid}, inplace = True)
			opdata.set_index(['Time','Year','Month'], inplace = True)
		elif output_csv:
			op_fname = 'cr2c_opdata_{}_{}_{}_{}.csv'.format(stype, sid, tperiod, ttype)
			op_path = os.path.join(outdir, op_fname)
			opdata.to_csv(op_path, index = False, encoding = 'utf-8')

		opdata_all['{}_{}_{}_{}_AVERAGES'.format(stype, sid, tperiod, ttype)] = opdata

	if combine_all:

		opdata_all = pd.concat(
			[df for df in list(opdata_all.values())], 
			axis = 1, 
			ignore_index = False,
			sort = True
		) 
		opdata_all.reset_index(inplace = True)
	
		if output_csv:

			stypes = list(set(stypes))
			op_fname = 'cr2c_opdata_' + '_'.join(stypes) + '.csv'
			opdata_all.to_csv(os.path.join(outdir, op_fname), encoding = 'utf-8')

	return opdata_all


# Takes a list of file paths and concatenates all of the files
def cat_dfs(ip_paths, idx_var = None, output_csv = False, outdir = None, out_dsn = None):
	
	concat_dlist = []
	for ip_path in ip_paths:
		concat_dlist.append(pd.read_csv(ip_path, low_memory = False))
	concat_data = pd.concat([df for df in concat_dlist], ignore_index = True, sort = True)
	# Remove duplicates (may be some overlap)
	concat_data.drop_duplicates(keep = 'first', inplace = True)

	if output:

		concat_data.to_csv(
			os.path.join(outdir, out_dsn), 
			index = False, 
			encoding = 'utf-8'
		)
	
	return concat_data


# Primary op data aggregation class
class opdata_agg:

	def __init__(self, start_dt_str, end_dt_str, ip_path):

		self.start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
		self.end_dt = dt.strptime(end_dt_str,'%m-%d-%y') + timedelta(days = 1)
		self.ip_path = ip_path


	def prep_opdata(self, stype, sid):

		# This is the type of query (unlikely to change)
		qtype = 'RAW'

		# Read in raw op data
		try:
			opdata = pd.read_csv(self.ip_path, low_memory = False)
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
		elif stype == 'COND':
			hi_limit = 5000
			lo_limit = 0
			
		# Load variables and set output variable names
		varname = 'CR2C.CODIGA.{}.SCALEDVALUE {} [{}]'
		# Rename value variable
		opdata.loc[:,'Value'] = \
			opdata[varname.format(sid,'Value', qtype)]

		# Set low/negative values to 0 (if a flow, otherwise remove) and remove unreasonably high values
		if stype in ['GAS','WATER','LEVEL']:
			opdata.loc[opdata['Value'] < lo_limit, 'Value'] = 0
		else:
			opdata.loc[opdata['Value'] < lo_limit, 'Value'] = np.NaN

		opdata.loc[opdata['Value'] > hi_limit, 'Value'] = np.NaN

		# Rename and format corresponding timestamp variable
		opdata.loc[:,'Time' ] = \
			opdata[varname.format(sid, 'Time', qtype)]
		# Subset to "Time" and "Value" variables
		opdata = opdata.loc[:,['Time','Value']]
		# Eliminate missing values and reset index
		opdata.dropna(axis = 0, how = 'any', inplace = True)

		# Set Time as datetime variable at second resolution (uses less memory than nanosecond!)
		opdata.loc[:,'Time' ] = \
			pd.to_datetime(opdata['Time']).values.astype('datetime64[s]')
		# Create datetime index
		opdata.set_index(pd.DatetimeIndex(opdata['Time']), inplace = True)
		# Remove Time variable from dataset
		opdata.drop('Time', axis = 1, inplace = True)

		if opdata.empty:

			return opdata

		else:

			# Get first and last available time stamps in index
			first_ts, last_ts = opdata.index[0], opdata.index[-1]

			# Check to make sure that the totals/averages do not include the first
			# and last days for which data are available (just to ensure accuracy)
			if first_ts >= self.start_dt or last_ts <= self.end_dt:

				# Set dates for warning message (set to 0:00 of the given day)
				start_dt_warn = first_ts + timedelta(days = 1)
				start_dt_warn = datetime.datetime(start_dt_warn.year, start_dt_warn.month, start_dt_warn.day)
				end_dt_warn   = last_ts - timedelta(days = 1)
				end_dt_warn = datetime.datetime(end_dt_warn.year, end_dt_warn.month, end_dt_warn.day)
				# Issue warning
				msg = \
					'Given the range of data available for {}, accurate aggregate values can only be obtained for: {} to {}'
				wn.warn(msg.format(sid, dt.strftime(start_dt_warn, '%m-%d-%y'), dt.strftime(end_dt_warn, '%m-%d-%y')))
				# Change start_dt and end_dt of system to avoid overwriting sql file with empty data
				self.start_dt = start_dt_warn
				# Need to set the self.end_dt to midnight of the NEXT day (so NOT end_dt_warn)
				self.end_dt = datetime.datetime(last_ts.year, last_ts.month, last_ts.day) 

			return opdata


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

		# Clean inputs
		ttypes, stypes = [ttype.upper() for ttype in ttypes], [stype.upper() for stype in stypes]

		for tperiod, ttype, sid, stype in zip(tperiods, ttypes, sids, stypes):

			print('Getting aggregated data for {} ({}{})...'.format(sid, tperiod, ttype))

			# Get prepped data
			opdata = self.prep_opdata(stype, sid)
			# Get totalized values
			tots_res = self.get_average(opdata, tperiod, ttype)
			# Get year and month (for partitioning purposes)
			tots_res.loc[:,'Year'] = tots_res['Time'].dt.year
			tots_res.loc[:,'Month'] = tots_res['Time'].dt.month
			# Reorder columns
			tots_res = tots_res[['Time','Year','Month','Value']].copy()

			if output_csv:
				if not outdir:
					outdir = askdirectory()
				out_dsn = '{}_{}_{}_{}_AVERAGES.csv'.format(stype, sid, tperiod, ttype)
				tots_res.to_csv(os.path.join(outdir, out_dsn), index = False, encoding = 'utf-8')

			# Load data to Google BigQuery
			projectid = 'cr2c-monitoring'
			dataset_id = 'opdata'
			# Make sure only new records are being appended to the dataset
			tots_res_already = \
				get_data(
					[stype],
					[sid],
					[tperiod],
					[ttype], 
					combine_all = False
				)['{}_{}_{}_{}_AVERAGES'.format(stype, sid, tperiod, ttype)]

			tots_res_new = tots_res.loc[~tots_res['Time'].isin(tots_res_already['Time']),:]
			# Remove duplicates and missing values
			tots_res_new.dropna(inplace = True)
			tots_res_new.drop_duplicates(inplace = True)
			# Write to gbq table
			if not tots_res_new.empty:
				tots_res_new.to_gbq(
					'{}.{}_{}_{}_{}_AVERAGES'.format(dataset_id, stype, sid, tperiod, ttype), 
					projectid, 
					if_exists = 'append'
				)
