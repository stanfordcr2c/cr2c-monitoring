''' This script calculates totals and averages for any given HMI data point(s), 
	time period, and date range. If selected, it outputs a csv file with average/total 
	data for the data points and time periods chosen
'''
import numpy as np
import scipy as sp
from scipy import interpolate as ip
import pandas as pd
import datetime as datetime
from datetime import datetime as dt
from datetime import timedelta as tdelt
from pandas import read_excel
import os
import sys


class hmi_data_agg:

	def __init__(
		self, 
		data_indir,
		report_oudir, 
		in_filename, 
		op_filename,
		qtype, 
		start_dt, 
		end_dt, 
		tperiod, 
		elids,
		agg_types,
		lo_limits,
		hi_limits
	):

		self.data_indir = data_indir
		self.report_oudir = report_oudir
		self.in_filename = in_filename
		self.op_filename = op_filename
		self.qtype = qtype.upper()
		self.start_dt = dt.strptime(start_dt,'%m-%d-%y')
		self.end_dt = dt.strptime(end_dt,'%m-%d-%y')
		self.tperiod = tperiod
		self.elids = elids
		self.agg_types = agg_types
		self.lo_limits = lo_limits
		self.hi_limits = hi_limits

	def prep_data(self, elid, lo_limit, hi_limit):

		# Load data
		os.chdir(self.data_indir)
		self.hmi_data = pd.read_csv(self.in_filename)

		# Load variables and set output variable names
		varname = 'CR2C.CODIGA.{0}.SCALEDVALUE {1} [{2}]'

		# Rename variable		
		self.hmi_data[elid + '_value'] = \
			self.hmi_data[varname.format(elid,'Value', self.qtype)]
		# Set low/negative values to 0 and remove unreasonably high values
		self.hmi_data.loc[:, elid + '_value'][self.hmi_data[elid + '_value'] < lo_limit] = 0
		self.hmi_data.loc[:, elid + '_value'][self.hmi_data[elid + '_value'] > hi_limit] = np.NaN			
		# Rename and format corresponding timestamp variable 
		self.hmi_data[elid + '_ts'] = \
			self.hmi_data[varname.format(elid, 'Time', self.qtype)]
		self.hmi_data[elid + '_ts'] = \
			pd.to_datetime(self.hmi_data[elid + '_ts'])

		# Filter dataset to clean values and variable selected
		self.hmi_data = self.hmi_data.loc[:, [elid + '_value', elid + '_ts']]
		self.hmi_data.dropna(axis = 0, how = 'any', inplace = True)


	def get_tot_var(self, elid, agg_type):

		xvar = elid + '_ts'
		yvar = elid + '_value'

		# Get numeric time elapsed
		first_ts = self.hmi_data[xvar][0]
		last_ts = self.hmi_data[xvar][len(self.hmi_data) - 1]

		# Check to make sure that the totals/averages do not include the first
		# and last days for which data are available (just to ensure accuracy)
		if first_ts.day >= self.start_dt.day or last_ts.day <= self.end_dt.day:
			start_dt_warn = first_ts + np.timedelta64(1,'D')
			end_dt_warn = last_ts + np.timedelta64(1,'D')
			start_dt_warn = dt.strftime(start_dt_warn, '%m-%d-%y')
			end_dt_warn = dt.strftime(end_dt_warn, '%m-%d-%y')
			warn_msg = \
				'Given the range of data available for {0}, accurate aggregate values can only be obtained for: {1} to {2}'
			print(warn_msg.format(elid, start_dt_warn, end_dt_warn))
			sys.exit()

		# Calculating time elapsed in minutes (since highest resolution is ~30s)
		self.hmi_data['tel'] = self.hmi_data[xvar] - first_ts
		self.hmi_data['tel'] = self.hmi_data['tel'] / np.timedelta64(60,'s')

		# Creat variables for manually calculating area under curve
		tel_next  = np.append(self.hmi_data.loc[1:,'tel'].values, [0]) 
		yvar_next = np.append(self.hmi_data.loc[1:,yvar].values, [0])
		self.hmi_data['tel_next'] = tel_next
		self.hmi_data['yvar_next'] = yvar_next
		self.hmi_data['tot'] = \
			(self.hmi_data['tel_next'] - self.hmi_data['tel'])*\
			(self.hmi_data['yvar_next'] + self.hmi_data[yvar])/2

		# Compute the area under the curve for each time period
		nperiods = (self.end_dt.day - self.start_dt.day)*24/self.tperiod
		nperiods = int(nperiods)
		tots_res = []
		for period in range(nperiods):
			start_tel = (self.start_dt - first_ts) / np.timedelta64(1,'m') + period*60*self.tperiod
			end_tel = start_tel + 60*24
			start_ts = self.start_dt + datetime.timedelta(hours = period*self.tperiod)
			ip_tot = self.hmi_data.loc[
				(self.hmi_data['tel'] >= start_tel) & 
				(self.hmi_data['tel'] <= end_tel),'tot'
			].sum()
			if agg_type == 'average':
				ip_tot = ip_tot/(60*self.tperiod)
			tots_row = [start_ts, ip_tot.tolist()]
			tots_res.append(tots_row)
		
		return tots_res


	def run_report(self):

		for elid in self.elids:
			elid_no = self.elids.index(elid)
			lo_limit = self.lo_limits[elid_no]
			hi_limit = self.hi_limits[elid_no]
			agg_type = self.agg_types[elid_no]
			# Get prepped data
			self.prep_data(elid, lo_limit, hi_limit)
			# Get totalized values'
			report_dat = self.get_tot_var(elid, agg_type)
			if elid_no == 0:
				res_df = pd.DataFrame([row[0] for row in report_dat], columns = ['Time'])
			# Skip time variable for all other elements we are getting data for
			res_df[elid + '_' + agg_type] = [row[1] for row in report_dat]

		# Output to directory given
		os.chdir(self.report_oudir)
		res_df.to_csv(self.op_filename, index = False, encoding = 'utf-8')

if __name__ == '__main__':
	hmi_dat = hmi_data_agg(
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data',
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data',
		'Reactor Feeding - Raw_20170721102913.csv',
		'test_report.csv',
		'raw',
		'7-11-17',
		'7-20-17',
		24,
		['FT202','FT305','AT305','AT311'],
		['total','total','average','average'],
		[0, 0, 0, 0],
		[30, 30, 14, 14]
	)
	hmi_dat.run_report()