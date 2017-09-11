''' This script calculates totals and averages for any given HMI data point(s), 
	time period, and date range for which a raw eDNA query has been run (and a csv file
	for that query obtained)
	If selected, it outputs a csv file with average/total 
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
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename

class hmi_data_agg:

	def __init__(
		self, 
		qtype, 
		stype,
		start_dt, 
		end_dt, 
		tperiod, 
		elids,
		agg_types
	):

		self.qtype = qtype.upper()
		self.stype = stype.upper()
		self.start_dt = dt.strptime(start_dt,'%m-%d-%y')
		self.end_dt = dt.strptime(end_dt,'%m-%d-%y')
		self.tperiod = tperiod
		self.elids = elids
		self.agg_types = [agg_type.upper() for agg_type in agg_types]

	def prep_data(self, elid):

		# Set high and low limits for sensors based on type (water, gas, ph, conductivity, temp)
		if self.stype == 'WATER':
			hi_limit = 30
			lo_limit = 0.5
		elif self.stype == 'GAS':
			hi_limit = 10
			lo_limit = -10
		elif self.stype == 'PH':
			hi_limit = 10
			lo_limit = 4

		# Load data
		try:
			self.hmi_data = pd.read_csv(self.in_path)
		except FileNotFoundError:
			print('Please choose an existing input file with the HMI data')
			sys.exit()

		# Load variables and set output variable names
		varname = 'CR2C.CODIGA.{0}.SCALEDVALUE {1} [{2}]'

		# Rename variable		
		self.hmi_data[elid + '_value'] = \
			self.hmi_data[varname.format(elid,'Value', self.qtype)]
		# Set low/negative values to 0 and remove unreasonably high values
		self.hmi_data.loc[self.hmi_data[elid + '_value'] < lo_limit, elid + '_value'] = 0
		self.hmi_data.loc[self.hmi_data[elid + '_value'] > hi_limit, elid + '_value'] = np.NaN			
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
		if first_ts >= self.start_dt or last_ts <= self.end_dt:
			start_dt_warn = first_ts + np.timedelta64(1,'D')
			end_dt_warn = last_ts - np.timedelta64(1,'D')
			start_dt_warn = dt.strftime(start_dt_warn, '%m-%d-%y')
			end_dt_warn = dt.strftime(end_dt_warn, '%m-%d-%y')
			warn_msg = \
				'Given the range of data available for {0}, accurate aggregate values can only be obtained for: {1} to {2}'
			print(warn_msg.format(elid, start_dt_warn, end_dt_warn))

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
		nperiods = (self.end_dt - self.start_dt).days*24/self.tperiod
		nperiods = int(nperiods)
		tots_res = []
		for period in range(nperiods):
			start_tel = (self.start_dt - first_ts) / np.timedelta64(1,'m') + period*60*self.tperiod
			end_tel = start_tel + 60*self.tperiod
			start_ts = self.start_dt + datetime.timedelta(hours = period*self.tperiod)
			ip_tot = self.hmi_data.loc[
				(self.hmi_data['tel'] >= start_tel) & 
				(self.hmi_data['tel'] <= end_tel),'tot'
			].sum()
			if agg_type == 'AVERAGE':
				ip_tot = ip_tot/(60*self.tperiod)
			tots_row = [start_ts, ip_tot]
			tots_res.append(tots_row)
		return tots_res


	def run_report(self):

		# Select input data file
		self.in_path = askopenfilename(title = 'Select data input file')

		for elid in self.elids:
			elid_no = self.elids.index(elid)
			agg_type = self.agg_types[elid_no]
			# Get prepped data
			self.prep_data(elid)
			# Get totalized values'
			report_dat = self.get_tot_var(elid, agg_type)
			if elid_no == 0:
				res_df = pd.DataFrame([row[0] for row in report_dat], columns = ['Time'])
			agg_type = self.agg_types[elid_no]
			# Skip time variable for all other elements we are getting data for
			res_df[elid + '_' + agg_type] = [row[1] for row in report_dat]

		# Output to directory given
		op_path = asksaveasfilename(title = 'Save Output File as:')
		res_df.to_csv(op_path, index = False, encoding = 'utf-8')

if __name__ == '__main__':
	hmi_dat = hmi_data_agg(
		'raw', # Type of eDNA query (case insensitive, can be raw, 1 min, 1 hour)
		'water', # Type of sensor (case insensitive, can be water, gas, pH, conductivity or temperature)
		'5-11-17', # Start of date range you want summary data for
		'9-8-17', # End if date range you want summary data for
		1, # Number of hours you want to sum/average over
		['FT202','FT305'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
		['total','total'], # Type of aggregate function you want (can be total or average)
	)
	hmi_dat.run_report()
