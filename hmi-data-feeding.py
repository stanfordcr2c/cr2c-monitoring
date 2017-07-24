''' This script takes HMI data from flow meter FT-202 (into the reactors from the pre-treatment area), 
	FT-304 (permeate flow out of the Research AFMBR), FT-305 (permeate flow out of the Duty AFMBR),
	and analyzer elements AIT-305, AIT-308, AIT-311 (measuring pH in the AFBR, Research AFMBR and Duty
	AFMBR respectively), and computes totalizations and time averages for the parameters given by the user
'''
import numpy as np
import scipy as sp
from scipy import interpolate as ip
import pandas as pd
import datetime as datetime
from datetime import datetime as dt
from datetime import timedelta as tdelt
from pandas import read_excel
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import warnings
import os
import sys

class hmi_data_feeding:

	def __init__(self, data_indir, filename, qtype, fr_tp):
		self.data_indir = data_indir
		self.filename = filename
		self.qtype = qtype.upper()
		self.fr_tp = fr_tp

	def get_data(self):
		os.chdir(self.data_indir)
		self.hmi_data = pd.read_csv(self.filename)

	def interp_frate(self, xvar, yvar, nhours, interp_type, spline_type):

		# Set flowrate thresholds
		fr_limit_lo = 0.5
		fr_limit_hi = 30

		# Create data frame for interpolation variables, clean and remove missing values
		interp_data = self.hmi_data.loc[:,[xvar, yvar]]
		interp_data.loc[:,yvar][interp_data[yvar] < fr_limit_lo] = 0
		interp_data.loc[:,yvar][interp_data[yvar] > fr_limit_hi] = np.NaN
		interp_data.dropna(axis = 0, how = 'any', inplace = True)

		# Get numeric time elapsed
		first_ts = interp_data[xvar][0]
		# Calculating time elapsed in minutes (since highest resolution is ~30s)
		interp_data['tel'] = interp_data[xvar] - first_ts
		interp_data['tel'] = interp_data['tel'] / np.timedelta64(60,'s')

		# Run interpolation
		if interp_type == 'spline' and spline_type == 'cubic':
			ip_func = ip.CubicSpline(
				np.array(interp_data['tel'].values), 
				np.array(interp_data[yvar].values),
				axis = 0, 
				bc_type = 'natural'
			)
		if interp_type == 'linear':
			# Creat variables for manually calculating area under curve
			tel_next  = np.append(interp_data.loc[1:,'tel'].values, [0]) 
			yvar_next = np.append(interp_data.loc[1:,yvar].values, [0])
			interp_data['tel_next'] = tel_next
			interp_data['yvar_next'] = yvar_next
			interp_data['vol'] = \
			(interp_data['tel_next'] - interp_data['tel'])*\
			(interp_data['yvar_next'] + interp_data[yvar])/2
			# Get linear interpolation function (for plotting)
			ip_func = ip.interp1d(
				interp_data['tel'].values, 
				interp_data[yvar].values,
				kind = 'linear'
			)

		# Plot
		zoom_d_st = 5
		zoom_h_st = 12
		zoom_h_end = 13
		zoom_ts_st = zoom_d_st*24*60 + zoom_h_st*60
		zoom_ts_end = zoom_d_st*24*60 + zoom_h_end*60
		x = interp_data['tel'].values
		y = interp_data[yvar].values
		y = y[(x > zoom_ts_st) & (x < zoom_ts_end)]
		x = x[(x > zoom_ts_st) & (x < zoom_ts_end)]
		xnew = np.arange(zoom_ts_st, zoom_ts_end, 0.1)
		time = pd.to_timedelta(x,'m') + first_ts
		time_new = pd.to_timedelta(xnew, 'm') + first_ts
		ynew = ip_func(xnew)
		hfmt = dates.DateFormatter('%m/%d %H:%M')
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(time, y,'o', time_new, ynew,'-')
		ax.xaxis.set_major_locator(dates.MinuteLocator([0,10,20,30,40,50]))
		ax.xaxis.set_major_formatter(hfmt)
		plt.xticks(rotation='vertical')
		plt.subplots_adjust(bottom=.3)
		plt.title('Interpolation:' + interp_type)
		plt.xlabel('Time') 	 	 	
		plt.ylabel('Flowrate (gpm)')
		plt.show()

		# Compute the area under the curve for each time period
		nperiods = (self.end_ts_fr.day - self.start_ts_fr.day)*24/nhours
		nperiods = int(nperiods)
		vols_res = []
		for period in range(nperiods):
			start_tel = (self.start_ts_fr - first_ts) / np.timedelta64(1,'m') + period*60*nhours
			end_tel = start_tel + 60*24
			start_ts = self.start_ts_fr + datetime.timedelta(hours = period*nhours)
			if interp_type == 'spline':
				ip_vol = ip.PPoly.integrate(ip_func, start_tel, end_tel)
			if interp_type == 'linear':
				ip_vol = interp_data.loc[
					(interp_data['tel'] >= start_tel) & 
					(interp_data['tel'] <= end_tel),'vol'
				].sum()
			vols_row = [start_ts, ip_vol.tolist()]
			vols_res.append(vols_row)
		
		return vols_res
			

	def get_tot_flow(
		self, 
		start_dt, 
		end_dt, 
		nhours, 
		el_id, 
		verify, 
		el_id_ver, 
		interp_type, 
		spline_type
	):

		self.start_ts_fr = dt.strptime(start_dt, '%m-%d-%y')
		self.end_ts_fr   = dt.strptime(end_dt, '%m-%d-%y')
		
		# Load Data
		self.get_data
		# Load variables
		varname = 'CR2C.CODIGA.{0}.SCALEDVALUE {1} [{2}]'
		self.hmi_data['frate'] = \
			self.hmi_data[varname.format(el_id,'Value', self.qtype)]
		self.hmi_data['tstamp'] = \
			self.hmi_data[varname.format(el_id, 'Time', self.qtype)]
		self.hmi_data['tstamp'] = pd.to_datetime(self.hmi_data['tstamp'])
		self.hmi_data['frate_ver'] = \
			self.hmi_data[varname.format(el_id_ver,'Value', self.qtype)]
		self.hmi_data['tstamp_ver'] = \
			self.hmi_data[varname.format(el_id_ver,'Time', self.qtype)]
		self.hmi_data['tstamp_ver'] = pd.to_datetime(self.hmi_data['tstamp_ver'])

		# Get totalized flowrate
		vols = self.interp_frate('tstamp','frate', 24, interp_type, spline_type)
		if verify == 1:
			vols_ver = self.interp_frate('tstamp_ver','frate_ver', 24, interp_type, spline_type)

		op_varname = '{0} Vol. (gallons, {1})'
		res_df = pd.DataFrame(vols)
		res_df.columns = ['Time',op_varname.format('AFBR Inf.', el_id)]
		res_df[op_varname.format('AFMBR. Perm.', el_id_ver)] = [row[1] for row in vols_ver]

		# Output report dataset
		data_filename = 'Feeding Volumes {0} to {1} {2}.csv'
		filename = data_filename.format(start_dt, end_dt, interp_type)
		res_df.to_csv(filename, index = False, encoding = 'utf-8')

if __name__ == '__main__':
	hmi_dat = hmi_data_feeding(
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data',
		'Reactor Feeding - Raw_20170721102913.csv',
		'raw',
		'daily'
	)
	hmi_dat.get_data()
	hmi_dat.get_tot_flow(
		'7-11-17',
		'7-20-17',
		24,
		'FT202',
		1,
		'FT305',
		'spline',
		'cubic'
	)
