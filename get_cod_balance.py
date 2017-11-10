'''
	Computes a mass balance for COD-CH4 in the reactor area for any range of dates
	takes dates as inputs and outputs a summary file with mass balance info
'''

from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import pylab as pl
import numpy as np
import scipy as sp
from scipy import interpolate as ip
import pandas as pd
import datetime as datetime
from datetime import datetime as dt
from pandas import read_excel
import os
import sys
import functools
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory
import get_lab_data as gld
import hmi_data_agg as hmi
from hmi_data_agg import hmi_data_agg as hmi_run
import est_biogas_prod as ebg

def get_cod_bal(
	start_dt_str,
	end_dt_str,
	tperiod,
	ttype,
	outdir = None,
	run_agg_feeding = False,
	run_agg_gasprod = False,
	run_agg_temp = False
):

	start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
	end_dt   = dt.strptime(end_dt_str,'%m-%d-%y')

	ttype = ttype.upper()
	if ttype == 'MINUTE':
		tperiod_hrs = tperiod/60
	else:
		tperiod_hrs = tperiod

	if not outdir:
		outdir = askdirectory(title = 'Directory to output summary statistics/plots to')

	gas_elids  = ['FT700','FT704']
	temp_elids = ['AT304','AT310']
	inf_elid   = 'FT202'
	eff_elid   = 'FT305'

	# Reactor volumes
	afbr_vol = 1100 # in L
	afmbr_vol = 1700 # in L
	l_p_gal = 3.78541 # Liters/Gallon

	# If requested, run the hmi_data_agg script for the reactor meters and time period of interest
	get_hmi = hmi_run(start_dt_str, end_dt_str)
	if run_agg_feeding:
		get_hmi.run_report(
			[tperiod]*2, # Number of hours you want to average over
			[ttype]*2, # Type of time period (can be "hour" or "minute")
			[inf_elid, eff_elid], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
			['water']*2, # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
		)	
	if run_agg_gasprod:
		get_hmi.run_report(
			[tperiod]*len(gas_elids), # Number of hours you want to average over
			[ttype]*len(gas_elids), # Type of time period (can be "hour" or "minute")
			gas_elids, # Sensor ids that you want summary data for (have to be in HMI data file obviously)
			['gas']*len(gas_elids), # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
		)
	if run_agg_temp:
		get_hmi.run_report(
			[tperiod]*len(temp_elids), # Number of hours you want to average over
			[ttype]*len(temp_elids), # Type of time period (can be "hour" or "minute")
			temp_elids, # Sensor ids that you want summary data for (have to be in HMI data file obviously)
			['temp']*len(temp_elids), # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
		)

	# Read in the data (data requested in lines above is now in sql file and can be queried)
	year = start_dt.year
	gasprod_all = hmi.get_data(
		gas_elids,
		[tperiod]*len(gas_elids),
		[ttype]*len(gas_elids), 
		year = year, 
		start_dt_str = start_dt_str, 
		end_dt_str = end_dt_str
	)
	gasprod_dat = gasprod_all['{0}_{1}{2}_AVERAGES'.format(gas_elids[0],tperiod,ttype)]
	# Rename the "Value" variables to the relevant element ids (elids)
	for elid in gas_elids:
		gasprod_dat[elid] = gasprod_all['{0}_{1}{2}_AVERAGES'.format(elid,tperiod,ttype)]['Value']

	# Do the same for feeding and temperature
	feeding_all = hmi.get_data(
		[inf_elid, eff_elid],
		[tperiod]*2, 
		[ttype]*2, 
		year = year,
		start_dt_str = start_dt_str,
		end_dt_str = end_dt_str
	)
	feeding_dat = feeding_all['{0}_{1}{2}_AVERAGES'.format(inf_elid,tperiod,ttype)]
	feeding_dat[inf_elid] = feeding_dat['Value']*l_p_gal
	feeding_dat[eff_elid] = feeding_all['{0}_{1}{2}_AVERAGES'.format(eff_elid,tperiod,ttype)]['Value']*l_p_gal
	temp_all = hmi.get_data(
		temp_elids,
		[tperiod]*len(temp_elids), 
		[ttype]*len(temp_elids), 
		year = year,
		start_dt_str = start_dt_str,
		end_dt_str = end_dt_str
	)
	temp_dat = temp_all['{0}_{1}{2}_AVERAGES'.format(temp_elids[0],tperiod,ttype)]
	for elid in temp_elids:
		temp_dat[elid] = temp_all['{0}_{1}{2}_AVERAGES'.format(elid,tperiod,ttype)]['Value'] 

	# Get lab data from file on box and filter to desired dates
	labdat  = gld.get_data(['COD','GasComp'])
	
	# COD data
	cod_dat = labdat['COD']
	cod_dat['Date_Time'] = pd.to_datetime(cod_dat['Date_Time'])
	cod_dat = cod_dat.loc[
		(cod_dat['Date_Time'] >= start_dt) &
		(cod_dat['Date_Time'] <= end_dt)
	]
	# Drop duplicates
	cod_dat.drop_duplicates(keep = 'first', inplace = True)
	# Get average of multiple values taken on same day
	cod_dat = cod_dat.groupby(['Date_Time','Stage','Type']).mean()
	# Convert to wide to get COD in and out of the reactors
	cod_dat_wide = cod_dat.unstack(['Stage','Type'])
	cod_dat_wide['CODs MS']  = cod_dat_wide['Value']['Microscreen']['Soluble']
	cod_dat_wide['CODp MS']  = cod_dat_wide['Value']['Microscreen']['Particulate']
	# Weighted aveage COD concentrations in the reactors
	cod_dat_wide['CODs R'] = \
		(cod_dat_wide['Value']['AFBR']['Soluble']*afbr_vol +\
		cod_dat_wide['Value']['Duty AFMBR MLSS']['Soluble']*afmbr_vol)/\
		(afbr_vol + afmbr_vol)
	cod_dat_wide['CODp R'] = \
		(cod_dat_wide['Value']['AFBR']['Particulate']*afbr_vol +\
		cod_dat_wide['Value']['Duty AFMBR MLSS']['Particulate']*afmbr_vol)/\
		(afbr_vol + afmbr_vol)
	cod_dat_wide['CODs Out'] = cod_dat_wide['Value']['Duty AFMBR Effluent']['Soluble']
	cod_dat_wide.reset_index(inplace = True)

	if ttype == 'HOUR':
		cod_dat_wide['Time'] = cod_dat_wide['Date_Time'].values.astype('datetime64[h]')
	else:
		cod_dat_wide['Time'] = cod_dat_wide['Date_Time'].values.astype('datetime64[m]')

	cod_dat_cln = cod_dat_wide[['Time','CODs MS','CODp MS','CODs R','CODp R','CODs Out']]
	cod_dat_cln.columns = ['Time','CODs MS','CODp MS','CODs R','CODp R','CODs Out']

	# Gas Composition Data
	gc_dat = labdat['GasComp']
	gc_dat['Date_Time'] = pd.to_datetime(gc_dat['Date_Time'])
	gc_dat = gc_dat.loc[
		(gc_dat['Date_Time'] >= start_dt) &
		(gc_dat['Date_Time'] <= end_dt) &
		(gc_dat['Type'].isin(['Methane (%)','Carbon Dioxide (%)']))
	]
	gc_dat = gc_dat.groupby(['Date_Time','Type']).mean()
	gc_dat_wide = gc_dat.unstack('Type')
	gc_dat_wide['CH4%'] = gc_dat_wide['Value']['Methane (%)']
	gc_dat_wide['CO2%'] = gc_dat_wide['Value']['Carbon Dioxide (%)']
	gc_dat_wide.reset_index(inplace = True)
	if ttype == 'HOUR':
		gc_dat_wide['Time'] = gc_dat_wide['Date_Time'].values.astype('datetime64[h]')
	else:
		gc_dat_wide['Time'] = gc_dat_wide['Date_Time'].values.astype('datetime64[m]')
	gc_dat_cln = gc_dat_wide[['Time','CH4%','CO2%']]
	gc_dat_cln.columns = ['Time','CH4%','CO2%']

	# Prep the HMI data
	# NOTE 1: # Getting totals as the average flow (in liters or gallons per minute) x 
	# 60 x 
	# time period in hour equivalents x
	# size of the time period
	# NOTE 2: for now, getting the date of the time step to merge onto daily data
	# In the future we could linearly interpolate between two different values on two days...
	gasprod_dat['Time']       = pd.to_datetime(gasprod_dat['Time'])
	gasprod_dat['Meas Biogas Prod'] = (gasprod_dat['FT700'] + gasprod_dat['FT704'])*60*tperiod_hrs*tperiod
	gasprod_dat_cln           = gasprod_dat[['Time','Meas Biogas Prod']]

	# Feeding HMI Data
	feeding_dat['Time']     = pd.to_datetime(feeding_dat['Time'])
	feeding_dat['Flow In']  = feeding_dat[inf_elid]*60*tperiod_hrs*tperiod
	feeding_dat['Flow Out'] = feeding_dat[eff_elid]*60*tperiod_hrs*tperiod
	feeding_dat_cln         = feeding_dat[['Time','Flow In','Flow Out']]

	# Reactor Temperature HMI data
	temp_dat['Time']         = pd.to_datetime(temp_dat['Time'])
	temp_dat['Reactor Temp'] = temp_dat[temp_elids].mean(axis = 1)
	temp_dat_cln             = temp_dat[['Time','Reactor Temp']]

	# List of hmi dataframes
	hmi_dflist = [temp_dat_cln, feeding_dat_cln, gasprod_dat_cln]
	# Merge hmi datasets
	hmidat = functools.reduce(lambda left,right: pd.merge(left,right, on='Time', how = 'outer'), hmi_dflist)
	hmidat['Date'] = hmidat['Time'].dt.date
	hmidat.rename(columns = {'Time': 'HMITime'}, inplace = True)
	# Merge lab data by time
	labdat = cod_dat_cln.merge(gc_dat_cln, on = 'Time', how = 'outer')
	labdat['Date'] = labdat['Time'].dt.date
	labdat.rename(columns = {'Time': 'LabTime'}, inplace = True)
	# Merge all data
	cod_bal_dat = labdat.merge(hmidat, on = 'Date', how = 'outer')
	# Dedupe (merging many files, so any duplicates will cause big problems!)
	cod_bal_dat.drop_duplicates(inplace = True)

	# Offset time (after sampling each day, so it is associated wih the next sampling day) 
	cod_bal_dat.loc[cod_bal_dat['HMITime'] > cod_bal_dat['LabTime'],'Date'] = cod_bal_dat['Date'] + datetime.timedelta(days = 1)


	dly_tots  = cod_bal_dat[['Date','Flow In','Flow Out','Meas Biogas Prod']].groupby('Date').sum()
	dly_tots.reset_index(inplace = True)
	dly_means = cod_bal_dat[['Date','LabTime','Reactor Temp','CODs MS','CODp MS','CODs R','CODp R','CODs Out','CH4%','CO2%']].groupby('Date').mean()
	dly_means.reset_index(inplace = True)

	cod_bal_dly = dly_tots.merge(dly_means, on = 'Date', how = 'outer')
	cod_bal_dly.set_index('Date')
	cod_bal_dly[['CH4%','CO2%','CODs MS','CODp MS','CODs R','CODp R','CODs Out']] = \
		cod_bal_dly[['CH4%','CO2%','CODs MS','CODp MS','CODs R','CODp R','CODs Out']].interpolate()
	cod_bal_dly.reset_index(inplace = True)

	# ============> Estimating COD Consumption <============

	# First estimate particulate COD hydrolized by comparing the particulate COD
	# that should accumulate in the reactor from influent particulate COD vs actual particulate COD
	cod_bal_dly['CODp R pot'] = \
		(
			# Mass that was in the reactors in the prior timestep
			cod_bal_dly['CODp R'].shift(1)*(afbr_vol + afmbr_vol) +
			# Mass that was added by influent particulate COD
			cod_bal_dly['CODp MS'].shift(1)*cod_bal_dly['Flow In'].shift(1)
		)/\
		(afbr_vol + afmbr_vol)
	# The hydrolized COD is the difference between the accumulated vs observed particulate COD
	cod_bal_dly['CODp R hyd'] = cod_bal_dly['CODp R pot'] - cod_bal_dly['CODp R']

	# Next compute the soluble COD that would accumulate without consumption by the biology
	cod_bal_dly['CODs R pot'] = \
		(
			# Mass that was in the reactors in the prior timestep
			cod_bal_dly['CODs R'].shift(1)*(afbr_vol + afmbr_vol) +
			# Mass that flowed in from the microscreen
			cod_bal_dly['CODs MS'].shift(1)*cod_bal_dly['Flow In'].shift(1) + 
			# Mass that hydrolyzed
			cod_bal_dly['CODp R hyd'] - 
			# Mass that flowed out through the membranes
			cod_bal_dly['CODs Out']*cod_bal_dly['Flow Out'].shift(1)
		)/\
		(afbr_vol + afmbr_vol)
	# Consumed COD is the difference between the accumulated vs observed soluble COD
	cod_bal_dly['COD Consumed'] = cod_bal_dly['CODs R pot'] - cod_bal_dly['CODs R']

	# ============> Estimating COD Consumption <============

	# # Get theoretical estimated methane output
	gasprod_thry = []
	for index,row in cod_bal_dly.iterrows():
		gasprod_thry.append(
			ebg.get_biogas_prod(
				BODrem = row['COD Consumed'], 
				infSO4 = 4, 
				temp = row['Reactor Temp'], 
				percCH4 = row['CH4%']/100, 
				percCO2 = row['CO2%']/100, 
				flowrate = row['Flow In']/1000, 
				precision = 1E-6
			)
		)

	cod_bal_dly['Thr CH4 Prod'] = [row[0] for row in gasprod_thry]
	cod_bal_dly['Thr Biogas Prod'] = [row[1] for row in gasprod_thry]
	# Actual estimated CH4 production
	cod_bal_dly['Meas CH4 Prod'] = cod_bal_dly['Meas Biogas Prod']*cod_bal_dly['CH4%']/100
	cod_bal_dly['Biogas Discrep (%)'] =	(cod_bal_dly['Meas Biogas Prod']/cod_bal_dly['Thr Biogas Prod'] - 1)*100
	cod_bal_dly.loc[np.isnan(cod_bal_dly['Biogas Discrep (%)']),'Biogas Discrep (%)'] = 100
	cod_bal_dly.loc[cod_bal_dly['Biogas Discrep (%)'] > 100,'Biogas Discrep (%)'] = 100
	cod_bal_dly['CH4 Discrep (%)']    =	(cod_bal_dly['Meas CH4 Prod']/cod_bal_dly['Thr CH4 Prod'] - 1)*100
	cod_bal_dly.loc[np.isnan(cod_bal_dly['CH4 Discrep (%)']),'CH4 Discrep (%)'] = 100
	cod_bal_dly.loc[cod_bal_dly['CH4 Discrep (%)'] > 100,'CH4 Discrep (%)'] = 100

	# Output csv with summary statistics
	os.chdir(outdir)
	output_vars = ['Date','Meas Biogas Prod','Thr Biogas Prod','Meas CH4 Prod','Thr CH4 Prod','Biogas Discrep (%)','CH4 Discrep (%)']
	cod_bal_dly[output_vars].to_csv('COD Balance.csv')
	days_el = (cod_bal_dly['Date'] - cod_bal_dly['Date'][0])/np.timedelta64(24,'h')

	cod_bal_dly.rename(
		columns = {
			'Thr Biogas Prod': 'Theoretical',
			'Meas Biogas Prod': 'Measured'
		},
		inplace = True
	)

	fig, (ax1, ax2) = plt.subplots(2, sharex = True)
	ax1.plot(cod_bal_dly['Date'], cod_bal_dly['Theoretical'])
	ax1.plot(cod_bal_dly['Date'], cod_bal_dly['Measured'])
	ax1.set_ylabel('Production (L/day)')
	ax2.plot(cod_bal_dly['Date'], cod_bal_dly['Biogas Discrep (%)'])
	ax2.set_ylabel('Discrepancy (%)')
	ax1.legend()
	ax2.axhline(linewidth = 0.5, color = 'black', linestyle = '--')
	labels = ax2.get_xticklabels()
	plt.setp(labels, rotation = 45)
	plt.tight_layout()

	plt.savefig(
		'COD Balance.png', 
		bbox_inches = 'tight'
	)



get_cod_bal(
	'8-12-17',
	'11-1-17',
	1,
	'hour',
	# run_agg_temp = True,
	outdir = '/Users/josebolorinos/Google Drive/Codiga Center/Miscellany',

	
)


