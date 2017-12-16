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
from datetime import timedelta
from datetime import datetime as dt
from pandas import read_excel
import os
import sys
import functools
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory
import process_lab_data as pld
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
		tkTitle = 'Directory to output summary statistics/plots to'
		print(tkTitle)
		outdir = askdirectory(title = tkTitle)

	gas_elids  = ['FT700','FT704']
	temp_elids = ['AT304','AT310']
	inf_elid   = 'FT202'
	eff_elid   = 'FT305'

	# Reactor volumes
	afbr_vol = 1100 # in L
	afmbr_vol = 1700 # in L
	l_p_gal = 3.78541 # Liters/Gallon


	#=========================================> HMI DATA <=========================================
	
	# If requested, run the hmi_data_agg script for the reactor meters and time period of interest
	if run_agg_feeding or run_agg_gasprod or run_agg_temp:
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

	# Read in the data
	year = start_dt.year

	gasprod_dat = hmi.get_data(
		gas_elids,
		[tperiod]*len(gas_elids),
		[ttype]*len(gas_elids), 
		year = year, 
		start_dt_str = start_dt_str, 
		end_dt_str = end_dt_str
	)
	# Do the same for feeding and temperature
	feeding_dat = hmi.get_data(
		[inf_elid, eff_elid],
		[tperiod]*2, 
		[ttype]*2, 
		year = year,
		start_dt_str = start_dt_str,
		end_dt_str = end_dt_str
	)
	temp_dat = hmi.get_data(
		temp_elids,
		[tperiod]*len(temp_elids), 
		[ttype]*len(temp_elids), 
		year = year,
		start_dt_str = start_dt_str,
		end_dt_str = end_dt_str
	) 

	# Prep the HMI data
	# NOTE 1: # Getting totals as the average flow (in liters or gallons per minute) x 
	# 60 x 
	# time period in hour equivalents x
	# size of the time period
	# NOTE 2: for now, getting the date of the time step to merge onto daily data
	# In the future we could linearly interpolate between two different values on two days...
	gasprod_dat['Meas Biogas Prod'] = (gasprod_dat['FT700'] + gasprod_dat['FT704'])*60*tperiod_hrs*tperiod
	gasprod_dat_cln                 = gasprod_dat[['Time','Meas Biogas Prod']]

	# Feeding HMI Data
	feeding_dat['Flow In']  = feeding_dat[inf_elid]*60*tperiod_hrs*tperiod*l_p_gal
	feeding_dat['Flow Out'] = feeding_dat[eff_elid]*60*tperiod_hrs*tperiod*l_p_gal
	feeding_dat_cln         = feeding_dat[['Time','Flow In','Flow Out']]

	# Reactor Temperature HMI data
	temp_dat['Reactor Temp'] = temp_dat[temp_elids].mean(axis = 1)
	temp_dat_cln             = temp_dat[['Time','Reactor Temp']]

	# List of hmi dataframes
	hmi_dflist = [temp_dat_cln, feeding_dat_cln, gasprod_dat_cln]
	# Merge hmi datasets
	hmidat = functools.reduce(lambda left,right: pd.merge(left,right, on='Time', how = 'outer'), hmi_dflist)
	hmidat['Date'] = hmidat['Time'].dt.date
	
	#=========================================> HMI DATA <=========================================

	#=========================================> LAB DATA <=========================================
	
	# Get lab data from file on box and filter to desired dates
	labdat  = pld.labrun().get_data(['COD','GasComp'])
	
	# COD data
	cod_dat = labdat['COD']
	cod_dat['Date_Time'] = pd.to_datetime(cod_dat['Date_Time'])
	cod_dat = cod_dat.loc[
		(cod_dat['Date_Time'] >= start_dt) &
		(cod_dat['Date_Time'] <= end_dt + timedelta(days = 1))
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

	# Merge lab data by time
	labdat = cod_dat_cln.merge(gc_dat_cln, on = 'Time', how = 'outer')
	labdat['Date'] = labdat['Time'].dt.date
	# Get daily average of readings if multiple readings in a day (also prevents merging issues!)
	labdat_ud = labdat.groupby('Date').mean()
	labdat_ud.reset_index(inplace = True)

	#=========================================> LAB DATA <=========================================

	#=======================================> MERGE & PREP <=======================================	
	
	# Merge Lab and HMI data
	cod_bal_dat = labdat_ud.merge(hmidat, on = 'Date', how = 'outer')

	# Dedupe (merging many files, so any duplicates will cause big problems!)
	# cod_bal_dat.drop_duplicates(inplace = True)

	# Calculate daily totals and daily means for each date
	dly_tots  = cod_bal_dat[['Date','Flow In','Flow Out','Meas Biogas Prod']].groupby('Date').sum()
	dly_tots.reset_index(inplace = True)
	dly_means = cod_bal_dat[['Date','Reactor Temp','CODs MS','CODp MS','CODs R','CODp R','CODs Out','CH4%','CO2%']].groupby('Date').mean()
	dly_means.reset_index(inplace = True)

	# Merge and fill in missing values
	cod_bal_dly = dly_tots.merge(dly_means, on = 'Date', how = 'outer')
	cod_bal_dly.set_index('Date')
	cod_bal_dly[['CH4%','CO2%','CODs MS','CODp MS','CODs R','CODp R','CODs Out']] = \
		cod_bal_dly[['CH4%','CO2%','CODs MS','CODp MS','CODs R','CODp R','CODs Out']].interpolate()

	# Get moving average of COD in reactors (data bounce around a lot)
	ma_win = 14
	cod_bal_dly['CODp R'] = cod_bal_dly['CODp R'].rolling(ma_win).mean()
	cod_bal_dly['CODs R'] = cod_bal_dly['CODs R'].rolling(ma_win).mean()
	cod_bal_dly['CODp MS'] = cod_bal_dly['CODp MS'].rolling(ma_win).mean()
	cod_bal_dly['CODs MS'] = cod_bal_dly['CODs MS'].rolling(ma_win).mean()
	cod_bal_dly['CODs Out'] = cod_bal_dly['CODs Out'].rolling(ma_win).mean()

	cod_bal_dly.reset_index(inplace = True)
	
	#=======================================> MERGE & PREP <=======================================	

	#=================================> Estimate COD Consumption <=================================	

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
	cod_bal_dly.loc[:,'CODp R hyd'] = cod_bal_dly['CODp R pot'].values - cod_bal_dly['CODp R'].values
	# Replace negative values with zero (no observable hydrolysis)
	cod_bal_dly.loc[cod_bal_dly['CODp R hyd'] < 0,'CODp R hyd'] = 0

	# Next compute the soluble COD that would accumulate without consumption by the biology
	cod_bal_dly.loc[:,'CODs R pot'] = \
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
	# Consumed COD is the difference between the accumulated vs observed soluble COD (dividing by 1000 to get kg per m^3)
	cod_bal_dly.loc[:,'COD Consumed'] = \
		(cod_bal_dly['CODs R pot'] - cod_bal_dly['CODs R'])*(afbr_vol + afmbr_vol)/1000
	# Replace negative values with zero (no observable COD consumption)
	cod_bal_dly.loc[cod_bal_dly['COD Consumed'] < 0, 'COD Consumed'] = 0

	#=================================> Estimate COD Consumption <=================================	

	#========================================> COD Balance <=======================================	
	
	# Get theoretical estimated methane output
	gasprod_thry = []
	for index,row in cod_bal_dly.iterrows():
		gasprod_thry.append(
			ebg.get_biogas_prod(
				BODrem = row['COD Consumed'], 
				infSO4 = 4, 
				temp = row['Reactor Temp'], 
				percCH4 = row['CH4%']/100, 
				percCO2 = row['CO2%']/100, 
				flowrate = 1, 
				precision = 1E-6
			)
		)

	cod_bal_dly['Thr CH4 Prod'] = [row[0] for row in gasprod_thry]
	cod_bal_dly['Thr Biogas Prod'] = [row[1] for row in gasprod_thry]
	# Actual estimated CH4 production
	cod_bal_dly['Meas CH4 Prod'] = cod_bal_dly['Meas Biogas Prod']*cod_bal_dly['CH4%']/100
	cod_bal_dly['Biogas Discrep (%)'] =	(cod_bal_dly['Meas Biogas Prod']/cod_bal_dly['Thr Biogas Prod'] - 1)*100
	cod_bal_dly.loc[cod_bal_dly['Biogas Discrep (%)'] > 100,'Biogas Discrep (%)'] = 100
	cod_bal_dly['CH4 Discrep (%)']    =	(cod_bal_dly['Meas CH4 Prod']/cod_bal_dly['Thr CH4 Prod'] - 1)*100
	cod_bal_dly.loc[cod_bal_dly['CH4 Discrep (%)'] > 100,'CH4 Discrep (%)'] = 100

	#========================================> COD Balance <=======================================	

	# Output csv with summary statistics
	os.chdir(outdir)
	output_vars = ['Date','Meas Biogas Prod','Thr Biogas Prod','Meas CH4 Prod','Thr CH4 Prod','Biogas Discrep (%)','CH4 Discrep (%)']
	cod_bal_dly.to_csv('COD Balance Full.csv')
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
	'12-10-17',
	1,
	'hour',
	outdir = '/Users/josebolorinos/Google Drive/Codiga Center/Miscellany'
	# run_agg_temp = True 

	
)


