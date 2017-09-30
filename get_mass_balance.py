'''
	Computes a mass balance for water for any two Flowmeters 
	or for COD-CH4 in the reactor area for any range of dates
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
import est_biogas_prod as ebg

def get_cod_bal(
	start_dt,
	end_dt,
	hmi_path = None, 
	gasprod_path = None, 
	feeding_path = None,
	temp_path = None
):

	start_dt_str = start_dt
	end_dt_str = end_dt
	start_dt = dt.strptime(start_dt,'%m-%d-%y')
	end_dt   = dt.strptime(end_dt,'%m-%d-%y')

	gas_elids  = ['FT700','FT704']
	temp_elids = ['AT304','AT310']
	inf_elid   = 'FT202'
	eff_elid   = 'FT305'

	# Read in gas production and feeding volumes from HMI (or preprocess them if no volumes file has been provided)
	if gasprod_path:
		gasprod_dly = pd.read_csv(gasprod_path)
	elif hmi_path:
		hmi_gas = hmi.hmi_data_agg('raw','gas')
		gasprod_dly = hmi_gas.run_report(
			24,
			gas_elids,
			['total','total'],
			start_dt_str,
			end_dt_str,
			hmi_path = hmi_path,
			output_csv = 1
		)
	else:
		print('You need to give either a path to the file with HMI data or a summary report of processed HMI data')
		sys.exit()
			
	if feeding_path:
		feeding_dly = pd.read_csv(feeding_path)
	elif hmi_path:
		hmi_water = hmi.hmi_data_agg('raw','water')
		feeding_dly = hmi_water.run_report(
			24,
			[inf_elid, eff_elid],
			['total','total'],
			start_dt_str,
			end_dt_str,
			hmi_path = hmi_path,
			output_csv = 1
		)	

	if temp_path:
		temp_dly = pd.read_csv(temp_path)
	elif hmi_path:
		hmi_temp = hmi.hmi_data_agg('raw','temp')
		temp_dly = hmi_temp.run_report(
			24,
			temp_elids,
			['average','average'],
			start_dt_str,
			end_dt_str,
			hmi_path = hmi_path,
			output_csv = 1
		)

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
	cod_dly = cod_dat.groupby(['Date_Time','Stage','Type']).mean()
	# Convert to wide to get COD in and out of the reactors
	cod_dly_wide = cod_dly.unstack(['Stage','Type'])
	cod_dly_wide['COD In']  = cod_dly_wide['Value']['Microscreen']['Soluble']
	cod_dly_wide['COD Out'] = cod_dly_wide['Value']['Duty AFMBR Effluent']['Soluble']
	cod_dly_wide.reset_index(inplace = True)
	cod_dly_wide['Date'] = cod_dly_wide['Date_Time'].dt.date
	cod_dly_clean = pd.DataFrame(
		cod_dly_wide[['Date','COD In','COD Out']].values,
		columns = ['Date','COD In','COD Out']
	)

	# Gas Composition Data
	gc_dat = labdat['GasComp']
	gc_dat['Date_Time'] = pd.to_datetime(gc_dat['Date_Time'])
	gc_dly = gc_dat.loc[
		(gc_dat['Date_Time'] >= start_dt) &
		(gc_dat['Date_Time'] <= end_dt) &
		(gc_dat['Type'].isin(['Methane (%)','Carbon Dioxide (%)']))
	]
	gc_dly = gc_dly.groupby(['Date_Time','Type']).mean()
	gc_dly_wide = gc_dly.unstack('Type')
	gc_dly_wide['CH4%'] = gc_dly_wide['Value']['Methane (%)']
	gc_dly_wide['CO2%'] = gc_dly_wide['Value']['Carbon Dioxide (%)']
	gc_dly_wide.reset_index(inplace = True)
	gc_dly_wide['Date'] = gc_dly_wide['Date_Time'].dt.date
	gc_dly_clean = pd.DataFrame(
		gc_dly_wide[['Date','CH4%','CO2%']].values,
		columns = ['Date','CH4%','CO2%']
	)


	# Gas Production HMI Data
	gasprod_dly['Time']       = pd.to_datetime(gasprod_dly['Time'])
	gasprod_dly['Date']       = gasprod_dly['Time'].dt.date
	gasprod_dly['Biogas Out'] = gasprod_dly['FT700_TOTAL'] + gasprod_dly['FT704_TOTAL']
	gasprod_dly_clean         = gasprod_dly[['Date','Biogas Out']]

	# Feeding HMI Data
	feeding_dly['Time']     = pd.to_datetime(feeding_dly['Time'])
	feeding_dly['Date']     = feeding_dly['Time'].dt.date
	feeding_dly['Flow In']  = feeding_dly[inf_elid + '_TOTAL']
	feeding_dly['Flow Out'] = feeding_dly[eff_elid + '_TOTAL']
	feeding_dly_clean       = feeding_dly[['Date','Flow In','Flow Out']]

	# Reactor Temperature HMI data
	temp_dly['Time']         = pd.to_datetime(temp_dly['Time'])
	temp_dly['Date']         = temp_dly['Time'].dt.date
	temp_dly['Reactor Temp'] = temp_dly[[elid + '_AVERAGE' for elid in temp_elids]].mean(axis = 1)
	temp_dly_clean = temp_dly[['Date','Reactor Temp']]

	# List of all dataframes
	dfs_dly = [temp_dly_clean, feeding_dly_clean, gasprod_dly_clean, cod_dly_clean, gc_dly_clean]

	# Merge all datasets
	cod_bal_dly = functools.reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs_dly)
	cod_bal_dly.dropna(axis = 0, how = 'any', inplace = True)
	
	# Get actual estimated methane output (L)
	cod_bal_dly['Est CH4 Prod'] = cod_bal_dly['Biogas Out']*cod_bal_dly['CH4%']/100
	cod_bal_dly['COD Consumed'] = cod_bal_dly['COD In'] - cod_bal_dly['COD Out']

	# Get theoretical estimated methane output
	gasprod_thry = []
	for index,row in cod_bal_dly.iterrows():
		gasprod_thry.append(
			ebg.get_biogas_prod(
				BODrem = row['COD Consumed'], 
				infSO4 = 0, 
				temp = row['Reactor Temp'], 
				percCH4 = row['CH4%']/100, 
				percCO2 = row['CO2%']/100, 
				# This script takes units of m3/day
				flowrate = row['Flow In']*0.00378541, 
				precision = 1E-6
			)
		)

	cod_bal_dly['Thr CH4 Prod'] = [row[0] for row in gasprod_thry]
	cod_bal_dly['Thr Biogas Prod'] = [row[1] for row in gasprod_thry]
	cod_bal_dly['CH4 Prod Discrep (%)'] = \
		(cod_bal_dly['Est CH4 Prod'] - cod_bal_dly['Thr CH4 Prod'])/cod_bal_dly['Thr CH4 Prod']	

	cod_bal_dly.to_csv('C:/Users/jbolorinos/Google Drive/Codiga Center/Miscellany/balance.csv')

	# cod_bal_dly[['CH4 Discrep %']] = 


get_cod_bal(
	'7-26-17',
	'9-26-17',
	gasprod_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/Miscellany/HMIGAS_FT700_FT704_07-26-17_09-26-17.csv',
	feeding_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/Miscellany/HMIWATER_FT202_FT305_07-26-17_09-26-17.csv',
	temp_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/Miscellany/HMITEMP_AT304_AT310_07-26-17_09-26-17.csv'
	# hmi_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/Reactor Feeding - Raw_20170927064036.csv'
)


