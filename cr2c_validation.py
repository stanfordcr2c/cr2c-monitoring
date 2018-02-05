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
import math
import functools
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory
import cr2c_utils as cut
import cr2c_labdata as pld
import cr2c_hmidata as hmi
from cr2c_hmidata import hmi_data_agg as hmi_run


def adj_Hcp(Hcp_gas, deriv_gas, temp):
	return Hcp_gas*math.exp(deriv_gas*(1/(273 + temp) - (1/298)))

def est_biogas_prod(BODrem, infSO4, temp, percCH4, percCO2, flowrate, precision):
	
	# =======> UNITS OF INPUT VARIABLES <=======
	# infBOD_ult/infSO4 in mg/L, 
	# temp in C 
	# percents as decimals, 
	# fowrate as m^3/day
	# precision as a decimal


	# Assumed Henry's constants (from Sander 2015)
	# Units of mM/atm @ 25 degrees C
	Hcp_CH4 = 1.4
	Hcp_CO2 = 33.4
	Hcp_H2S = 101.325
	Hcp_N2  = 0.65

	# Assumed Clausius-Clapeyron Constants (dlnHcp/d(l/T))
	deriv_ccCH4 = 1600
	deriv_ccCO2 = 2300
	deriv_ccH2S = 2100
	deriv_ccN2  = 1300

	# Volume of gas at STP (mL/mmol)
	Vol_STP = 22.4
	cubicft_p_L = 0.03531467

	# Assumed N2 in air
	percN2 = 0.78

	# Assumed fraction of electrons used for respiration
	fe = 0.9

	# Observed pH at Korean plant (range 6.6-6.8)
	pH = 6.7 

	# Assumed BOD and SO4 removals 
	# (observed at previous pilot tests of 93%-100% and 94%-97% at 59-77 degrees F)
	perc_BODrem = 0.95
	perc_SO4rem = 0.96
	# Adjust gas constants to temperature
	Hcp_CH4_adj = adj_Hcp(Hcp_CH4, deriv_ccCH4, temp)
	Hcp_CO2_adj = adj_Hcp(Hcp_CO2, deriv_ccCO2, temp)
	Hcp_H2S_adj = adj_Hcp(Hcp_H2S, deriv_ccH2S, temp)
	Hcp_N2_adj  = adj_Hcp(Hcp_N2,  deriv_ccN2,  temp)
	Vol_adj     = Vol_STP*(temp + 273)/273
	# Get estimated CH4 production from BOD in wastewater
	# BOD removed from methanogens (minus BOD from SO4 reducers, 1.5g SO4 reduced by 1 g BOD)
	# and Converted to CH4 in wastewater
	BODrem_SO4 = infSO4*perc_SO4rem/(1.5*fe)
	BODconv_CH4 = (BODrem - BODrem_SO4)*fe 
	# Moles of CH4: 1 mole of CH4 is 64 g of BOD, gets mol CH4 per cubic m (mmol/L)
	CH4_prod_mol = BODconv_CH4/64
	H2S_prod_mol = infSO4*perc_SO4rem/96
	# CO2 estimate assumes given fraction of CH4 in biogas (by volume!)
	CO2_prod_mol = CH4_prod_mol*percCO2/percCH4
	# N2 estimate (not production per se) assumes equilibrium partitioning between air and water 
	N2_prod_mol  = percN2*Hcp_N2_adj
	# Get molar total for biogas
	gas_prod_mol = CH4_prod_mol + CO2_prod_mol + H2S_prod_mol + N2_prod_mol
	# Start with initial amount gas that partitions out of solution into headspace
	# (assume 50% of total volume of gas produced) as well as the percentage discrepancy
	# (start off at 50%)
	gas_part_mol = 0.5*gas_prod_mol
	balance_perc = -0.5
	# Perform loops necessary to get within desired level of precision
	while abs(balance_perc) >= precision:
		try:
			# Update the assumed amount of gas partitioned into the headspace
			gas_part_mol   = gas_part_mol*(1 + balance_perc)
			# Calculate the equilibrium partitioning of each gas into this amount of gas 
			# (at the given temp and pressure)
			CH4_gas_eq_mol = CH4_prod_mol/(1 + (Hcp_CH4_adj/gas_part_mol))
			CO2_gas_eq_mol = CO2_prod_mol/(1 + (Hcp_CO2_adj/gas_part_mol))
			N2_gas_eq_mol  = N2_prod_mol /(1 + (Hcp_N2_adj /gas_part_mol))
			H2S_gas_eq_mol = H2S_prod_mol/(1 + (Hcp_H2S_adj/gas_part_mol))
			gas_eq_mol     = CH4_gas_eq_mol + CO2_gas_eq_mol + H2S_gas_eq_mol + N2_gas_eq_mol
			# Compare partitioned gas calculation to original amount assumed to have partitioned into the gas phase
			balance_perc = (gas_eq_mol - gas_part_mol)/gas_part_mol
			# Update CH4/Biogas calculations
			percCH4_biogas = CH4_gas_eq_mol/gas_eq_mol 
			CH4_gas_vol    = CH4_gas_eq_mol*Vol_adj*flowrate
			biogas_gas_vol = CH4_gas_vol/percCH4_biogas
		except ZeroDivisionError:
			CH4_gas_vol, biogas_gas_vol = 0,0
			break

	return [CH4_gas_vol, biogas_gas_vol]


# Function to estimate the sum of a set of variables in a pandas dataframe
def get_sumvar(df, coefs):

	nvars = len(coefs)

	# Make sure there are as many coefficients as variables being summed
	# Need to make this a proper error message
	if len(df.columns) != nvars:
		print('The number of coefficients and DataFrame columns must be equal!')
		sys.exit()

	# Convert the list of coefficients to a length nvars vector
	coefs = np.array(coefs).reshape(nvars,1)
	# Create nvar copies of that vector
	coefcop = np.tile(coefs,(1,nvars))
	# Get all pairwise products of coefficients
	coefmat = np.matmul(coefcop,np.eye(nvars)*coefs)
	# Get covariance matrix from columns in dataset
	cov = df.cov().values
	# Sum the elementwise coefficient products by the covariances to get variance of sum
	sumvar = np.multiply(cov, coefmat).sum().sum()

	return sumvar

def get_cod_bal(
	start_dt_str,
	end_dt_str,
	outdir = None,
	hmi_path = None,
	run_agg_feeding = False,
	run_agg_gasprod = False,
	run_agg_temp = False,
):
	
	# Window for moving average calculation
	ma_win = 1
	start_dt = dt.strptime(start_dt_str,'%m-%d-%y').date()
	start_dt_query = start_dt - timedelta(days = ma_win)
	start_dt_qstr = dt.strftime(start_dt_query,'%m-%d-%y')
	end_dt   = dt.strptime(end_dt_str,'%m-%d-%y').date()

	# This variable might not be necessary in the future, should probably remove
	tperiod = 1

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

	# L in a mol of gas at STP
	Vol_STP = 22.4

	#=========================================> HMI DATA <=========================================
	
	# If requested, run the hmi_data_agg script for the reactor meters and time period of interest
	if run_agg_feeding or run_agg_gasprod or run_agg_temp:
		get_hmi = hmi_run(start_dt_str, end_dt_str, hmi_path = hmi_path)
	if run_agg_feeding:
		get_hmi.run_report(
			[tperiod]*2, # Number of hours you want to average over
			['HOUR']*2, # Type of time period (can be "hour" or "minute")
			[inf_elid, eff_elid], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
			['water']*2, # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
		)	
	if run_agg_gasprod:
		get_hmi.run_report(
			[tperiod]*len(gas_elids), # Number of hours you want to average over
			['HOUR']*len(gas_elids), # Type of time period (can be "hour" or "minute")
			gas_elids, # Sensor ids that you want summary data for (have to be in HMI data file obviously)
			['gas']*len(gas_elids), # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
		)
	if run_agg_temp:
		get_hmi.run_report(
			[tperiod]*len(temp_elids), # Number of hours you want to average over
			['HOUR']*len(temp_elids), # Type of time period (can be "hour" or "minute")
			temp_elids, # Sensor ids that you want summary data for (have to be in HMI data file obviously)
			['temp']*len(temp_elids), # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
		)

	# Read in the data
	year = start_dt.year

	gasprod_dat = hmi.get_data(
		gas_elids,
		[tperiod]*len(gas_elids),
		['HOUR']*len(gas_elids), 
		start_dt_str = start_dt_str, 
		end_dt_str = end_dt_str
	)
	# Do the same for feeding and temperature
	feeding_dat = hmi.get_data(
		[inf_elid, eff_elid],
		[tperiod]*2, 
		['HOUR']*2, 
		start_dt_str = start_dt_str,
		end_dt_str = end_dt_str
	)
	temp_dat = hmi.get_data(
		temp_elids,
		[tperiod]*len(temp_elids), 
		['HOUR']*len(temp_elids), 
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
	gasprod_dat['Meas Biogas Prod'] = (gasprod_dat['FT700'] + gasprod_dat['FT704'])*60*tperiod
	gasprod_dat_cln                 = gasprod_dat[['Time','Meas Biogas Prod']]

	# Feeding HMI Data
	feeding_dat['Flow In']  = feeding_dat[inf_elid]*60*tperiod*l_p_gal
	feeding_dat['Flow Out'] = feeding_dat[eff_elid]*60*tperiod*l_p_gal
	feeding_dat_cln         = feeding_dat[['Time','Flow In','Flow Out']]

	# Reactor Temperature HMI data
	temp_dat['Reactor Temp'] = temp_dat[temp_elids].mean(axis = 1)
	temp_dat_cln             = temp_dat[['Time','Reactor Temp']]

	# List of hmi dataframes
	hmi_dflist = [feeding_dat_cln, gasprod_dat_cln]
	# Merge hmi datasets
	hmidat = functools.reduce(lambda left,right: pd.merge(left,right, on='Time', how = 'outer'), hmi_dflist)
	hmidat['Date'] = hmidat['Time'].dt.date
	#=========================================> HMI DATA <=========================================

	#=========================================> LAB DATA <=========================================
	
	# Get lab data from file on box and filter to desired dates
	labdat  = pld.labrun().get_data(['COD','Sulfate','GasComp'], start_dt_qstr, end_dt_str)

	# COD data
	cod_dat = labdat['COD']
	cod_dat['Date'] = cod_dat['Date_Time'].dt.date

	# Drop duplicates
	cod_dat.drop_duplicates(keep = 'first', inplace = True)
	# Get average of multiple values taken on same day
	cod_dat = cod_dat.groupby(['Date','Stage','Type']).mean()

	# Convert to wide to get COD in and out of the reactors
	cod_dat_wide = cod_dat.unstack(['Stage','Type'])
	cod_dat_wide['CODt MS'] = cod_dat_wide['Value']['Microscreen']['Total']
	# Weighted aveage COD concentrations in the reactors
	cod_dat_wide['CODt R'] = \
		(cod_dat_wide['Value']['AFBR']['Total']*afbr_vol +\
		cod_dat_wide['Value']['Duty AFMBR MLSS']['Total']*afmbr_vol)/\
		(afbr_vol + afmbr_vol)
	cod_dat_wide['CODt Out'] = cod_dat_wide['Value']['Duty AFMBR Effluent']['Total']
	cod_dat_wide.reset_index(inplace = True)
	cod_dat_cln = cod_dat_wide[['Date','CODt MS','CODt R','CODt Out']]
	cod_dat_cln.columns = ['Date','CODt MS','CODt R','CODt Out']

	# Sulfate data
	so4_dat = labdat['Sulfate']
	so4_dat['Date'] = so4_dat['Date_Time']
	so4_dat.set_index(['Date','Stage'], inplace = True)
	so4_dat_wide = so4_dat.unstack(['Stage'])
	so4_dat_wide['SO4 MS'] = so4_dat_wide['Value']['Microscreen']
	so4_dat_wide.reset_index(inplace = True)
	so4_dat_cln = so4_dat_wide[['Date','SO4 MS']]
	so4_dat_cln.columns = ['Date','SO4 MS']
	so4_dat_cln.loc[:,'Date'] = so4_dat_cln['Date'].dt.date

	# Gas Composition Data
	gc_dat = labdat['GasComp']
	gc_dat['Date'] = gc_dat['Date_Time'].dt.date
	gc_dat = gc_dat.loc[(gc_dat['Type'].isin(['Methane (%)','Carbon Dioxide (%)']))]
	gc_dat = gc_dat.groupby(['Date','Type']).mean()
	gc_dat_wide = gc_dat.unstack('Type')
	gc_dat_wide['CH4%'] = gc_dat_wide['Value']['Methane (%)']
	gc_dat_wide['CO2%'] = gc_dat_wide['Value']['Carbon Dioxide (%)']
	gc_dat_wide.reset_index(inplace = True)
	gc_dat_cln = gc_dat_wide[['Date','CH4%','CO2%']]
	gc_dat_cln.columns = ['Date','CH4%','CO2%']

	# Solids Wasting Data
	waste_dat = cut.get_gsheet_data(['Wasted Solids'])
	waste_dat['Date'] = pd.to_datetime(waste_dat['Date']).dt.date
	waste_dat['Volume (gallons)'] = waste_dat['Volume (gallons)'].astype('float')
	waste_dat['Wasted (L)'] = waste_dat['Volume (gallons)']*l_p_gal
	waste_dat = waste_dat.loc[(waste_dat['Date'] >= start_dt) & (waste_dat['Date'] <= end_dt),:]
	waste_dat_cln = waste_dat[['Date','Wasted (L)']]
	# List of lab dataframes
	lab_dflist = [cod_dat_cln, gc_dat_cln, waste_dat_cln, so4_dat_cln]

	# Merge lab datasets
	labdat = functools.reduce(lambda left,right: pd.merge(left,right, on='Date', how = 'outer'), lab_dflist)
	# Get daily average of readings if multiple readings in a day (also prevents merging issues!)
	labdat_ud = labdat.groupby('Date').mean()
	labdat_ud.reset_index(inplace = True)

	#=========================================> LAB DATA <=========================================

	#=======================================> MERGE & PREP <=======================================	
	
	# Merge Lab and HMI 
	cod_bal_dat = labdat_ud.merge(hmidat, on = 'Date', how = 'outer')

	# Dedupe (merging many files, so any duplicates will cause big problems!)
	cod_bal_dat.drop_duplicates(inplace = True)

	# Calculate daily totals and daily means for each date
	dly_tots  = cod_bal_dat[['Date','Flow In','Flow Out','Meas Biogas Prod']].groupby('Date').sum()
	dly_tots.reset_index(inplace = True)
	dly_means = cod_bal_dat[['Date','CODt MS','CODt R','CODt Out','SO4 MS','CH4%','CO2%','Wasted (L)']].groupby('Date').mean()
	dly_means.reset_index(inplace = True)

	# Merge and fill in missing values
	cod_bal_dly = dly_tots.merge(dly_means, on = 'Date', how = 'outer')
	cod_bal_dly.set_index('Date')
	cod_bal_dly[['CH4%','CO2%','CODt MS','CODt R','CODt Out','SO4 MS']] = \
		cod_bal_dly[['CH4%','CO2%','CODt MS','CODt R','CODt Out','SO4 MS']].interpolate()

	# Convert missing wasting data to 0 (assume no solids wasted that day)
	cod_bal_dly.loc[np.isnan(cod_bal_dly['Wasted (L)']),'Wasted (L)'] = 0

	# Get moving average of COD in reactors (data bounce around a lot)
	cod_cols = ['CODt MS','CODt R','CODt Out']
	cod_bal_dly[cod_cols] = cod_bal_dly[cod_cols].rolling(ma_win).mean()
	# Eliminate missing values (from period prior to start_dt) and reset index
	# cod_bal_dly.dropna(axis = 0, how = 'any', inplace = True)
	cod_bal_dly.reset_index(inplace = True)

	# Put dates into weekly bins (relative to end date), denoted by beginning of week
	cod_bal_dly['Weeks Back'] = pd.to_timedelta(np.floor((cod_bal_dly['Date'] - end_dt)/np.timedelta64(7,'D'))*7, unit = 'D') - timedelta(days = -7)
	cod_bal_dly['Week Start'] = end_dt + cod_bal_dly['Weeks Back']
 
	#=======================================> MERGE & PREP <=======================================	

	#========================================> COD Balance <=======================================	
	# Note: dividing by 1E6 to express in kg
	# COD coming in from the Microscreen
	cod_bal_dly['COD In']   = cod_bal_dly['CODt MS']*cod_bal_dly['Flow In']/1E6
	# COD leaving the reactor
	cod_bal_dly['COD Out'] = cod_bal_dly['CODt Out']*cod_bal_dly['Flow Out']/1E6
	# COD wasted
	cod_bal_dly['Wasted']   = cod_bal_dly['CODt R']*cod_bal_dly['Wasted (L)']/1E6
	# COD content of gas (assumes that volume given by flowmeter is in STP)
	cod_bal_dly['Biogas']   = cod_bal_dly['Meas Biogas Prod']*cod_bal_dly['CH4%']/100/Vol_STP*64/1000
	# COD from sulfate reduction (1.5g COD per g SO4)
	cod_bal_dly['Sulfate Reduction'] = cod_bal_dly['SO4 MS']*cod_bal_dly['Flow In']/1.5/1E6
	#========================================> COD Balance <=======================================	

	cod_bal_wkly = cod_bal_dly.groupby('Week Start').sum(numeric_only = True)
	cod_bal_wkly.reset_index(inplace = True)
	cod_bal_wkly.loc[:,'Week Start'] = cod_bal_wkly['Week Start'].dt.date
	cod_bal_wkly = cod_bal_wkly.loc[cod_bal_wkly['Week Start'] < end_dt,:]

	#===========================================> Plot! <==========================================
	nWeeks = np.arange(len(cod_bal_wkly))
	bWidth = 0.8
	pIn = plt.scatter(nWeeks,cod_bal_wkly['COD In'], c = 'r')
	pBiogas = plt.bar(nWeeks,cod_bal_wkly['Biogas'], bWidth)
	pOut = plt.bar(nWeeks,cod_bal_wkly['COD Out'], bWidth, bottom = cod_bal_wkly['Biogas'])
	pWasted = plt.bar(nWeeks,cod_bal_wkly['Wasted'], bWidth, bottom = cod_bal_wkly['COD Out'] + cod_bal_wkly['Biogas'])
	pSO4 = plt.bar(nWeeks,cod_bal_wkly['Sulfate Reduction'], bWidth, bottom = cod_bal_wkly['COD Out'] + cod_bal_wkly['Biogas'] + cod_bal_wkly['Wasted'])
	plt.xticks(nWeeks,cod_bal_wkly['Week Start'], rotation = 45) 
	plt.legend((pIn,pBiogas[0],pOut[0],pWasted[0],pSO4[0]),('COD In','Biogas','COD Out','Solids Wasting','Sulfate Reduction'))
	plt.ylabel('kg of COD Equivalents')
	plt.tight_layout()
	plt.savefig(
		os.path.join(outdir, 'COD Balance.png'),
		width = 30,
		height = 50
	) 
	plt.close()
	#===========================================> Plot! <==========================================		


'''
Verify pressure sensor readings from HMI data and manometer readings from Google sheets.
Calculate water head from pressure sensor readings, and compare it with the manometer readings.
Plot the merged data to show results.
'''
def pressure_validation(
	start_dt_str,
	end_dt_str,
	pr_elids,
	field_ids,
	run_report = False,
	hmi_path = None
):

    # Get field pressure measurements
    #            AFBR  RAFMBR DAFMBR
    reactors = ['R300','R301','R302']
    fieldvals_sheet = cut.get_gsheet_data(['DailyLogResponses'])
    fieldvals_list = fieldvals_sheet[0]['values']
    headers = ['TimeStamp'] + fieldvals_list.pop(0)[1:]
    fieldvals_df = pd.DataFrame(fieldvals_list, columns = headers)
    pdat_field = fieldvals_df[['TimeStamp'] + ['Manometer Pressure: ' + reactor for reactor in reactors]]
    pdat_field['TimeStamp'] = pd.to_datetime(pdat_field['TimeStamp'])
    pdat_field['TS_mins'] = pdat_field['TimeStamp'].values.astype('datetime64[m]')

    # First subset hmi data to dates for which field measurements are available
    first_lts = pdat_field['TimeStamp'][0]
    last_lts = pdat_field['TimeStamp'][len(pdat_field) - 1]
    first_lts_str = dt.strftime(first_lts, format = '%m-%d-%y')
    last_lts_str = dt.strftime(last_lts, format = '%m-%d-%y')

    # Get HMI pressure data
    # Create time and sensor type variables
    tperiods = [1]*len(pr_elids)
    ttypes = ['MINUTE']*len(pr_elids)
    stypes = ['PRESSURE']*len(pr_elids)

    # load reactor pressures from hmi csv file to sql database if path is provided
    if run_report:
    	get_hmi = hmi_run(start_dt_str, end_dt_str, hmi_path = hmi_path)
    	get_hmi.run_report(
			tperiods, # Number of hours you want to average over
			ttypes, # Type of time period (can be "hour" or "minute")
			pr_elids, # Sensor ids that you want summary data for (have to be in HMI data file obviously)
			stypes # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
		)	

    # get reactor pressures from sql database
    pdat_hmi = hmi.get_data(pr_elids, tperiods, ttypes, 2017)

    for tperiod, ttypes, pr_elid in zip(tperiods, ttypes, pr_elids):
        # Create keys of pressure sensor with specified time period. e.g. 'PIT700_1HOUR_AVERAGES'
        pr_elid_hmi = pr_elid + '_' + str(tperiod) + 'HOUR' + '_AVERAGES'
        # Create columns of gauge pressure for the sensor
        pr_head = pr_elid + ' Gauge Pr. (in)'

        # Convert pressure readings to inches of head (comparable to field measurements)
        pdat_hmi[pr_elid_hmi][pr_head] = pdat_hmi[pr_elid_hmi]['Value'].apply(lambda x: (x - 14.7) * 27.7076)

        # Merge the two datasets only hmi data observations in the field measurement data (minute timescale here)
        pr_head_hmi = pdat_hmi[pr_elid_hmi][['Time', pr_head]]
        if 'merged_pr' not in locals():
            merged_pr = pd.merge_asof(pdat_field, pr_head_hmi, left_on = 'TS_mins', right_on = 'Time')
        else:
            merged_pr = pd.merge_asof(merged_pr, pr_head_hmi, left_on = 'TS_mins', right_on = 'Time')

        # Delete additional Time column
        merged_pr = merged_pr.drop('Time', 1)

    # Plot manometer pressures vs HMI sensor gauge pressure
    nrows = 3
    fig, axes = plt.subplots(nrows, sharex = True)
    fig.set_size_inches(8, 20)
    ids_hmi = ['PIT700', 'PIT702', 'PIT704']
    ids_gsheet = ['R300', 'R301', 'R302']
    for ax_idx, (id_hmi, id_gsheet) in enumerate(zip(ids_hmi, ids_gsheet)):
        axes[ax_idx].plot(merged_pr['TS_mins'], merged_pr[id_hmi + ' Gauge Pr. (in)'])
        axes[ax_idx].plot(merged_pr['TS_mins'], pd.to_numeric(merged_pr['Manometer Pressure: ' + id_gsheet], 'coerce'))
        axes[ax_idx].legend()

    # Display only months and days on the x axis
    date_fmt = dates.DateFormatter('%m/%d')
    axes[ax_idx].xaxis.set_major_formatter(date_fmt)
    plt.show()

# pressure_validation(
# 	'5-10-17',
# 	'10-9-17',
# 	['PIT700','PIT704'],
# 	['Manometer Pressure: R300', 'Manometer Pressure: R302'],
# 	run_report = True
# )


