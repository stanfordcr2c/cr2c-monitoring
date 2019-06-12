'''
	Computes a mass balance for COD-CH4 in the reactor area for any range of dates
	takes dates as inputs and outputs a summary file with mass balance info
'''

# Data Prep
import numpy as np
import pandas as pd
import datetime as datetime
from datetime import timedelta
from datetime import datetime as dt
from pandas import read_excel
import math

# Utilities
import os
import sys
import functools

# CR2C
from dependencies import cr2c_labdata as pld
from dependencies import cr2c_opdata as op
from dependencies.cr2c_opdata import opdata_agg as op_run
from dependencies import cr2c_fielddata as fld
from dependencies import cr2c_utils as cut


class cr2c_validation:

	def __init__(
		self, 
		ip_path = None, 
		run_agg_feeding = False, 
		run_agg_gasprod = False, 
		run_agg_temp = False,
		run_agg_press = False
	):
		
		self.ip_path = ip_path
		self.run_agg_feeding = run_agg_feeding
		self.run_agg_gasprod = run_agg_gasprod
		self.run_agg_temp = run_agg_temp
		self.afbr_vol = 1100 # in L
		self.afmbr_vol = 1700 # in L
		self.react_vol = 4500 # in L


	def adj_Hcp(self, Hcp_gas, deriv_gas, temp):
		return Hcp_gas*math.exp(deriv_gas*(1/(273 + temp) - (1/298)))


	def est_diss_ch4(self, temp, percCH4):
		
		# =======> UNITS OF INPUT VARIABLES <=======
		# gasVol in sL/m 
		# temp in C 
		# percents as decimals x 100
		# Assumed Henry's constants (from Sander 2015)
		# Units of mM/atm @ 25 degrees C
		Hcp_CH4 = 1.4
		# Assumed Clausius-Clapeyron Constants (dlnHcp/d(l/T))
		deriv_ccCH4 = 1900
		# Volume of gas at STP (L/mol)
		Vol_STP = 22.4
		# Adjust gas constants to temperature
		Hcp_CH4_adj = self.adj_Hcp(Hcp_CH4, deriv_ccCH4, temp)
		# Moles of CH4: 1 mole of CH4 is 64 g of BOD
		CH4_gas_atm = percCH4/100
		# Assuming 1atm in reactors 
		# (this is a good assumption, even 10 inches on manometer is equivalent to just 0.02 atm)
		COD_diss_conc = CH4_gas_atm*Hcp_CH4_adj*64

		return COD_diss_conc


	def get_biotech_params(self, end_dt_str, nweeks, if_exists_cod_balance_table = 'append', if_exists_vss_params_table = 'append', output_csv = False, outdir = None):
		
		# Window for moving average calculation
		ma_win = 1
		end_weekday = dt.strptime(end_dt_str,'%m-%d-%y').weekday()
		end_dt   = dt.strptime(end_dt_str,'%m-%d-%y').date() - timedelta(days = end_weekday)
		start_dt = end_dt - timedelta(days = 7*nweeks) 
		start_dt_str = dt.strftime(start_dt, '%m-%d-%y')
		start_dt_query = start_dt - timedelta(days = ma_win)
		start_dt_qstr = dt.strftime(start_dt_query,'%m-%d-%y')

		# op element IDs for gas, temperature and influent/effluent flow meters 
		gas_sids   = ['FT700','FT702','FT704']
		temp_sids  = ['AT304','AT307','AT310']
		inf_sid    = 'FT202'
		eff_sids   = ['FT304','FT305']

		# Reactor volumes
		l_p_gal = 3.78541 # Liters/Gallon
		# L in a mol of gas at STP
		Vol_STP = 22.4

		#=========================================> HMI DATA <=========================================
		
		# If requested, run the op_data_agg script for the reactor meters and time period of interest
		if self.run_agg_feeding or self.run_agg_gasprod or self.run_agg_temp:
			get_op = op_run(start_dt_str, end_dt_str, ip_path = self.ip_path)
		if self.run_agg_feeding:
			get_op.run_agg(
				['water']*2, # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
				[inf_sid, eff_sid], # Sensor ids that you want summary data for (have to be in op data file obviously)
				[1]*2, # Number of hours you want to average over
				['HOUR']*2 # Type of time period (can be "hour" or "minute")
			)	
		if self.run_agg_gasprod:
			get_op.run_agg(
				['GAS']*len(gas_sids), # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
				gas_sids, # Sensor ids that you want summary data for (have to be in op data file obviously)
				[1]*len(gas_sids), # Number of hours you want to average over
				['HOUR']*len(gas_sids), # Type of time period (can be "hour" or "minute")
			)
		if self.run_agg_temp:
			get_op.run_agg(
				['TEMP']*len(temp_sids), # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
				temp_sids, # Sensor ids that you want summary data for (have to be in op data file obviously)
				[1]*len(temp_sids), # Number of hours you want to average over
				['HOUR']*len(temp_sids), # Type of time period (can be "hour" or "minute")
			)

		# Get gas production data data
		gasprod_dat = cut.get_data(
			'opdata',
			['GAS_{}_1_HOUR_AVERAGES'.format(sid) for sid in gas_sids],
			start_dt_str = start_dt_str, 
			end_dt_str = end_dt_str			
		)
		# Feeding and temperature data
		water_sids = [inf_sid] + eff_sids
		feeding_dat = cut.get_data(
			'opdata',
			['WATER_{}_1_HOUR_AVERAGES'.format(sid) for sid in water_sids],
			start_dt_str = start_dt_str, 
			end_dt_str = end_dt_str			
		)
		temp_dat = cut.get_data(
			'opdata',
			['TEMP_{}_1_HOUR_AVERAGES'.format(sid) for sid in temp_sids],
			start_dt_str = start_dt_str, 
			end_dt_str = end_dt_str			
		) 

		# Merge each type of opdata to single wide table
		gasprod_dat = cut.merge_tables(gasprod_dat, ['Time'],['Value']*len(gas_sids), merged_varnames = gas_sids)
		feeding_dat = cut.merge_tables(feeding_dat, ['Time'],['Value']*len(water_sids), merged_varnames = water_sids)
		temp_dat = cut.merge_tables(temp_dat, ['Time'],['Value']*len(temp_sids), merged_varnames = temp_sids)

		# Prep the op data
		gasprod_dat['Meas Biogas Prod'] = (gasprod_dat['FT700'] + gasprod_dat['FT702'] + gasprod_dat['FT704'])*60
		gasprod_dat['Date'] = gasprod_dat['Time'].dt.date
		gasprod_dat_cln = gasprod_dat[['Date','Meas Biogas Prod']]
		gasprod_dat_cln = gasprod_dat_cln.groupby('Date').sum()
		gasprod_dat_cln.reset_index(inplace = True)

		# Feeding op Data
		feeding_dat['Flow In']  = feeding_dat[inf_sid]*60*l_p_gal
		feeding_dat['Flow Out'] = (feeding_dat[eff_sids[0]] + feeding_dat[eff_sids[1]])*60*l_p_gal
		feeding_dat['Date'] = feeding_dat['Time'].dt.date
		feeding_dat_cln = feeding_dat[['Date','Flow In','Flow Out']]
		feeding_dat_cln = feeding_dat_cln.groupby('Date').sum()
		feeding_dat_cln.reset_index(inplace = True)

		# Reactor Temperature op data
		temp_dat['Reactor Temp (C)'] = \
			(temp_dat['AT304']*self.afbr_vol + (temp_dat['AT307'] + temp_dat['AT310'])*self.afmbr_vol)/self.react_vol
		temp_dat['Date'] = temp_dat['Time'].dt.date
		temp_dat_cln = temp_dat[['Date','Reactor Temp (C)']]
		temp_dat_cln = temp_dat_cln.groupby('Date').mean()
		temp_dat_cln.reset_index(inplace = True)

		# List of op dataframes
		op_dflist = [feeding_dat_cln, gasprod_dat_cln, temp_dat_cln]
		# Merge op datasets
		opdat_ud = functools.reduce(
			lambda left,right: pd.merge(left,right, on = 'Date', how = 'outer'), 
			op_dflist
		)
		#=========================================> HMI DATA <=========================================

		#=========================================> LAB DATA <=========================================
		# Get lab data from file on box and filter to desired dates
		labdat  = cut.get_data('labdata',['COD','TSS_VSS','SULFATE','GASCOMP'])

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
			(cod_dat_wide['Value']['AFBR']['Total']*self.afbr_vol +\
			(cod_dat_wide['Value']['Research AFMBR MLSS']['Total'] + cod_dat_wide['Value']['Duty AFMBR MLSS']['Total'])*self.afmbr_vol)/\
			(self.react_vol)
		cod_dat_wide['CODt Out'] = cod_dat_wide['Value']['Research AFMBR Effluent']['Total'] + cod_dat_wide['Value']['Duty AFMBR Effluent']['Total']
		cod_dat_wide.reset_index(inplace = True)
		cod_dat_cln = cod_dat_wide[['Date','CODt MS','CODt R','CODt Out']]
		cod_dat_cln.columns = ['Date','CODt MS','CODt R','CODt Out']

		# Gas Composition Data
		gc_dat = labdat['GASCOMP']
		gc_dat['Date'] = gc_dat['Date_Time'].dt.date
		gc_dat = gc_dat.loc[(gc_dat['Type'].isin(['Methane (%)','Carbon Dioxide (%)']))]
		gc_dat = gc_dat.groupby(['Date','Type']).mean()
		gc_dat_wide = gc_dat.unstack('Type')
		gc_dat_wide['CH4%'] = gc_dat_wide['Value']['Methane (%)']
		gc_dat_wide['CO2%'] = gc_dat_wide['Value']['Carbon Dioxide (%)']
		gc_dat_wide.reset_index(inplace = True)
		gc_dat_cln = gc_dat_wide[['Date','CH4%','CO2%']]
		gc_dat_cln.columns = ['Date','CH4%','CO2%']

		# VSS Data
		vss_dat = labdat['TSS_VSS']
		vss_dat['Date'] = vss_dat['Date_Time'].dt.date
		# Drop duplicates
		vss_dat.drop_duplicates(keep = 'first', inplace = True)
		# Get average of multiple values taken on same day
		vss_dat = vss_dat.groupby(['Date','Stage','Type']).mean()

		# Convert to wide to get COD in and out of the reactors
		vss_dat_wide = vss_dat.unstack(['Stage','Type'])
		# Weighted aveage COD concentrations in the reactors
		vss_dat_wide['VSS R'] = \
			(
				vss_dat_wide['Value']['AFBR']['VSS']*self.afbr_vol +\
				vss_dat_wide['Value']['Research AFMBR MLSS']['VSS']*self.afmbr_vol +\
				vss_dat_wide['Value']['Duty AFMBR MLSS']['VSS']*self.afmbr_vol
			)/\
			(self.afbr_vol + self.afmbr_vol)
		vss_dat_wide['VSS Out'] = \
			vss_dat_wide['Value']['Research AFMBR Effluent']['VSS'] +\
			vss_dat_wide['Value']['Duty AFMBR Effluent']['VSS']
		vss_dat_wide.reset_index(inplace = True)
		vss_dat_cln = vss_dat_wide[['Date','VSS R','VSS Out']]
		vss_dat_cln.columns = ['Date','VSS R','VSS Out']	

		# Solids Wasting Data (from gsheet)
		waste_dat = cut.get_data('labdata',['WASTED_SOLIDS'])['WASTED_SOLIDS']
		waste_dat['Date'] = pd.to_datetime(waste_dat['Date_Time']).dt.date
		waste_dat['Wasted (L)'] = waste_dat['Value']*l_p_gal
		waste_dat_cln = waste_dat[['Date','Wasted (L)']]

		# Sulfate data
		so4_dat = labdat['SULFATE']
		so4_dat['Date'] = so4_dat['Date_Time']
		so4_dat = so4_dat.groupby(['Date','Stage']).mean()
		so4_dat_wide = so4_dat.unstack(['Stage'])
		so4_dat_wide['SO4 MS'] = so4_dat_wide['Value']['Microscreen']
		so4_dat_wide.reset_index(inplace = True)
		so4_dat_cln = so4_dat_wide[['Date','SO4 MS']]
		so4_dat_cln.columns = ['Date','SO4 MS']
		so4_dat_cln.loc[:,'Date'] = so4_dat_cln['Date'].dt.date
		
		# List of lab dataframes
		lab_dflist = [cod_dat_cln, gc_dat_cln, waste_dat_cln, so4_dat_cln, vss_dat_cln]

		# Merge lab datasets
		labdat = functools.reduce(lambda left,right: pd.merge(left,right, on='Date', how = 'outer'), lab_dflist)
		# Get daily average of readings if multiple readings in a day (also prevents merging issues!)
		labdat_ud = labdat.groupby('Date').mean()
		labdat_ud.reset_index(inplace = True)
		#=========================================> LAB DATA <=========================================

		#=======================================> MERGE & PREP <=======================================		
		
		# Merge Lab and op
		cod_bal_dat = labdat_ud.merge(opdat_ud, on = 'Date', how = 'outer')
		# Dedupe (merging many files, so any duplicates can cause big problems!)
		cod_bal_dat.drop_duplicates(inplace = True)

		# Convert missing wasting data to 0 (assume no solids wasted that day)
		cod_bal_dat.loc[np.isnan(cod_bal_dat['Wasted (L)']),'Wasted (L)'] = 0
		# Fill in missing lab data
		# First get means of observed data
		cod_bal_means = \
			cod_bal_dat[[
				'CH4%','CO2%',
				'CODt MS','CODt R','CODt Out',
				'VSS R','VSS Out',
				'SO4 MS'
			]].mean()
		# Then interpolate
		cod_bal_dat.sort_values(['Date'], inplace = True)
		cod_bal_dat.set_index('Date', inplace = True)
		cod_bal_dat[[
			'CH4%','CO2%',
			'CODt MS','CODt R','CODt Out',
			'VSS R','VSS Out',
			'SO4 MS'
		]] = \
			cod_bal_dat[[
				'CH4%','CO2%',
				'CODt MS','CODt R','CODt Out',
				'VSS R','VSS Out',
				'SO4 MS'
			]].interpolate()

		# Then fill remaining missing values with the means of all variables
		fill_values = {
			'CH4%': cod_bal_means['CH4%'],
			'CO2%': cod_bal_means['CO2%'],
			'CODt MS': cod_bal_means['CODt MS'],
			'CODt R': cod_bal_means['CODt R'],
			'CODt Out': cod_bal_means['CODt Out'],
			'VSS R': cod_bal_means['VSS R'],
			'VSS Out': cod_bal_means['VSS Out'],
			'SO4 MS': cod_bal_means['SO4 MS']
		}
		cod_bal_dat.fillna(value = fill_values, inplace = True)

		# Get moving average of COD in reactors (data bounce around a lot)
		cod_cols = ['CODt MS','CODt R','CODt Out']
		cod_bal_dat[cod_cols] = cod_bal_dat[cod_cols].rolling(ma_win).mean()
		# Reset index
		cod_bal_dat.reset_index(inplace = True)
		# Put dates into weekly bins (relative to end date), denoted by beginning of week
		cod_bal_dat['Weeks Back'] = \
			pd.to_timedelta(np.floor((cod_bal_dat['Date'] - end_dt)/np.timedelta64(7,'D'))*7, unit = 'D')
		cod_bal_dat['Week Start'] = pd.to_datetime(end_dt) + cod_bal_dat['Weeks Back']
		cod_bal_dat = cod_bal_dat.loc[
			(cod_bal_dat['Date'] >= start_dt) & (cod_bal_dat['Date'] <= end_dt),
			:
		]

		#=======================================> MERGE & PREP <=======================================	

		#========================================> COD Balance <=======================================	
		# Note: dividing by 1E6 to express in kg
		# COD coming in from the Microscreen
		cod_bal_dat['COD In']   = cod_bal_dat['CODt MS']*cod_bal_dat['Flow In']/1E6
		# COD leaving the reactor
		cod_bal_dat['COD Out']  = cod_bal_dat['CODt Out']*cod_bal_dat['Flow Out']/1E6
		# COD wasted
		cod_bal_dat['COD Wasted'] = cod_bal_dat['CODt R']*cod_bal_dat['Wasted (L)']/1E6
		# COD content of gas (assumes that volume given by flowmeter is in STP)
		cod_bal_dat['Biogas']   = cod_bal_dat['Meas Biogas Prod']*cod_bal_dat['CH4%']/100/Vol_STP*64/1000
		# COD content of dissolved methane (estimated from temperature of reactors)
		cod_diss_conc = map(
			self.est_diss_ch4,
			cod_bal_dat['Reactor Temp (C)'].values, 
			cod_bal_dat['CH4%'].values
		)

		cod_bal_dat['Dissolved CH4'] = np.array(list(cod_diss_conc))*cod_bal_dat['Flow Out']/1E6
		# COD from sulfate reduction (1.5g COD per g SO4, units are in mg/L S)
		cod_bal_dat['Sulfate Reduction'] = cod_bal_dat['SO4 MS']*cod_bal_dat['Flow In']/1.5/1E6*48/16
		#========================================> COD Balance <=======================================	

		# Convert to weekly data
		cod_bal_wkly = cod_bal_dat.groupby('Week Start').sum(numeric_only = True)
		cod_bal_wkly.reset_index(inplace = True)
		cod_bal_wkly.loc[:,'Week Start'] = cod_bal_wkly['Week Start'].dt.date
		cod_bal_wkly = cod_bal_wkly.loc[cod_bal_wkly['Week Start'] <= end_dt,:]

		# Dividing by 1E6 and 7 because units are totals for week and are in mg/L
		# whereas COD units are in kg
		cod_bal_wkly['gVSS wasted/gCOD Removed'] = \
			(
				cod_bal_wkly['VSS R']*cod_bal_wkly['Wasted (L)'] + 
				cod_bal_wkly['VSS Out']*cod_bal_wkly['Flow Out']
			)/1E6/7/\
			(cod_bal_wkly['COD In'] - cod_bal_wkly['COD Out'] - cod_bal_wkly['Sulfate Reduction'])

		# No need to divide VSS concentration by 1E6 or 7 because same units in numerator and denominator
		cod_bal_wkly['VSS SRT (days)'] = (self.afbr_vol + self.afmbr_vol)/cod_bal_wkly['Wasted (L)']

		cod_bal_long = pd.melt(
			cod_bal_wkly, 
			id_vars = ['Week Start'], 
			value_vars = ['COD In','COD Out','Biogas','COD Wasted','Dissolved CH4','Sulfate Reduction','gVSS wasted/gCOD Removed','VSS SRT (days)']
		)


		cod_bal_long.columns = ['Date_Time','Type','Value']
		# Create key unique by Date_Time, Stage, Type, and obs_id
		cod_bal_long.loc[:,'Dkey'] = cod_bal_long['Date_Time'].astype(str) + cod_bal_long['Type']
		# Reorder columns to put DKey as first column
		colnames = list(cod_bal_long.columns.values)
		cod_bal_long = cod_bal_long[colnames[-1:] + colnames[0:-1]]

		# Split into cod_balance and vss_params
		vss_params_long = cod_bal_long.loc[cod_bal_long['Type'].isin(['gVSS wasted/gCOD Removed','VSS SRT (days)']),:]
		cod_bal_long = cod_bal_long.loc[~cod_bal_long['Type'].isin(['gVSS wasted/gCOD Removed','VSS SRT (days)']),:]

		if output_csv:
			cod_bal_long.to_csv(
				os.path.join(outdir, 'COD Balance.csv'),
				index = False,
				encoding = 'utf-8'				
			)
			vss_params_long.to_csv(
				os.path.join(outdir, 'VSS Parameters.csv'),
				index = False,
				encoding = 'utf-8'				
			)

		#Load COD Balance data to database(s)
		cut.write_to_db(vss_params_long,'cr2c-monitoring','valdata','vss_params', if_exists = if_exists_vss_params_table)
		cut.write_to_db(cod_bal_long,'cr2c-monitoring','valdata','cod_balance', if_exists = if_exists_cod_balance_table)

		return cod_bal_long, vss_params_long


	'''
	Verify pressure and ph sensor readings from op data and manometer readings from Google sheets.
	Calculate water head from pressure sensor readings, and compare it with the manometer readings
	'''
	def instr_val(
		self, 
		valtypes, op_sids, 
		start_dt_str = None, end_dt_str = None, 
		fld_varnames = None, 
		ltypes = None, lstages = None, 
		create_table = None,
		run_op_report = False, ip_path = None,
		output_csv = False,
		outdir = None
	):


		# Validation data are from field measurements (daily log sheet)
		if fld_varnames:

			query_varnames = ['Barometer Pressure (mmHg)']
			for varname in fld_varnames:
				# Sometimes the user needs to specify a PAIR of variables (eg pressure upstream AND downstream of pump)
				if type(varname) == tuple:
					query_varnames.append(varname[0])
					query_varnames.append(varname[1])
				# Otherwise just single variable name
				else:
					query_varnames.append(varname)

			# Clean the query variables
			query_varnames = [fld.clean_varname(varname) for varname in query_varnames]
			# Query the field data (using clean variable names)
			table_names = cut.get_table_names('fielddata')
			valdat = cut.get_data('fielddata', table_names)
			# Stack measurements from all tables
			valdat = cut.stack_tables(valdat, ['TIMESTAMP'] + query_varnames)

			# Create time variable with minute resolution from field data TIMESTAMP variable
			valdat['Time'] = pd.to_datetime(valdat['TIMESTAMP']).values.astype('datetime64[m]')
			# Replace missing barometric pressure readings with the mean psi at sea level
			valdat.loc[:,'BAROMETER_PRESSURE_MMHG'] = pd.to_numeric(valdat['BAROMETER_PRESSURE_MMHG'], errors = 'coerce')
			valdat.loc[np.isnan(valdat['BAROMETER_PRESSURE_MMHG']),'BAROMETER_PRESSURE_MMHG'] = 760
			
			# Loop through field variables to convert to numeric and calculate differences (if necessary)
			for varInd,varname in enumerate(fld_varnames):
				if type(varname) == tuple:
					valdat.loc[:,op_sids[varInd] + 'VAL'] = \
						pd.to_numeric(valdat[fld.clean_varname(varname[1])], errors = 'coerce') - \
						pd.to_numeric(valdat[fld.clean_varname(varname[0])], errors = 'coerce')
				else:
					valdat[op_sids[varInd] + 'VAL'] = pd.to_numeric(valdat[fld.clean_varname(varname)], errors = 'coerce')

			valdat = valdat[['Time','BAROMETER_PRESSURE_MMHG'] + [sid + 'VAL' for sid in op_sids]]
		
		# Validation data are from lab measurements
		elif ltypes or lstages:

			# Get data for unique set of ltypes requested
			valdat = cut.get_data('labdata', list(set(ltypes)))
			# Stack lab data
			valdat_long = cut.stack_tables(valdat)
			# valdat_long = pd.concat([cut.get_data('labdata',[ltype])[ltype] for ltype in ltypes], axis = 0, sort = True)
			valdat_long = valdat_long.loc[valdat_long['Stage'].isin(lstages),:]
			# Convert to wide format
			# Calculate mean by obsid to account for possibility of multiple PH measurements taken for single sample
			valdat_long = valdat_long.groupby(['Date_Time','Stage','Type','obs_id']).mean()
			valdat_wide = valdat_long.unstack(['Type','Stage'])
			valdat_wide.reset_index(inplace = True)
			# valdat = valdatWide['Date_Time']
			valdat = pd.DataFrame(valdat_wide['Date_Time'].values, columns = ['Time'])
			valdat_colnames = [op_sids[lind] + 'VAL' for lind,ltype in enumerate(ltypes)]
			for lind,ltype in enumerate(ltypes):
				valdat[valdat_colnames[lind]] = valdat_wide['Value'][ltype][lstages[lind]]
			valdat = valdat[['Time'] + valdat_colnames]


		# Expand valdat to get copies of each logged value for each of:
		# 10 minutes before and 10 minutes after it was entered into the google form
		valdatList = []
		for minDiff in range(-10,11):
			valdatDiff = valdat.copy()
			valdatDiff['Time']  = valdatDiff['Time'] + timedelta(seconds = minDiff*60)
			valdatList.append(valdatDiff)
		valdatAll = pd.concat(valdatList, axis = 0, sort = True)

		# Get op data for the element ids whose measurements are being validated
		nsids = len(op_sids)
		# Run op report if requested (minute level)
		if run_op_report:

			op_run = op.op_data_agg(start_dt_str, end_dt_str, ip_path = ip_path)
			op_run.run_agg(
				valtypes, # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
				op_sids, # Sensor ids that you want summary data for (have to be in op data file obviously)
				[1]*nsids, # Number of minutes you want to average over
				['MINUTE']*nsids, # Type of time period (can be "hour" or "minute")
			)

		# Retrieve data from SQL file
		table_names = ['{}_{}_1_MINUTE_AVERAGES'.format(stype, sid) for stype, sid in zip(valtypes, op_sids)]
		opdat = cut.get_data('opdata', table_names, start_dt_str = start_dt_str, end_dt_str = end_dt_str)
		# Merge tables
		opdat = cut.merge_tables(opdat, ['Time'], ['Value']*len(op_sids), merged_varnames = op_sids)

		# Merge the op data with the validation data
		valdatMerged = opdat.merge(valdatAll, on = 'Time', how = 'inner')
		# Merge all values on a day (since we are validating on a -10 to +10 minute window)
		valdatMerged.loc[:,'Date'] = valdatMerged['Time'].dt.date
		# Take average of time Window
		valdatMerged = valdatMerged.groupby('Date').mean()
		valdatMerged.reset_index(inplace = True)
		valdatMerged.loc[:,'Date'] = pd.to_datetime(valdatMerged['Date'])

		valdatStack = []
		# Loop through each instrument to compute error
		for sind, sid in enumerate(op_sids):

			if valtypes[sind] == 'PRESSURE':
				# Convert barometric pressure readings to psi
				valdatMerged.loc[:,'BAROMETER_PRESSURE_MMHG'] = valdatMerged['BAROMETER_PRESSURE_MMHG']*0.0193368
				# Convert pressure to inches of head
				valdatMerged.loc[:,sid] = (valdatMerged[sid] - valdatMerged['BAROMETER_PRESSURE_MMHG'])*27.7076

			# Compute the percentage error (op measurement vs validation)
			valdatMerged.loc[:,'Error'] = (valdatMerged[sid] - valdatMerged[sid + 'VAL'])
			valdatMerged.loc[:,'Percentage Error'] = (valdatMerged['Error'])/valdatMerged[sid + 'VAL']
			# Subset to the element of interest
			valdatSub = valdatMerged.loc[:,['Date', sid, sid + 'VAL', 'Error']]
			valdatSub.loc[:,'Sensor ID'] = sid
			valdatSub.columns = ['Date_Time','Sensor Value','Validated Measurement','Error','Sensor ID'] 
			valdatSub = valdatSub[['Date_Time','Sensor ID','Sensor Value','Validated Measurement','Error']]

			if output_csv:
				op_fname = '{}_validation.csv'.format(sid)
				valdatSub.to_csv(os.path.join(outdir, op_fname), index = False, encoding = 'utf-8')

			# Only continue if there are observations (sometimes there arent...)
			if valdatSub.size > 0:
				valdatStack.append(valdatSub)
		
		valdatStack = pd.concat(valdatStack, sort = True)
		instr_val_long = pd.melt(valdatStack, id_vars = ['Date_Time','Sensor ID'], value_vars = ['Sensor Value','Validated Measurement','Error'])

		instr_val_long.columns = ['Date_Time','Sensor_ID','Type','Value']

		# Create key unique by Date_Time, Stage, Type, and obs_id
		instr_val_long.loc[:,'Dkey'] = instr_val_long['Date_Time'].astype(str) + instr_val_long['Sensor_ID'] + instr_val_long['Type']
		# Reorder columns to put DKey as first column
		colnames = list(instr_val_long.columns.values)
		instr_val_long = instr_val_long[colnames[-1:] + colnames[0:-1]]

		if output_csv:
			instr_val_long.to_csv(
				os.path.join(outdir, 'Instrument Validation.csv'),
				index = False,
				encoding = 'utf-8'				
			)

		# Load COD Balance data to database(s)
		cut.write_to_db(instr_val_long,'cr2c-monitoring','valdata','instr_validation', create_mode = create_table)

		return instr_val_long

