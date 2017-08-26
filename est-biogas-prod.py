import math
import pandas as pd 
from pandas.tseries.offsets import MonthEnd
import datetime
from datetime import datetime as dt
from pandas import read_excel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import os


def adj_Hcp(Hcp_gas, deriv_gas, temp):
	return Hcp_gas*math.exp(deriv_gas*(1/(273 + temp) - (1/298)))

def get_biogas_prod(infBOD_ult, infSO4, temp, percCH4, percCO2, flowrate, precision):
	
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
	BODconv_CH4 = (infBOD_ult*perc_BODrem - BODrem_SO4)*fe 
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
			CH4_gas_vol    = CH4_gas_eq_mol*Vol_adj*flowrate*cubicft_p_L
			biogas_gas_vol = CH4_gas_vol/percCH4_biogas
		except ZeroDivisionError:
			CH4_gas_vol, biogas_gas_vol = 0,0
			break

	return [CH4_gas_vol, biogas_gas_vol]

def get_input_data(input_dir, input_filename, BODsheet, SO4sheet, tempsheet):

	# Read in data
	file_str = os.path.join(input_dir, input_filename)
	data_file = pd.ExcelFile(file_str)
	BODdata  = data_file.parse(BODsheet)
	SO4data  = data_file.parse(SO4sheet)
	tempdata = data_file.parse(tempsheet) 

	# Get clean BOD sample collection dates and measurement values
	BODdata['Sample datediff'] = BODdata['Sample Collect End'] - BODdata['Sample Collect Start']
	BODdata['Date'] = BODdata['Sample datediff']/2 + BODdata['Sample Collect Start']
	BODdata['BOD'] = BODdata['Measured Value']*BODdata['Dilution Factor']
	BODdata = BODdata[['Date','BOD']]
	BODdata['Date'] = BODdata.Date.dt.date
	BODdata.reset_index(inplace = True)
	# Get clean SO4 sample collection dates and measurement values
	SO4data['Sample datediff'] = SO4data['Sample Collect End'] - SO4data['Sample Collect Start']
	SO4data['Date'] = SO4data['Sample datediff']/2 + SO4data['Sample Collect Start']
	SO4data['Date'] = SO4data.Date.dt.date
	SO4data['SO4'] = SO4data['Measured Value']*SO4data['Dilution Factor']
	SO4data = SO4data[['Date','SO4']]
	SO4data.reset_index(inplace = True)
	# Get mean temperature and convert to C
	tempdata['temp'] = \
		(
			pd.to_numeric(tempdata['High (F)'], errors = 'coerce') + 
			pd.to_numeric(tempdata['Low (F)'], errors = 'coerce')
		)/2
	tempdata['temp'] = (tempdata['temp'] - 32)/1.8
	tempdata['Date'] = pd.to_datetime(tempdata['Date'])
	tempdata['Date'] = tempdata.Date.dt.date
	tempdata.reset_index(inplace = True)
	tempdata = tempdata[['Date','temp']]
	# Merge to get single array with Date, BOD, SO4 and temperature
	WWdata = pd.merge(BODdata, SO4data, on = 'Date', how = 'left')
	WWdata = pd.merge(WWdata, tempdata, on = 'Date', how = 'left')
	WWdata['Date'] = pd.to_datetime(WWdata['Date'])
	WWdata['month'] = WWdata['Date'].dt.month
	# Get monthly average BOD and SO4 values
	WWmonthly = WWdata.groupby('month').mean()
	WWmonthly.reset_index(inplace = True)
	# Merge back onto data
	WWdata = pd.merge(WWdata,WWmonthly, on = 'month', how = 'left')
	# Replace missing values with their monthly counterparts
	WWdata['BOD'] = WWdata['BOD_x']
	WWdata['SO4'] = WWdata['SO4_x']
	WWdata.loc[np.isnan(WWdata['BOD_x']),'BOD'] = WWdata['BOD_y']
	WWdata.loc[np.isnan(WWdata['SO4_x']),'SO4'] = WWdata['SO4_y']
	# Rename temperature variable
	WWdata['temp'] = WWdata['temp_x']
	WWdata = WWdata[['Date','BOD','SO4','temp','month']]

	return WWdata

WWdata = \
	get_input_data(
		'C:/Users/jbolorinos/Google Drive/Codiga Center/Charts and Data',
		'SulfateBOD_SVCWdata.xlsx',
		'BOD PE',
		'Sulfate PE',
		'Temperature 2015-2017'
	)

flowrate_m3d = 87.204
BGproddata = []
# Calculate biogas production for each day in the dataset
for WWrow in WWdata.values:
	BGprodrow = get_biogas_prod(WWrow[1], WWrow[2], WWrow[3], 0.65, 0.35, flowrate_m3d, 1E-6)
	BGproddata.append(BGprodrow)
# Convert results to dataframe
BGproddata = pd.DataFrame(BGproddata)
# Add columns from input dataset
WWdata[r'$CH_4$'] = BGproddata.loc[:,0]
WWdata['Biogas']  = BGproddata.loc[:,1]
# Eliminate missing values
WWdata.dropna(how = 'any', inplace = True)
# Use linear interpolation to estimate biogas production across all dates
WWdata.sort_values(['Date'], inplace = True)
WWdata.reset_index(inplace = True)
WWdata['del'] = (WWdata['Date'] - WWdata['Date'][0])/np.timedelta64(24,'h')
WWdata['del_next']   = WWdata['del'].shift(-1)
WWdata['del_prev']   = WWdata['del'].shift(1)
WWdata['ndays'] = (WWdata['del_next'] - WWdata['del'])
WWdata['CH4_next'] = WWdata[r'$CH_4$'].shift(-1)
WWdata['CH4_prev'] = WWdata[r'$CH_4$'].shift(1)
# Linearly interpolate between all values
WWdata['CH4_tot'] = \
	(WWdata['ndays'])*\
	(WWdata['CH4_next'] + WWdata[r'$CH_4$'])/2
# Add month cutoff to first and last days with data for a given month
WWdata['month_next']  = WWdata['month'].shift(-1)
WWdata['month_prev']  = WWdata['month'].shift(1)
WWdata['mstrt']       = WWdata['Date'].values.astype('datetime64[M]')
WWdata['mend']        = WWdata.Date  + MonthEnd(0)
WWdata.loc[WWdata.month_prev != WWdata.month,'CH4_tot'] = \
	(WWdata['CH4_prev'] - WWdata[r'$CH_4$'])/\
	(WWdata['del'] - WWdata['del_prev'])*\
	(WWdata['Date'] - WWdata['mstrt'])/np.timedelta64(24,'h')
WWdata.loc[WWdata.month_next != WWdata.month,'CH4_tot'] = \
	(WWdata['CH4_prev'] - WWdata[r'$CH_4$'])/\
	(WWdata['del_next'] - WWdata['del'])*\
	(WWdata['Date'] - WWdata['mend'])/np.timedelta64(24,'h')

WWsumst = WWdata.groupby('mstrt').sum()
WWsumst['Biogas_tot'] = WWsumst['CH4_tot']/0.65
WWsumst.reset_index(inplace = True)
WWsumst = WWsumst[['mstrt','CH4_tot','Biogas_tot']]


# Output to CSV
input_dir = 'C:/Users/jbolorinos/Google Drive/Codiga Center/Charts and Data/biogas_prod_monthly.csv'
WWsumst.to_csv(input_dir, index = False, encoding = 'utf-8')


# Produce some plots
os.chdir('C:/Users/jbolorinos/Google Drive/Codiga Center/Charts and Data')
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.hist(WWdata[r'$CH_4$'], bins = 100, fill = 'orange')
plt.xlabel('Gas production (' + r'$ft^3$' + ' per day)')
plt.ylabel('Frequency')
ax.xaxis.set_major_formatter(
	tkr.FuncFormatter(lambda x, p: format(int(x), ','))
)
plt.savefig(
	'Methane Production Histogram.png',
	width = 20, 
	height = 10
)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.plot(
	WWdata['Date'].values, WWdata[r'$CH_4$'], c = 'orange',
)
plt.plot(
	WWdata['Date'].values, WWdata['Biogas'], c = 'brown',
)
plt.ylabel('Gas production (' + r'$ft^3$' + ' per day)')
ax.yaxis.set_major_formatter(
	tkr.FuncFormatter(lambda x, p: format(int(x), ','))
)
plt.xticks(rotation = 45)
plt.legend()
plt.savefig(
	'Methane and Biogas Production Timeseries.png',
	width = 20, 
	height = 10
)
