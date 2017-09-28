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

def get_biogas_prod(BODrem, infSO4, temp, percCH4, percCO2, flowrate, precision):
	
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

