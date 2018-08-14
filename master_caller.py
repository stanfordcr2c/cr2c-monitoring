from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import seaborn as sns
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import warnings
import os 
import math
import sys
import cr2c_labdata as pld
import cr2c_hmidata as hmi
import cr2c_fielddata as fld
import cr2c_validation as val


# outdir_root = '/Users/josebolorinos/Google Drive/Codiga Center/Charts and Data/Monitoring Reports'
# hmi_path = '/Users/josebolorinos/Google Drive/Codiga Center/HMI Data/Reactor Feeding - Raw_20180720025935.csv'
# start_dt_str = '5-19-18'
# hmi_start_dt_str = '7-11-18'
# end_dt_str = '7-19-18'
# # Get start date looking 1,6 months (30,180 days) back
# mo6_start_dt = dt.strptime(end_dt_str,'%m-%d-%y') - timedelta(days = 180)
# mo6_start_dt_str = dt.strftime(mo6_start_dt,'%m-%d-%y')
# mo1_start_dt = dt.strptime(end_dt_str,'%m-%d-%y') - timedelta(days = 30)
# mo1_start_dt_str = dt.strftime(mo1_start_dt,'%m-%d-%y')
# # Make directory for report!
# outdir = os.path.join(outdir_root,'Monitoring Report ' + end_dt_str)
# if not os.path.exists(outdir): 
#     os.mkdir(outdir)

# #==========================> Lab Data <===========================
# cr2c_lr = pld.labrun() 
# # cr2c_lr.process_data()
# # pld.get_data(['PH','COD','TSS_VSS','ALKALINITY','VFA','GasComp','Ammonia','Sulfate','TKN','BOD'], output_csv = True)


# labOutdir = os.path.join(outdir,'Lab Data')
# if not os.path.exists(labOutdir): 
# 	os.mkdir(labOutdir)

# # Create and output charts
# pld.get_lab_plots(
# 	# Plot start date
# 	start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['PH','ALKALINITY'], 
# 	# Variable to break down into panels according to
# 	'Stage',
# 	# Stages to Subset to
# 	['Microscreen','AFBR','Duty AFMBR Effluent','Duty AFMBR MLSS'],
# 	outdir = labOutdir
# )
# pld.get_lab_plots(
# 	# Plot start date
# 	start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['COD','TSS'], 
# 	# Variable to break down into panels according to
# 	'Type',
# 	# Stages to Subset to
# 	stage_sub = ['AFBR','Duty AFMBR MLSS'],
# 	type_sub = ['TSS','VSS','Total','Soluble','Particulate'],
# 	outdir = labOutdir,
# 	opfile_suff = 'AFBR_DAFMBRMLSS'
# )
# pld.get_lab_plots(
# 	# Plot start date
# 	start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['COD','TSS'], 
# 	# Variable to break down into panels according to
# 	'Stage',
# 	# Stages to Subset to
# 	stage_sub = ['Microscreen','Duty AFMBR Effluent'],
# 	type_sub = ['TSS','VSS','Total','Soluble','Particulate'],
# 	outdir = labOutdir,
# 	opfile_suff = 'MS_DAFMBREFF'
# )
# pld.get_lab_plots(
# 	# Plot start date
# 	start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['BOD'], 
# 	# Variable to break down into panels according to
# 	'Stage',
# 	# Stages to Subset to
# 	['Microscreen','Duty AFMBR Effluent'],
# 	outdir = labOutdir, 
# 	opfile_suff = 'MS_DAFMBREFF'
# )
# pld.get_lab_plots(
# 	# Plot start date
# 	mo6_start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['Ammonia','TKN'], 
# 	# Variable to break down into panels according to
# 	'Stage',
# 	# Stages to Subset to
# 	['Microscreen','Duty AFMBR Effluent'],
# 	outdir = labOutdir,
# 	opfile_suff = 'MS_DAFMBREFF'
# )
# pld.get_lab_plots(
# 	# Plot start date
# 	mo6_start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['VFA'], 
# 	# Variable to break down into panels according to
# 	'Type',
# 	# Stages to Subset to
# 	outdir = labOutdir
# )
# # Get wide tables
# cr2c_lr.summarize_tables(end_dt_str, 240, add_time_el = True, outdir = labOutdir)
	
# #==========================> Lab Data <===========================

# #==========================> HMI Data <===========================

# HMIOutdir = os.path.join(outdir,'Operational Data')
# if not os.path.exists(HMIOutdir): 
# 	os.mkdir(HMIOutdir)

# hmi_run = hmi.hmi_data_agg(
# 	hmi_start_dt_str, # Start of date range you want summary data for 
# 	end_dt_str, # End of date range you want summary data for)
# 	hmi_path = hmi_path
# )
# hmi_run.run_report(
# 	[1,1,1,1,5,1,5,1,1], # Number of hours/minutes you want to average over
# 	['hour','hour','hour','hour','minute','hour','minute','hour','hour'], # Type of time period (can be "hour" or "minute")
# 	['FT700','FT704','FT202','FT305','FT305','AIT302','AIT302','AT304','AT310'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
# 	['gas','gas','water','water','water','tmp','tmp','temp','temp'] # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
# )
# hmi_run.get_tmp_plots(
# 	start_dt_str,
# 	end_dt_str,
# 	outdir = HMIOutdir
# )
# hmi_run.get_temp_plots(
# 	end_dt_str, 
# 	plt_colors =['c','m'],
# 	outdir = HMIOutdir
# )
# hmi_run.get_feed_sumst(
# 	'gas',
# 	['plot','table'],
# 	start_dt_str,
# 	end_dt_str,
# 	sum_period = 'DAY',
# 	plt_type = 'bar', 
# 	plt_colors = ['#90775a','#eeae10'],
# 	ylabel = 'Biogas Production (L/day)',
# 	outdir = HMIOutdir
# )
# hmi_run.get_feed_sumst(
# 	'water',
# 	['plot','table'],
# 	start_dt_str,
# 	end_dt_str,
# 	sum_period = 'DAY',  
# 	plt_type = 'bar', 
# 	plt_colors = ['#8c9c81','#7fbfff'],
# 	ylabel = 'Reactor Feeding (Gal/Day)',
# 	outdir = HMIOutdir
# )

# # ==========================> HMI Data <===========================

# # =========================> Field Data <==========================
# fld.process_data(tableName = 'DailyLogResponsesV2')
# fld.get_data(varNames = ['AFMBR_Volume_Wasted_Gal'], output_csv = True, outdir = HMIOutdir)
# # # =========================> Field Data <==========================

# # # =========================> VALIDATION <==========================

# valOutdir = os.path.join(outdir,'Validation')
# if not os.path.exists(valOutdir): 
# 	os.mkdir(valOutdir)

# cr2c_vl = val.cr2c_validation(outdir = valOutdir, hmi_path = hmi_path)

# cr2c_vl.get_cod_bal(end_dt_str, 8, plot = True)
# cr2c_vl.get_biotech_params(end_dt_str, 8, plot = True)

# # Run HMI aggregation scripts on instruments we are going to validate
# hmi_run.run_report(
# 	[1,1,1,1,1,1], # Number of hours/minutes you want to average over
# 	['minute','minute','minute','minute','minute','minute'], # Type of time period (can be "hour" or "minute")
# 	['AT203','AT305','DPIT300','DPIT301','PIT700','PIT704'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
# 	['PH','PH','DPI','DPI','PRESSURE','PRESSURE'] # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
# )

# cr2c_vl.instr_val(
# 	valtypes = ['PH','PH'],
# 	start_dt_str = mo1_start_dt_str,
# 	end_dt_str = end_dt_str,
# 	hmi_elids = ['AT203','AT305'],
# 	ltypes = ['PH','PH'],	
# 	lstages = ['Microscreen','AFBR']
# )
# cr2c_vl.instr_val(
# 	valtypes = ['DPI','DPI','PRESSURE','PRESSURE'],
# 	start_dt_str = mo1_start_dt_str,
# 	end_dt_str = end_dt_str,
# 	hmi_elids = ['DPIT300','DPIT301','PIT700','PIT704'],
# 	fld_varnames = [('Before Pump: R300','After Pump: R300'),('Before Pump: R301','After Pump: R301'),'Manometer Pressure: R300','Manometer Pressure: R301']
# )

# # =========================> VALIDATION <==========================


