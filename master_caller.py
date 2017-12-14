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
import process_lab_data as pld
import hmi_data_agg as hmi

outdir_root = '/Users/josebolorinos/Google Drive/Codiga Center/Charts and Data/Monitoring Reports'
start_dt_str = '10-9-17'
hmi_start_dt_str = '12-3-17'
end_dt_str = '12-10-17'

# Make directory for report!

outdir = os.path.join(outdir_root,'Monitoring Report' + end_dt_str)
if not os.path.exists(outdir):
    os.mkdir(outdir)

# ==========================> Lab Data <===========================

cr2c_lr = pld.labrun()
# cr2c_lr.process_data()

# Create and output charts
cr2c_lr.get_lab_plots(
	# Plot start date
	start_dt_str,
	# Plot end date
	end_dt_str,
	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
	['PH','ALKALINITY'], 
	# Variable to break down into panels according to
	'Stage',
	# Stages to Subset to
	['Microscreen','AFBR','Duty AFMBR Effluent','Duty AFMBR MLSS'],
	outdir = outdir
)
cr2c_lr.get_lab_plots(
	# Plot start date
	start_dt_str,
	# Plot end date
	end_dt_str,
	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
	['COD','TSS','VFA'], 
	# Variable to break down into panels according to
	'Type',
	# Stages to Subset to
	['AFBR','Duty AFMBR MLSS'],
	outdir = outdir,
	opfile_suff = 'AFBR_DAFMBRMLSS'
)
cr2c_lr.get_lab_plots(
	# Plot start date
	start_dt_str,
	# Plot end date
	end_dt_str,
	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
	['COD','TSS','VFA'], 
	# Variable to break down into panels according to
	'Type',
	# Stages to Subset to
	['Microscreen','Duty AFMBR Effluent'],
	outdir = outdir,
	opfile_suff = 'MS_DAFMBREFF'
)
# Get wide tables
cr2c_lr.summarize_tables(end_dt_str, 196, add_time_el = True, outdir = outdir)

# ==========================> Lab Data <===========================

# ==========================> HMI Data <===========================

hmi_run = hmi.hmi_data_agg(
	hmi_start_dt_str, # Start of date range you want summary data for 
	end_dt_str # End of date range you want summary data for)
)
hmi_run.run_report(
	[1,1,1,1,5,1,5], # Number of hours you want to average over
	['hour','hour','hour','hour','minute','hour','minute'], # Type of time period (can be "hour" or "minute")
	['FT700','FT704','FT202','FT305','FT305','AIT302','AIT302'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
	['gas','gas','water','water','water','tmp','tmp'] # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
)
hmi_run.get_tmp_plots(
	start_dt_str,
	end_dt_str,
	outdir = outdir
)
hmi_run.get_feed_sumst(
	'gas',
	['plot','table'],
	start_dt_str,
	end_dt_str,
	sum_period = 'DAY', 
	plt_type = 'bar', 
	plt_colors = ['#90775a','#eeae10'],
	ylabel = 'Biogas Production (L/day)',
	outdir = outdir
)
hmi_run.get_feed_sumst(
	'water',
	['plot','table'],
	start_dt_str,
	end_dt_str,
	sum_period = 'DAY', 
	plt_type = 'bar', 
	plt_colors = ['#8c9c81','#7fbfff'],
	ylabel = 'Reactor Feeding (Gal/Day)',
	outdir = outdir
)

# ==========================> HMI Data <===========================


