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


start_dt_str = '9-17-17'
hmi_start_dt_str = '11-10-17'
end_dt_str = '11-17-17'

# ==========================> Lab Data <===========================

# cr2c_lr = pld.labrun()
# cr2c_lr.process_data()

# # Create and output charts
# cr2c_lr.get_lab_plots(
# 	# Plot start date
# 	start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['PH','ALKALINITY','Ammonia'], 
# 	# Variable to break down into panels according to
# 	'Stage',
# 	# Stages to Subset to
# 	['Microscreen','AFBR','Duty AFMBR Effluent','Duty AFMBR MLSS']
# )
# cr2c_lr.get_lab_plots(
# 	# Plot start date
# 	start_dt_str,
# 	# Plot end date
# 	end_dt_str,
# 	# List of monitoring data types to produce charts for (correspond to tabs on gsheets workbook)
# 	['COD','TSS','VFA'], 
# 	# Variable to break down into panels according to
# 	'Type',
# 	# Stages to Subset to
# 	['Microscreen','AFBR','Duty AFMBR Effluent','Duty AFMBR MLSS']
# )

# # Get wide tables
# cr2c_lr.summarize_tables(end_dt_str, 182, add_time_el = True)

# ==========================> Lab Data <===========================

# ==========================> HMI Data <===========================

hmi_run = hmi.hmi_data_agg(
	'9-27-17', # Start of date range you want summary data for
	'9-28-17'# End of date range you want summary data for)
)
# hmi_run.run_report(
# 	[1,1,1,1,5,1,5], # Number of hours you want to average over
# 	['hour','hour','hour','hour','minute','hour','minute'], # Type of time period (can be "hour" or "minute")
# 	['FT700','FT704','FT202','FT305','FT305','AIT302','AIT302'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)
# 	['gas','gas','water','water','water','tmp','tmp'] # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
# )
hmi_run.get_tmp_plots(
	start_dt_str,
	end_dt_str
)
hmi_run.get_feed_sumst(
	'gas',
	['plot','table'],
	start_dt_str,
	end_dt_str,
	sum_period = 'DAY', 
	plt_type = 'bar', 
	plt_colors = ['#90775a','#eeae10'],
	ylabel = 'Biogas Production (L/day)'
)
hmi_run.get_feed_sumst(
	'water',
	['plot','table'],
	start_dt_str,
	end_dt_str,
	sum_period = 'DAY', 
	plt_type = 'bar', 
	plt_colors = ['#8c9c81','#7fbfff'],
	ylabel = 'Reactor Feeding (Gal/Day)'
)

# ==========================> HMI Data <===========================
