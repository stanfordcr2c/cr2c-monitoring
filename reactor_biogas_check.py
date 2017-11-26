	from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import seaborn as sns
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import warnings
import os
import math
import sys
import get_gsheet_data as gsd
import hmi_data_agg as hmi


def verify_reactor_pressure(pressure_path = None, hmi_path = None):

	# Get field pressure measurements
	#            AFBR  RAFMBR DAFMBR 
	reactors = ['R300','R301','R302']
	logvals_sheet = gsd.get_gsheet_data(['DailyLogResponses'])
	logvals_list = logvals_sheet[0]['values']
	headers = ['TimeStamp'] + logvals_list.pop(0)[1:]
	logvals_df = pd.DataFrame(logvals_list, columns = headers)
	pdat_log = logvals_df[['TimeStamp'] + ['Manometer Pressure: ' + reactor for reactor in reactors]]
	pdat_log['TimeStamp'] = pd.to_datetime(pdat_log['TimeStamp'])
	pdat_log['TS_mins'] = pdat_log['TimeStamp'].values.astype('datetime64[m]')

	# First subset hmi data to dates for which field measurements are available
	first_lts = pdat_log['TimeStamp'][0]
	last_lts = pdat_log['TimeStamp'][len(pdat_log) - 1]
	first_lts_str = dt.strftime(first_lts, format = '%m-%d-%y')
	last_lts_str = dt.strftime(last_lts, format = '%m-%d-%y')

	# Get HMI pressure data
	#             AFBR    RAFMBR   DAFMBR 
	pr_elids = ['PIT700','PIT702','PIT704']
	if pressure_path:
		pdat_hmi = pd.read_csv(pressure_path)
		pdat_hmi['Time'] = pd.to_datetime(pdat_hmi['Time'])
	else:
		hmi_pr = hmi.hmi_data_agg('raw','pressure')
		pdat_hmi = hmi_pr.run_report(
			1/60,
			pr_elids,
			['average','average','average'],
			first_lts_str,
			last_lts_str,
			hmi_path = hmi_path,
			output_csv = 1
		)
	# Convert pressure readings to inches of head (comparable to field measurements)
	for elid in pr_elids:
		pdat_hmi[elid + 'Gauge Pr. (in)'] = (pdat_hmi[elid + '_AVERAGE'] - 14.7)*27.7076

	# Merge the two datasets only hmi data observations in the field measurement data (minute timescale here)
	merged_pr = pdat_hmi.merge(pdat_log, how = 'right', left_on = 'Time', right_on = 'TS_mins')
	merged_pr.reset_index(inplace = True)
	# print(merged_pr['Manometer Pressure: R300'])
	# fig, ax = plt.subplots()
	# plt.plot(merged_pr['Time'], merged_pr['PIT700Gauge Pr. (in)'])
	# plt.plot(merged_pr['TS_mins'], merged_pr['Manometer Pressure: R300'])
	# plt.show()

verify_reactor_pressure(
	# hmi_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/Reactor Pressure_9-29-17_10-11-17.csv'
	pressure_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIPRESSURE_PIT700_PIT702_PIT704_09-29-17_10-11-17.csv'
)
