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

'''
Verify pressure sensor readings from HMI data and manometer readings from Google sheets.
Calculate water head from pressure sensor readings, and compare it with the manometer readings.
Plot the merged data to show results.
'''

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
        # Initialize hmi_data_agg instance
        hmi_pr = hmi.hmi_data_agg('09-29-17', '11-22-17', hmi_path)

        # Set up time and style variables
        tperiods = [1, 1, 1]
        ttypes = ['HOUR','HOUR','HOUR']
        stypes = ['PRESSURE', 'PRESSURE', 'PRESSURE']

        # Write the report to SQL database and then get data from it
        hmi_pr.run_report(tperiods,ttypes,pr_elids,stypes, output_csv = True)
        pdat_hmi = hmi_pr.get_data(pr_elids, tperiods, ttypes, 2017)

    for tperiod, ttype, pr_elid in zip(tperiods, ttypes, pr_elids):
        # Create keys of pressure sensor with specified time period. e.g. 'PIT700_1HOUR_AVERAGES'
        pr_elid_hmi = pr_elid + '_' + str(tperiod) + ttype + '_AVERAGES'
        # Create columns of gauge pressure for the sensor
        pr_head = pr_elid + ' Gauge Pr. (in)'

        # Convert pressure readings to inches of head (comparable to field measurements)
        pdat_hmi[pr_elid_hmi][pr_head] = pdat_hmi[pr_elid_hmi]['Value'].apply(lambda x: (x - 14.7) * 27.7076)

        # Merge the two datasets only hmi data observations in the field measurement data (minute timescale here)
        pr_head_hmi = pdat_hmi[pr_elid_hmi][['Time', pr_head]]
        if 'merged_pr' not in locals():
            merged_pr = pd.merge_asof(pdat_log, pr_head_hmi, left_on = 'TS_mins', right_on = 'Time')
        else:
            merged_pr = pd.merge_asof(merged_pr, pr_head_hmi, left_on = 'TS_mins', right_on = 'Time')

        # Delete additional Time column
        merged_pr = merged_pr.drop('Time', 1)

    # Plot manometer pressures vs HMI sensor gauge pressure
    nrows = 3
    fig = plt.figure()
    fig, axes = plt.subplots(nrows, sharex = True)
    fig.set_size_inches(8, 20)
    ids_hmi = ['PIT700', 'PIT702', 'PIT704']
    ids_gsheet = ['R300', 'R301', 'R302']
    for ax_idx, (id_hmi, id_gsheet) in enumerate(zip(ids_hmi, ids_gsheet)):
        axes[ax_idx].plot(merged_pr['TS_mins'], merged_pr[id_hmi + ' Gauge Pr. (in)'])
        axes[ax_idx].plot(merged_pr['TS_mins'], merged_pr['Manometer Pressure: ' + id_gsheet])

        # axes[ax_idx].yaxis.set_major_locator(tkr.AutoLocator())
        axes[ax_idx].legend()

    # Display only months and days on the x axis
    date_fmt = dates.DateFormatter('%m/%d')
    axes[ax_idx].xaxis.set_major_formatter(date_fmt)
    plt.show()

verify_reactor_pressure(
    hmi_path = '/Users/joannalin/Box Sync/CR2C.Operations/MonitoringProcedures/Data/HMIPRESSURE_PIT700_PIT702_PIT704_9-29-17_11-22-17.csv',
    # pressure_path = '/Users/joannalin/Box Sync/CR2C.Operations/MonitoringProcedures/Data/HMIPRESSURE_PIT700_PIT702_PIT704_9-29-17_11-22-17.csv'
)
