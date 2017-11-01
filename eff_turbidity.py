'''
sys_runtime.py combines AIT302 TMP and FT305 flow rate data from HMI, AFMBR Effluent 
Turbidity data from CR2CMonitoringData Google sheet. 

Outputs:
1. A csv file of the combined data showing whether system is running or not, and 
whether the effluent is clear or not at each sampling timestamp. 
2. The time series plot of run time and effluent turbidity at those timestamps.
'''

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from get_gsheet_data import get_gsheet_data
from matplotlib import pyplot as plt, dates as mdates

# find the path of a file located in Boxsync on local drive. Return targetdir if data_name = ''
def find_data_path(data_name):
	targetdir = 'Box Sync/CR2C.Operations/MonitoringProcedures/Data'
	for dirpath, dirname, filename in os.walk(os.path.expanduser('~')):
		if dirpath.find(targetdir) > 0:
			data_path = os.path.join(dirpath, data_name)
			return data_path
	
# obtain data on the minute resolution as Pandas Dataframe, and rename its column
def get_hmi_data(db_name, table, colname):
	db_path = find_data_path(db_name)
	conn = sqlite3.connect(db_path)
	cur = conn.cursor()
	sql = ("select * from '{}'").format(table)
	cur.execute(sql)
	results = cur.fetchall()
	df = pd.read_sql_query(sql, conn)

	# set df index to datetimeindex
	df = df.set_index(pd.DatetimeIndex(df['Time']))
	del df['Time']
	df = df.rename(columns = {'Value': colname})
	return df

# compute the system runtime at each timestamp. If the system is running, runtime is positive;
# if the system is idling, the runtime is negative 
def get_sys_runtime():
	df = get_hmi_data('cr2c_hmi_agg_data_2017.db', 'FT305_TOTALs_0.016666666666666666hour', 'Flowrate')
	# set values less than 0.5 as 0
	idle_threshold = 0.5
	run_times = np.zeros(len(df.index))
	run_time = 0

	# sum the hours that the system was running at each timestamp
	i = 0
	run_time = 0
	length = len(df.index)
	while i < length:
		# if the system is running, runtime is positive
		while i < length and df['Flowrate'][i] > idle_threshold:
			run_time += 1
			run_times[i] = run_time/60 # unit is hour
			i += 1
		run_time = 0
	    # if the system is idling, runtime is negative
		while i < length and df['Flowrate'][i] <= idle_threshold:
		    run_time -= 1
		    run_times[i] = run_time/60
		    i += 1
		run_time = 0
	# reindex df by datetime
	df.loc[:,'System RunTime'] = pd.Series(run_times, index=df.index)

	return df

# get effluent turbidity data from Google sheet as dataframe, 
def get_eff_turbidity():
	gsheet_values = get_gsheet_data(['AFMBREffluentTurbidity'])
	# convert data to dataframe
	df = pd.DataFrame(columns = gsheet_values[0]['values'][0])
	for i in range(len(gsheet_values[0]['values']) - 1):
		df.loc[i] = gsheet_values[0]['values'][i+1]
	df = df.set_index(pd.DatetimeIndex(df['Timestamp']))
	del df['Timestamp']
	return df

# plot and save the results combined data of hmi and survey 
def plot_result():
	runtime = get_sys_runtime()
	eff = get_eff_turbidity()
	tmp = get_hmi_data('cr2c_hmi_agg_data_2017.db', 'AIT302_TOTALs_0.016666666666666666hour', 'TMP')

	# join runtime and tmp dataframes based on nearest time to the effluent dataframe, save as csv
	result = eff.join(runtime.reindex(eff.index.unique(), method = 'nearest')).\
			join(tmp.reindex(eff.index.unique(), method = 'nearest'), lsuffix='_survey', rsuffix='_hmi')
	data_dir = find_data_path('')
	result.to_csv(os.path.join(data_dir, 'System_Runtime.csv'))

	# plot continuous TMP data as lines
	fig, ax = plt.subplots(figsize=(15, 5))
	tmp = ax.plot(tmp.index, tmp.TMP, label='TMP from HMI')

	# plot clear effluent at blue points, and non-clear effluent as red points
	ax2 = ax.twinx()
	clear = result[result['Effluent is clear?'] == 'Yes']
	not_clear = result[result['Effluent is clear?'] == 'No']
	clear = ax.scatter(clear.index, clear['System RunTime'], color='b', s=55, label='Clear')
	not_clear = ax.scatter(not_clear.index, not_clear['System RunTime'], color='r', label='Not Clear')

	# set up xy labels, ticks and figure legend
	ax.set_xlabel('Date')
	ax.set_ylabel('Historical Transmembrane Pressure (psi)', labelpad=45)
	ax.yaxis.set_label_position('right')
	ax2.set_ylabel('System Run Time (hr)', labelpad=25)
	ax2.set_ylim(-5, max(result['System RunTime']))
	ax2.yaxis.set_label_position('left')
	dateFmt = mdates.DateFormatter('%m/%d')
	ax.xaxis.set_major_formatter(dateFmt)
	ax.legend()

	# autoscale x and y axes to show a tight plot
	ax2.autoscale(enable=True, tight='True')
	plt.show()
	fig.savefig(os.path.join(data_dir,'run_time_vs_eff_turbidity.png'))

if __name__ == '__main__':
	plot_result()

