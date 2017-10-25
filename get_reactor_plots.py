from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import seaborn as sns
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import warnings
import os
import math
import sys
import hmi_data_agg as hmi
import cat_dfs as cat

def get_tmp_plots(
	start_dt_str,
	end_dt_str,
	detailed = False,
	outdir = None,
	hmi_path = None, 
	feeding_paths = None, 
	tmp_paths = None
):

	start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
	end_dt = dt.strptime(end_dt_str,'%m-%d-%y')


	if not outdir:
		outdir = askdirectory(title = 'Directory to output charts/tables to')

	if feeding_paths:

		feeding_dat	= cat.cat_dfs(
			feeding_paths, 
			idx_var = 'Time', 
			output = True, 
			output_dsn = 'HMITMP_{0}_{1}_{2}.csv'.format('AIT302', start_dt_str, end_dt_str)
		)
		feeding_dat['Time'] = pd.to_datetime(feeding_dat['Time'])
	else:
		hmi_wtr = hmi.hmi_data_agg('raw','water')
		feeding_dat = hmi_wtr.run_report(
			1/12,
			['FT305'],
			['average'],
			start_dt_str,
			end_dt_str,
			hmi_path = hmi_path,
			output_csv = 1
		)

	if tmp_paths:
		tmp_dat = cat.cat_dfs(
			tmp_paths, 
			idx_var = 'Time', 
			output = True,
			output_dsn = 'HMIWATER_{0}_{1}_{2}.csv'.format('FT305', start_dt_str, end_dt_str)

		)
		tmp_dat['Time'] = pd.to_datetime(tmp_dat['Time'])
	else:
		hmi_tmp = hmi.hmi_data_agg('raw','tmp')
		tmp_dat = hmi_tmp.run_report(
			1/12,
			['AIT302'],
			['average'],
			start_dt_str,
			end_dt_str,
			hmi_path = hmi_path,
			output_csv = 1
		)

	tmp_feed_dat = feeding_dat.merge(tmp_dat, on = 'Time')
	# Remove index!
	tmp_feed_dat.reset_index(inplace = True)

	# Group the dataset into days and weeks
	tmp_feed_dat['Week'] = tmp_feed_dat['Time'].dt.week
	# tmp_feed_dat['Week'] = 
	tmp_feed_dat['tel_day'] = tmp_feed_dat['Time'].dt.hour*60 + tmp_feed_dat['Time'].dt.minute

	tmp_feed_dat['Day']  = tmp_feed_dat['Time'].dt.weekday
	tmp_feed_dat['Hour'] = tmp_feed_dat['Time'].dt.hour + tmp_feed_dat['Time'].dt.weekday*24

	# Get data for last week and hour
	tmp_feed_week = tmp_feed_dat.loc[
		tmp_feed_dat['Time'].dt.date -  tmp_feed_dat['Time'].dt.date[len(tmp_feed_dat) - 1] >= \
		np.timedelta64(-6,'D'),
	]
	tmp_feed_day = tmp_feed_dat.loc[
		tmp_feed_dat['Time'].dt.date -  tmp_feed_dat['Time'].dt.date[len(tmp_feed_dat) - 1] > \
		np.timedelta64(-1,'D'),
	]

	# Plot!
	sns.set_style('white')
	# Last two months (or entire date range)
	# TMP
	ax1 = plt.subplot2grid((16,1),(0,0), rowspan = 2)
	ax1.plot(tmp_feed_dat['Time'],tmp_feed_dat['AIT302_AVERAGE'], 'g-', linewidth = 0.5)
	ax1.set_title(
		'Hourly Average TMP and Permeate Flux ({0} to {1})'.format(start_dt_str, end_dt_str),
		fontweight = 'bold'
	)
	ax1.set_ylabel('TMP (psia)')
	ax1.xaxis.set_ticklabels([])
	# Flow
	ax2 = plt.subplot2grid((16,1),(2,0), rowspan = 2)
	ax2.plot(tmp_feed_dat['Time'],tmp_feed_dat['FT305_AVERAGE'], 'b-', linewidth = 0.5)
	ax2.set_ylabel('Flow (gpm)')
	labels = ax2.get_xticklabels()
	plt.setp(labels, rotation=45, fontsize=10)
	# Last week
	# TMP
	ax3 = plt.subplot2grid((16,1),(6,0), rowspan = 2)
	ax3.plot(tmp_feed_week['Time'],tmp_feed_week['AIT302_AVERAGE'], 'g-', linewidth = 0.5)
	ax3.set_title(
		'Hourly Average TMP and Permeate Flux (last 7 days)',
		fontweight = 'bold'
	)
	ax3.set_ylabel('TMP (psia)')
	ax3.xaxis.set_ticklabels([])
	# Flow
	ax4 = plt.subplot2grid((16,1),(8,0), rowspan = 2)
	ax4.plot(tmp_feed_week['Time'],tmp_feed_week['FT305_AVERAGE'], 'b-', linewidth = 0.5)
	ax4.set_ylabel('Flow (gpm)')
	labels = ax4.get_xticklabels()
	plt.setp(labels, rotation=45, fontsize=10)
	# Last day
	# TMP
	ax5 = plt.subplot2grid((16,1),(12,0), rowspan = 2)
	ax5.plot(tmp_feed_day['Time'],tmp_feed_day['AIT302_AVERAGE'], 'g-', linewidth = 0.5)
	ax5.set_title(
		'Hourly Average TMP and Permeate Flux (last 24 hours)',
		fontweight = 'bold'
	)
	ax5.set_ylabel('TMP (psia)')
	ax5.xaxis.set_ticklabels([])
	# Flow
	ax6 = plt.subplot2grid((16,1),(14,0), rowspan = 2)
	ax6.plot(tmp_feed_day['Time'],tmp_feed_day['FT305_AVERAGE'], 'b-', linewidth = 0.5)
	ax6.set_ylabel('Flow (gpm)')
	labels = ax6.get_xticklabels()
	plt.setp(labels, rotation=45, fontsize=10)

	# Output plots and/or sumstats csv files to directory of choice
	plot_filename  = "FLOW_TMP_{0}_{1}.png".format(start_dt_str, end_dt_str)
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(7, 12)
	plt.savefig(
		os.path.join(outdir, plot_filename), 
		width = 20, 
		height = 160
	)
	

	# Plot hourly tmp values at finer time scale
	if detailed:
		sns.set_style('white')
		tplot = sns.FacetGrid(
			tmp_feed_dat,
			col = 'Week',
			hue = 'Day',
			sharey = True,
			col_wrap = 3,
			legend_out = True
		)
		tplot.map(plt.plot,'tel_day','AIT302_AVERAGE', linestyle = '-', linewidth = 0.5)
		tplot.set_xlabels('Minutes Elapsed (since 0:00)')
		tplot.set_ylabels('Average TMP (psia)')
		tplot.add_legend()

		# Output!
		plot_filename  = "TMP_DETAILED_{0}_{1}.png".format(start_dt_str, end_dt_str)
		plt.savefig(
			os.path.join(outdir, plot_filename), 
			width = 20, 
			height = 160
		)


def get_feed_sumst(
	stype,
	output_types,
	start_dt_str,
	end_dt_str,
	outdir = None,
	hmi_path = None,
	feeding_paths = None,
	sum_period = 'DAY', 
	plt_type = None, 
	plt_colors = None,
	ylabel = None,
	get_nhours = None
):
	

	start_dt = dt.strptime(start_dt_str,'%m-%d-%y')
	end_dt = dt.strptime(end_dt_str,'%m-%d-%y')

	# Clean case of input arguments
	sum_period = sum_period.upper()
	plt_type = plt_type.upper()
	if type(output_types) == list:
		output_types = [output_type.upper() for output_type in output_types]
	else:
		output_types = output_types.upper()

	# Define HMI element ids according to query type (water or biogas)
	stype = stype.upper()
	if stype == 'GAS':
		elids = ['FT700','FT704']
	if stype == 'WATER':
		elids = ['FT202','FT305']

	# Get output directory and string with all element ids from report
	if not outdir:
		outdir = askdirectory(title = 'Directory to output charts/tables to')

	if feeding_paths:
		feeding_dat	= cat.cat_dfs(
			feeding_paths, 
			idx_var = 'Time'
		)
		feeding_dat['Time'] = pd.to_datetime(feeding_dat['Time'])
	else:
		hmi_agg = hmi.hmi_data_agg('raw',stype)
		feeding_dat = hmi_agg.run_report(
			24,
			elids,
			['total','total'],
			start_dt_str,
			end_dt_str,
			hmi_path = hmi_path,
			output_csv = 1
		)

	# Retrieve element ids from aggregated data
	all_elids = '_'.join(elids)

	# Convert Time variable to pd.datetime variable
	feeding_dat['Time'] = pd.to_datetime(feeding_dat['Time'])
	feeding_dat['Date'] = feeding_dat['Time'].dt.date

	# Filter to the dates desired for the plots
	feeding_dat = feeding_dat.loc[
		(feeding_dat.Time >= start_dt) &
		(feeding_dat.Time <= end_dt)
	]

	# Get dataset aggregated by Day, Week or Month
	if sum_period == 'HOUR':
		xlabel = 'Time'
	else:
		feeding_dat['Date'] = feeding_dat['Time'].dt.date

	if sum_period == 'DAY':
		xlabel = 'Date'

	if sum_period == 'WEEK':
		xlabel = 'Weeks (since {0})'.format(start_dt_str)
		feeding_dat[xlabel] = np.floor((feeding_dat['Time'] - start_dt)/np.timedelta64(7,'D'))
	
	if sum_period == 'MONTH':
		xlabel = 'Months (since {0}, as 30 days)'.format(start_dt_str)
		feeding_dat[xlabel] = np.floor((feeding_dat['Time'] - start_dt)/np.timedelta64(30,'D'))

	if get_nhours == 1:
		for elid in elids:
			feeding_dat['Number Hours {0}'.format(elid)] = \
				np.where(feeding_dat[elid].values > 0, 1, 0)

	agg_sumst = feeding_dat.groupby(xlabel).sum()

	# Plot!
	if 'PLOT' in output_types:

		# Set the maximum number of tick labels
		nobs  = len(agg_sumst.index.values)
		nlims = nobs
		if sum_period == 'DAY':
			nlims = 12
		# Get the indices of the x-axis values according to these tick labels
		lim_len  = int(np.floor(nobs/nlims))
		tic_idxs = [lim*lim_len for lim in range(nlims)]
		tic_vals = [agg_sumst.index.values[tic_idx] for tic_idx in tic_idxs]
		
		if sum_period != 'DAY':
			tic_vals = ['{0} - {1}'.format(int(tic_val), int(tic_val + 1)) for tic_val in tic_vals]

		if plt_type == 'BAR':
			varnames = [elid + '_TOTAL' for elid in elids]
			ax = agg_sumst[varnames].plot.bar(stacked = False, width = 0.8, color = plt_colors)
			plt.xticks(tic_idxs,tic_vals)
		else:
			ax = agg_sumst[varnames].plot(color = plt_colors)

		plt.ylabel(ylabel)
		plt.legend()

		ax.yaxis.set_major_formatter(
			tkr.FuncFormatter(lambda y, p: format(int(y), ','))
		)
		
		plt.xticks(rotation = 45)
		plt.tight_layout()

		# Output plots and/or sumstats csv files to directory of choice
		plot_filename  = "HMI{0}_{1}_{2}_{3}.png".format(stype, all_elids, start_dt_str, end_dt_str)
		plt.savefig(
			os.path.join(outdir, plot_filename), 
			width = 20, 
			height = 50
		)

	if 'TABLE' in output_types:

		sumst_filename = "HMI{0}_{1}_{2}_{3}.csv".format(stype, all_elids, start_dt_str, end_dt_str)
		agg_sumst.reset_index(inplace = True)
		agg_sumst.to_csv(
			os.path.join(outdir, sumst_filename), 
			index = False,
			encoding = 'utf-8'
		)


# get_tmp_plots(
# 	'8-21-17',
# 	'10-21-17',
# 	outdir = 'C:/Users/jbolorinos/Google Drive/Codiga Center/Charts and Data/Monitoring Reports/10-21-17',
# 	detailed = 1,
# 	hmi_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/Reactor Feeding - Raw_20171022073204.csv',
# 	feeding_paths = [
# 		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIWATER0.08_FT305_08-19-17_10-01-17.csv',
# 		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIWATER0.08_FT305_10-01-17_10-13-17.csv',
# 		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIWATER0.08_FT305_10-13-17_10-21-17.csv'
# 	],
# 	tmp_paths = [
# 		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMITMP0.08_AIT302_08-13-17_10-13-17.csv',
# 		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMITMP0.08_AIT302_10-10-17_10-21-17.csv'
# 	]
# )


get_feed_sumst(
	'gas',
	['plot','table'],
	'8-21-17',
	'10-21-17',
	outdir = 'C:/Users/jbolorinos/Google Drive/Codiga Center/Charts and Data/Monitoring Reports/10-21-17',
	# hmi_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/Reactor Feeding - Raw_20171022073204.csv',
	feeding_paths = [
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIGAS_FT700_FT704_10-01-17_10-13-17.csv',
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIGAS_FT700_FT704_08-13-17_10-01-17.csv',
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIGAS1_FT700_FT704_10-10-17_10-21-17.csv'
	],
	sum_period = 'DAY', 
	plt_type = 'bar', 
	plt_colors = ['#90775a','#eeae10'],
	ylabel = 'Biogas Production (L/day)'
)
get_feed_sumst(
	'water',
	['plot','table'],
	'8-21-17',
	'10-21-17',
	outdir = 'C:/Users/jbolorinos/Google Drive/Codiga Center/Charts and Data/Monitoring Reports/10-21-17',
	hmi_path = 'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/Reactor Feeding - Raw_20171022073204.csv',
	feeding_paths = [
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIWATER_FT202_FT305_10-01-17_10-13-17.csv',
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIWATER_FT202_FT305_08-13-17_10-01-17.csv',
		'C:/Users/jbolorinos/Google Drive/Codiga Center/HMI Data/HMIWATER24_FT202_FT305_10-13-17_10-21-17.csv'
	],
	sum_period = 'DAY', 
	plt_type = 'bar', 
	plt_colors = ['#8c9c81','#7fbfff'],
	ylabel = 'Reactor Feeding (Gal/Day)'
)
