# Data Prep
import pandas as pd

# Utilities
import os
from os.path import expanduser
import sys
import json
from datetime import datetime as dt
from datetime import timedelta
import functools

# Google sheets API and dependencies
import httplib2
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from google.auth import compute_engine
from google.cloud import bigquery


# Gets valid user credentials from storage.
# If nothing has been stored, or if the stored credentials are invalid,
# the OAuth2 flow is completed to obtain the new credentials.
def get_credentials(pydir):

	SCOPES = 'https://www.googleapis.com/auth/spreadsheets.readonly'
	CLIENT_SECRET_FILE = 'client_secret.json'
	projectid = 'cr2c-monitoring'

	home_dir = os.path.expanduser('~')
	credential_dir = os.path.join(home_dir, '.credentials')

	if not os.path.exists(credential_dir):
		os.makedirs(credential_dir)
	credential_path = os.path.join(
		credential_dir,
		'sheets.googleapis.com-cr2c-monitoring.json'
	)
	store = Storage(credential_path)
	credentials = store.get()

	spreadsheetId_path = os.path.join(pydir,'spreadsheetId.txt')
	spreadsheetId = open(spreadsheetId_path).read()
		
	if not credentials or credentials.invalid:	
		flags = 'An unknown error occurred'
		flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
		flow.user_agent = projectid
		credentials = tools.run_flow(flow, store, flags)

	return credentials, spreadsheetId


# Retrieves data of specified tabs in a gsheets file
def get_gsheet_data(sheet_name, pydir):

	credentials, spreadsheetId = get_credentials(pydir)
	http = credentials.authorize(httplib2.Http())
	discoveryUrl = (
		'https://sheets.googleapis.com/$discovery/rest?'
		'version=v4'
	)
	service = discovery.build(
		'sheets', 
		'v4', 
		http = http,
		discoveryServiceUrl = discoveryUrl
	)
	range_name = [sheet_name + '!A:ZZ']
	gsheet_result = service.spreadsheets().values().batchGet(
		spreadsheetId = spreadsheetId, 
		ranges = range_name
	).execute()	

	# Get values list from dictionary and extact headers (first item in list)
	gsheet_values = gsheet_result['valueRanges'][0]['values']
	headers = gsheet_values.pop(0) 
	# Output a pandas dataframe
	df = pd.DataFrame(gsheet_values, columns = headers)

	return df


def get_table_names(dataset_id, local = False, data_dir = None):

	if local:

		# Create connection to SQL database
		conn = sqlite3.connect(os.path.join(data_dir,'{}.db'.format(dataset_id)))
		cursor = conn.cursor()
		# Execute
		cursor.execute(""" SELECT name FROM sqlite_master WHERE type ='table'""")
		table_names = [names[0] for names in cursor.fetchall()]

	else:

		gbq_str = """SELECT table_id FROM {}.__TABLES_SUMMARY__""".format(dataset_id)
		table_names_list = pd.read_gbq(gbq_str, 'cr2c-monitoring').values
		table_names = [table_name[0] for table_name in table_names_list]

	return table_names


def get_data(
	dataset_id, table_names, 
	varnames = None, 
	start_dt_str = None, end_dt_str = None, 
	local = False, local_dir = None, 
	output_csv = False, outdir = None
):
	
	# Convert date string inputs to dt variables
	if start_dt_str:
		start_dt = dt.strptime(start_dt_str, '%m-%d-%y')
	if end_dt_str:
		end_dt = dt.strptime(end_dt_str, '%m-%d-%y')

	# Different time variable names (this needs to be streamlined...)
	if dataset_id == 'opdata':
		time_var = key = 'Time'
	elif dataset_id == 'fielddata':
		time_var = key = 'TIMESTAMP'
	else:
		time_var = 'Date_Time'
		key = 'Dkey'

	all_data = {}
	for table_name in table_names:

		if varnames:
			varnames_db = time_var + ',' + ','.join(varnames)
			if dataset_id == 'fielddata':
				varnames_db = varnames_db.upper()
		else:
			varnames_db = '*'

		df = pd.read_gbq('SELECT {} FROM {}.{}'.format(varnames_db, dataset_id, table_name), 'cr2c-monitoring')
		# Remove duplicate records
		df.drop_duplicates(inplace = True)
		# Create datetime variable and remove missing timestamp values
		df.loc[:,time_var] = pd.to_datetime(df[time_var])
		df.dropna(subset = [time_var], inplace = True)

		if not df.empty:
			# Sort by Dkey
			df.sort_values([key], inplace = True)

		# Subset to dates of interest
		if start_dt_str:
			df = df.loc[df[time_var] >= start_dt,:]
		if end_dt_str:
			df = df.loc[df[time_var] < end_dt + timedelta(days = 1),:]

		# Output csv if desired 
		if output_csv:
			op_dsn = 'cr2c_{}_{}.csv'.format(dataset_id, table_name)
			df.to_csv(os.path.join(outdir, op_dsn), index = False, encoding = 'utf-8')

		all_data[table_name] = df			

	return all_data


def write_to_db(
	df, projectid, dataset_id, table_name, 
	if_exists = 'append',
	local = False, local_dir = None
):

	# Different time variable names (this needs to be streamlined...)
	if dataset_id == 'opdata':
		time_var = key = 'Time'
	elif dataset_id == 'fielddata':
		time_var = key = 'TIMESTAMP'
	else:
		time_var = 'Date_Time'
		key = 'Dkey'

	# If creating table (eg if necessary to modify gbq table schema)
	if if_exists == 'replace':
		df_new = df.copy()
	else:
		df_already = get_data(dataset_id, [table_name])[table_name]
		df_new = df.loc[~df[key].isin(df_already[key]),:]

	# Remove duplicates and missing values, sort
	df_new.dropna(subset = [time_var], inplace = True)
	df_new.drop_duplicates(inplace = True)
	df_new.sort_values([key], inplace = True)

	# Write to gbq table
	if not df_new.empty:
		df_new.to_gbq('{}.{}'.format(dataset_id, table_name), projectid, if_exists = if_exists)


# Function for merging tables in a dictionary by a set of id variables
def merge_tables(df_dict, id_vars, measure_varnames, merged_varnames = None):

	# Set merged_varnames to measure_vars if not specified (simplifies loop below!)
	if not merged_varnames:
		merged_varnames = measure_varnames

	# Change measure variable names according to user spec
	for df, measure_varname, merged_varname in zip(df_dict.values(), measure_varnames, merged_varnames):
		df.rename(columns = {measure_varname: merged_varname}, inplace = True)
		df.set_index(id_vars, inplace = True)

	df_merged = pd.concat(
		[df[merged_varname] for df, merged_varname in zip(df_dict.values(), merged_varnames)], 
		axis = 1, 
		ignore_index = False,
		sort = True
	) 
	df_merged.reset_index(inplace = True)

	return df_merged


# Function for stacking tables in a dictionary (subset to desired stacking variables)
def stack_tables(df_dict, keep_vars = None):

	df_stacked = pd.concat(
		[df for df in list(df_dict.values())], 
		axis = 0, 
		ignore_index = True,
		sort = True
	) 
	df_stacked.reset_index(inplace = True)
	df_stacked.to_csv(os.path.join('/Users/josebolorinos/Google Drive/Codiga Center/debugging','df_stacked.csv'))
	if keep_vars:
		df_stacked = df_stacked.loc[:,keep_vars]

	return df_stacked


# Takes a list of file paths and concatenates all of the files
def cat_dfs(ip_paths, idx_var = None, output_csv = False, outdir = None, out_dsn = None):
	
	concat_dlist = []
	for ip_path in ip_paths:
		concat_dlist.append(pd.read_csv(ip_path, low_memory = False))
	concat_data = pd.concat([df for df in concat_dlist], ignore_index = True, sort = True)
	# Remove duplicates (may be some overlap)
	concat_data.drop_duplicates(keep = 'first', inplace = True)

	if output:

		concat_data.to_csv(
			os.path.join(outdir, out_dsn), 
			index = False, 
			encoding = 'utf-8'
		)
	
	return concat_data
	




