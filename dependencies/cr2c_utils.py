# Data Prep
import pandas as pd

# Utilities
import os
from os.path import expanduser
import sys
import json

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
def get_credentials(local = True, pydir = None):

	SCOPES = 'https://www.googleapis.com/auth/spreadsheets.readonly'
	CLIENT_SECRET_FILE = 'client_secret.json'
	projectid = 'cr2c-monitoring'

	if local:

		pydir = get_dirs()[1]
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

		os.chdir(os.path.join(pydir,'GoogleProjectsAdmin'))
		spreadsheetId = open('spreadsheetId.txt').read()
		
	else:

		credentials = compute_engine.credentials()
		dataset_id = 'labdata'
		spreadsheetId = pd.read_gbq('SELECT * FROM {}.{}'.format(dataset_id, 'google_spreadsheetId'), projectid)['spreadsheetId'].values[0]

	if not credentials or credentials.invalid:	
		flags = 'An unknown error occurred'
		flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
		flow.user_agent = projectid
		credentials = tools.run_flow(flow, store, flags)

	return credentials, spreadsheetId


# Retrieves data of specified tabs in a gsheets file
def get_gsheet_data(sheet_name, local = True):

	credentials, spreadsheetId = get_credentials(local)
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

def get_table_names(dataset_id, local = True, data_dir = None):

	if local:

		# Create connection to SQL database
		os.chdir(data_dir)
		conn = sqlite3.connect('{}.db'.format(dataset_id))
		cursor = conn.cursor()
		# Execute
		cursor.execute(""" SELECT name FROM sqlite_master WHERE type ='table'""")
		table_names = [names[0] for names in cursor.fetchall()]

	else:

		client = bigquery.Client(credentials = get_credentials(local = local))
		tables = list(client.list_tables(dataset_ref))
		table_names = [table.table_id for table in tables]

	return table_names



