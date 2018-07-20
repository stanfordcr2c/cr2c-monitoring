import os
from os.path import expanduser
import sys
import pandas as pd

import httplib2
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

try:
    import argparse
    # flags = argparse.ArgumentParser(parents = [tools.argparser]).parse_args()
except ImportError:
    flags = None


# Manages output directories
def get_dirs():
	
	# Find the CR2C.Operations folder on Box Sync on the given machine
	targetdir = os.path.join('Box Sync','CR2C.Operations')
	mondir = None
	for dirpath, dirname, filename in os.walk(expanduser('~')):
		if dirpath.find(targetdir) > 0:
			mondir = os.path.join(dirpath,'Monitoring Data and Procedures')
			break
			
	# Alert user if Box Sync folder not found on machine
	if not mondir:
		if os.path.isdir('D:/'):
			for dirpath, dirname, filename in os.walk('D:/'):
				if dirpath.find(targetdir) > 0:
					mondir = os.path.join(dirpath,'Monitoring Data and Procedures')
					break
		if not mondir:
			print("Could not find Codiga Center's Operations folder in Box Sync")
			print('Please make sure that Box Sync is installed and the Operations folder is synced on your machine')
			sys.exit()

	pydir = os.path.join(mondir, 'Python')
	data_dir = os.path.join(mondir,os.path.join('Data','SQL'))

	return data_dir, pydir


# Gets valid user credentials from storage.
# If nothing has been stored, or if the stored credentials are invalid,
# the OAuth2 flow is completed to obtain the new credentials.
def get_credentials():

	pydir = get_dirs()[1]

	SCOPES = 'https://www.googleapis.com/auth/spreadsheets.readonly'
	CLIENT_SECRET_FILE = 'client_secret.json'
	APPLICATION_NAME = 'cr2c-monitoring'

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

	if not credentials or credentials.invalid:	
		flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
		flow.user_agent = APPLICATION_NAME
		credentials = tools.run_flow(flow, store, flags)

	return credentials, spreadsheetId


# Retrieves all data from a gsheets file given list of sheet names
def get_gsheet_data(sheet_names):

	credentials, spreadsheetId = get_credentials()
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
	range_names = [sheet_name + '!A:ZZ' for sheet_name in sheet_names]
	gsheet_result = service.spreadsheets().values().batchGet(
		spreadsheetId = spreadsheetId, 
		ranges = range_names
	).execute()	

	# Get values list from dictionary and extact headers (first item in list)
	gsheet_values = gsheet_result['valueRanges'][0]['values']
	headers = gsheet_values.pop(0) 
	# Output a pandas dataframe
	dframe = pd.DataFrame(gsheet_values, columns = headers)

	return dframe

