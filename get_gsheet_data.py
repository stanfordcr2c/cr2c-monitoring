import httplib2
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
import os
from os.path import expanduser

try:
	import argparse
	flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
	flags = None

# Manages output directories
def get_pydir():
	
	# Find the CR2C.Operations folder on Box Sync on the given machine
	targetdir = os.path.join('Box Sync','CR2C.Operations')
	mondir = None
	print("Searching for Codiga Center's Operations folder on Box Sync...")
	for dirpath, dirname, filename in os.walk(os.path.expanduser('~')):
		if dirpath.find(targetdir) > 0:
			mondir = os.path.join(dirpath,'MonitoringProcedures')
			print("Found Codiga Center's Operations folder on Box Sync")
			break
			
	# Alert user if Box Sync folder not found on machine
	if not mondir:
		if os.path.isdir('D:/'):
			for dirpath, dirname, filename in os.walk('D:/'):
				if dirpath.find(targetdir) > 0:
					mondir = os.path.join(dirpath,'MonitoringProcedures')
					print("Found Codiga Center's Operations folder on Box Sync")
					break
		if not mondir:
			print("Could not find Codiga Center's Operations folder in Box Sync")
			print('Please make sure that Box Sync is installed and the Operations folder is synced on your machine')
			sys.exit()
	
	return os.path.join(mondir, 'Python')


# Gets valid user credentials from storage.
# If nothing has been stored, or if the stored credentials are invalid,
# the OAuth2 flow is completed to obtain the new credentials.
def get_credentials():

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

	pydir = get_pydir()
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

	gsheet_values = gsheet_result['valueRanges']

	return gsheet_values
