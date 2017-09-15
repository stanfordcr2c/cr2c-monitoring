import os
from os.path import expanduser
import sys
import sqlite3
from tkinter.filedialog import askdirectory

# Manages output directories
def get_dirs():
	
	# Find the CR2C.Operations folder on Box Sync on the given machine
	targetdir = os.path.join('Box Sync','CR2C.Operations')
	mondir = None
	print("Searching for Codiga Center's Operations folder on Box Sync...")
	for dirpath, dirname, filename in os.walk(expanduser('~')):
		if dirpath.find(targetdir) > 0:
			mondir = os.path.join(dirpath,'MonitoringProcedures')
			print("Found Codiga Center's Operations folder on Box Sync")
			break
			
	# Alert user if Box Sync folder not found on machine
	if mondir == None:
		print("Could not find Codiga Center's Operations folder in Box Sync.")
		print('Please make sure that Box Sync is installed and the Operations folder is synced on your machine')
		sys.exit()
	
	return os.path.join(self.mondir,'Data')


def get_ldata(mtype):

	data_indir = get_dirs

	# Load data from SQL
	os.chdir(self.data_indir)
	conn = sqlite3.connect('cr2c_lab_data.db')
	# Clean user input wrt TSS_VSS
	if mtype.find('TSS') >= 0 or mtype.find('VSS') >= 0:
		mtype = 'TSS_VSS'

	mdata_long = pd.read_sql(
		'SELECT * FROM {}_data'.format(mtype), 
		conn, 
		coerce_float = True
	)

	return mdata_long