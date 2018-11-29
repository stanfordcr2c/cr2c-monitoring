'''
	Creates SQL tables in the Data directory 
	(NOTE: will fail if a table already exists in the directory)
'''

import os
import sqlite3
import get_lab_data as gld


def mk_sql_table(table_dir, mtype):

	if mtype in ['COD','TSS','VSS','TSS_VSS','VFA']:

		mk_sql_str = """
			CREATE TABLE {}_data(
				id INTEGER PRIMARY KEY Date_Time TEXT, Stage TEXT, Type TEXT, obs_id TEXT, Value REAL,
				unique(Date_Time, Stage, Type, obs_id, Value)
			)
		""".format(mtype)

	elif mtype in ['PH','ALKALINITY']:

		mk_sql_str = """
			CREATE TABLE {}_data(
				id INTEGER PRIMARY KEY, Date_Time TEXT, Stage TEXT, obs_id TEXT, Value REAL,
				unique(Date_Time, Stage, obs_id, Value)
			)
		""".format(mtype)

	elif mtype == 'GasComp':

		mk_sql_str = """
			CREATE TABLE GasComp_data(
				id INTEGER PRIMARY KEY, Date_Time TEXT, Hel_Pressure REAL, Type TEXT, obs_id TEXT, Value REAL,
				unique(Date_Time, Hel_Pressure, Type, obs_id, Value)
			)
		"""

	# Open SQL connection in data directory
	os.chdir(data_indir)
	conn = sqlite3.connect('cr2c_lab_data.db')
	curs = conn.cursor()

	# Execute the SQL command to create the table
	curs.execute(mk_sql_str)
	# Close connection
	conn.close()


if __name__ == '__main__':

	# Get directory with lab data
	table_dir = gld.get_indir()

	mtypes = ['COD','VFA','TSS_VSS','PH','ALKALINITY','GasComp']
	
	for mtype in mtypes:
		mk_sql_table(table_dir, mtype)

