import os
import sqlite3
import get_lab_data as gld


def mk_sql_table(table_dir, elid, agg_type, tperiod):

	mk_sql_str =  """
	CREATE TABLE {0}_{1}s_{2}hour(
		Date_Time TEXT, Value REAL,
		unique(Date_Time, Value)
	)
	""".format(elid, agg_type, tperiod)

	# Open SQL connection in data directory
	os.chdir(data_indir)
	conn = sqlite3.connect('cr2c_hmi_agg_data.db')
	curs = conn.cursor()

	# Execute the SQL command to create the table
	curs.execute(mk_sql_str)
	conn.close()

if __name__ == '__main__':

	# Get directory with lab data
	data_indir = gld.get_indir()
	elids = ['FT202','FT305','FT700','FT704']
	agg_types = ['total','total','average','average']
	tperiods = [1]*4

	for elid, agg_type, tperiod in zip(elids, agg_types, tperiods):

		mk_sql_table(
			data_indir, 
			elid, 
			agg_type, 
			tperiod
		)
