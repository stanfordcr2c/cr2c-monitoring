'''
	Creates SQL tables in the Data directory 
	(NOTE: will fail if a table already exists in the directory)
'''

import pandas as pd
import os
import sqlite3
import get_lab_data as gld


# Get directory with lab data
data_indir = gld.get_indir()

# Open SQL connection in data directory
os.chdir(data_indir)
conn = sqlite3.connect('cr2c_lab_data.db')
curs = conn.cursor()

# SQL commands for creating tables
mk_COD_table_sql     = """
	CREATE TABLE COD_data(
		Date_Time TEXT, Stage TEXT, Type TEXT, obs_id TEXT, Value REAL,
		unique(Date_Time, Stage, Type, obs_id, Value)
	)
"""
mk_TSS_VSS_table_sql = """
	CREATE TABLE TSS_VSS_data(
		id INTEGER PRIMARY KEY, Date_Time TEXT, Stage TEXT, Type TEXT, obs_id TEXT, Value REAL,
		unique(Date_Time, Stage, Type, obs_id, Value)
	)
"""
mk_ALK_table_sql     = """
	CREATE TABLE ALK_data(
		id INTEGER PRIMARY KEY, Date_Time TEXT, Stage TEXT, obs_id TEXT, Value REAL,
		unique(Date_Time, Stage, obs_id, Value)
	)
"""
mk_PH_table_sql      = """
	CREATE TABLE PH_data(
		id INTEGER PRIMARY KEY, Date_Time TEXT, Stage TEXT, obs_id TEXT, Value REAL,
		unique(Date_Time, Stage, obs_id, Value)
	)
"""
mk_VFA_table_sql     = """
	CREATE TABLE VFA_data(
		id INTEGER PRIMARY KEY, Date_Time TEXT, Stage TEXT, Type TEXT, obs_id TEXT, Value REAL,
		unique(Date_Time, Stage, Type, obs_id, Value)
	)
"""
mk_GasComp_table_sql     = """
	CREATE TABLE GasComp_data(
		id INTEGER PRIMARY KEY, Date_Time TEXT, Hel_Pressure REAL, Type TEXT, obs_id TEXT, Value REAL,
		unique(Date_Time, Hel_Pressure, Type, obs_id, Value)
	)
"""

# Execute SQL commands
# curs.execute(mk_COD_table_sql)
# curs.execute(mk_TSS_VSS_table_sql)
# curs.execute(mk_ALK_table_sql)
# curs.execute(mk_PH_table_sql)
# curs.execute(mk_VFA_table_sql)
curs.execute(mk_GasComp_table_sql)

# Close connection
conn.close()

