# Creates SQL tables in the Data directory 
# NOTE: DO NOT RUN OR WILL OVERWRITE EXISTING TABLE!!!!!!!!

import pandas as pd
import os
import sqlite3

os.chdir('C:/Users/jbolorinos/Box Sync/CR2C.Operations/MonitoringProcedures/Data')

conn = sqlite3.connect('cr2c_lab_data.db')
curs = conn.cursor()

mk_COD_table_sql     = 'CREATE TABLE COD_data(Date TEXT, Stage TEXT, Type TEXT, obs_id INTEGER, Value REAL)'
mk_TSS_VSS_table_sql = 'CREATE TABLE TSS_VSS_data(Date TEXT, Stage TEXT, Type TEXT, obs_id INTEGER, Value REAL)'
mk_ALK_table_sql     = 'CREATE TABLE ALK_data(Date TEXT, Stage TEXT, obs_id INTEGER, Value REAL)'
mk_PH_table_sql      = 'CREATE TABLE PH_data(Date TEXT, Stage TEXT, obs_id INTEGER, Value REAL)'
mk_VFA_table_sql     = 'CREATE TABLE VFA_data(Date TEXT, Stage TEXT, Type TEXT, obs_id INTEGER, Value REAL)'

curs.execute(mk_COD_table_sql)
curs.execute(mk_TSS_VSS_table_sql)
curs.execute(mk_ALK_table_sql)
curs.execute(mk_PH_table_sql)
curs.execute(mk_VFA_table_sql)

conn.close()