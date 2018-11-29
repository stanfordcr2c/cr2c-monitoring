
# CR2C
import cr2c_opdata as op
import cr2c_labdata as ld
import cr2c_validation as val
import os
import sqlite3


opagg = op.opdata_agg(start_dt_str = '7-1-18', end_dt_str = '10-30-18', ip_path = '/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Emir_RawData.csv')


opagg.run_agg(
 ['WATER']*4,
 ['FT206','FT406']*2,
 [1]*4,
 ['HOUR']*2 + ['MINUTE']*4,
 output_csv = True,
 outdir = '/Users/josebolorinos/Google Drive/Codiga Center/Miscellany'
)



# op.get_data(
# 	['COND']*4 + 
# 	['PH']*4 + 
# 	['WATER']*10 + 
# 	['GAS']*2 + 
# 	['TEMP']*3 + 
# 	['TMP']*1 +
# 	['DPI']*2 + 
# 	['PRESSURE']*4 + 
# 	['LEVEL']*5
# 	,
# 	['AT201','AT303','AT306','AT309'] + 
# 	['AT203','AT305','AT308','AT311'] + 
# 	['FT200','FT201','FT202','FT300','FT301','FT302','FT303','FT304','FT305','FIT600'] + 
# 	['FT700','FT704'] + 
# 	['AT202','AT304','AT310'] +
# 	['AIT302'] + 
# 	['DPIT300','DPIT301'] + 
# 	['PIT205','PIT700','PIT702','PIT704'] + 
# 	['LT200','LT201','LIT300','LIT301','LIT302']
# 	,
# 	[1]*35,
# 	['HOUR']*35,
# 	combine_all = False,
# 	output_csv = True,
# 	outdir = '/Users/josebolorinos/Box Sync/CR2C.Operations/Monitoring Data and Procedures/Data/SQL/CSV'
# )

# op.get_data(
#     ['water','tmp','ph','ph','dpi','dpi','pressure','pressure'], # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
#     ['FT305','AIT302','AT203','AT305','DPIT300','DPIT301','PIT700','PIT704'], # Sensor ids that you want summary data for (have to be in HMI data file obviously)    
#     [1]*8, # Number of hours/minutes we want to average over
#     ['minute']*8, # Type of time period (can be "hour" or "minute")
#     combine_all = False,
#     output_csv = True,
#     outdir = '/Users/josebolorinos/Box Sync/CR2C.Operations/Monitoring Data and Procedures/Data/SQL/CSV'
# )


# cr2c_vl = val.cr2c_validation()
# cr2c_vl.instr_val(
#     valtypes = ['PH','PH'],
#     op_sids = ['AT203','AT305'],
#     ltypes = ['PH','PH'],
#     lstages = ['Microscreen','AFBR'],
#     output_csv = True,
#     outdir = '/Users/josebolorinos/Google Drive/Coursework Stuff/CS230/CS230 Final Project/Data'

# )
# cr2c_vl.instr_val(
#     valtypes = ['DPI','DPI','PRESSURE','PRESSURE'],
#     op_sids = ['DPIT300','DPIT301','PIT700','PIT704'],
#     fld_varnames = [('Before Pump: R300','After Pump: R300'),('Before Pump: R301','After Pump: R301'),'Manometer Pressure: R300','Manometer Pressure: R301'],
#     output_csv = True,
#     outdir = '/Users/josebolorinos/Google Drive/Coursework Stuff/CS230/CS230 Final Project/Data'
# )




