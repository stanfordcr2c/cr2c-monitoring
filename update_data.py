
# Utilities
import os 
from datetime import datetime as dt
from datetime import timedelta

# CR2C
import cr2c_labdata as pld
import cr2c_opdata as op
import cr2c_fielddata as fld
import cr2c_validation as val

#========================================> Arguments <=========================================
pydir = '/Volumes/GoogleDrive/Team Drives/CR2C.Box/Monitoring Data and Procedures/Python'
hmi_start_dt_str = '2-19-19'
end_dt_str = '2-24-19'
hmi_path = '/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Most Plant Parameters_20190226101525.csv'

#==========================> Update with latest lab and field data <===========================

# Initialize lab data class
# cr2c_lr = pld.labrun() 
# Lab Data
# cr2c_lr.process_data(pydir = pydir)
# Field Data
fld.process_data(pydir = pydir, tableName = 'DailyLogResponsesV2')

#===========================> Update with latest operational data <============================

# Initialize HMI class
op_run = op.opdata_agg(
    hmi_start_dt_str, # Start of date range we want summary data for 
    end_dt_str, # End of date range we want summary data for)
    ip_path = hmi_path
)

# # Hourly averages
# op_run.run_agg(
#     ['COND']*4 + 
#     ['PH']*4 + 
#     ['WATER']*10 + 
#     ['GAS']*2 + 
#     ['TEMP']*3 + 
#     ['TMP']*1 +
#     ['DPI']*3 + 
#     ['PRESSURE']*3 + 
#     ['LEVEL']*5
#     ,
#     ['AT201','AT303','AT306','AT309'] + 
#     ['AT203','AT305','AT308','AT311'] + 
#     ['FT200','FT201','FT202','FT300','FT301','FT302','FT303','FT304','FT305','FIT600'] + 
#     ['FT700','FT704'] + 
#     ['AT202','AT304','AT310'] +
#     ['AIT302'] + 
#     ['DPIT300','DPIT301','DPIT302'] +
#     ['PIT205','PIT700','PIT702','PIT704'] + 
#     ['LT200','LT201','LIT300','LIT301','LIT302']
#     ,
#     [1]*35,
#     ['HOUR']*35
# )

# Minute averages (for sensors we are validating and membrane parameters)
# op_run.run_agg(
#     ['water'] +
#     ['tmp'] +
#     ['ph']*2 + 
#     ['dpi']*2 +
#     ['pressure']*2
#     , # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
#     ['FT305'] +
#     ['AIT302'] +
#     ['AT203','AT305'] + 
#     ['DPIT300','DPIT301'] +
#     ['PIT700','PIT704']
#     , # Sensor ids that you want summary data for (have to be in HMI data file obviously)    
#     [1]*8, # Number of hours/minutes we want to average over
#     ['minute']*8, # Type of time period (can be "hour" or "minute")
# )

#==============================> Get operational tables and plots<==============================
    
# # Initialize validation class
cr2c_vl = val.cr2c_validation()

# # Get COD balance and biotech parameters
cr2c_vl.get_biotech_params(end_dt_str, 8)

# Get validation plots
cr2c_vl.instr_val(
    valtypes = ['PH','PH'],
    op_sids = ['AT203','AT305'],
    ltypes = ['PH','PH'],
    lstages = ['Microscreen','AFBR'],
    output_csv = True, 
    outdir = '/Users/josebolorinos/Google Drive/Coursework Stuff/CS230/CS230 Final Project/Data'
)
cr2c_vl.instr_val(
    valtypes = ['DPI','DPI','PRESSURE','PRESSURE'],
    op_sids = ['DPIT300','DPIT301','PIT700','PIT704'],
    fld_varnames = [('Before Pump: R300','After Pump: R300'),('Before Pump: R301','After Pump: R301'),'Manometer Pressure: R300','Manometer Pressure: R301']
)

