
# Utilities
import os 
from datetime import datetime as dt
from datetime import timedelta

# CR2C
from dependencies import cr2c_labdata as pld
from dependencies import cr2c_opdata as op
from dependencies import cr2c_fielddata as fld
from dependencies import cr2c_validation as val
from dependencies import cr2c_utils as cut

#========================================> Arguments <=========================================
pydir = '/Volumes/GoogleDrive/Shared drives/CR2C.Box/Monitoring Data and Procedures/Python/GoogleProjectsAdmin'
hmi_start_dt_str = '7-2-19'
end_dt_str = '6-12-19'
hmi_path = '/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/LT100_2017_2019.csv'

#==========================> Update with latest lab and field data <===========================

# # Initialize lab data class
# cr2c_lr = pld.labrun() 
# # Lab Data
# cr2c_lr.process_data(pydir = pydir, if_exists = 'replace')
# outdir = '/Users/josebolorinos/Google Drive/Codiga Center/Charts and Data/Monitoring Reports/Monitoring Report {}'.format(end_dt_str)
# if not os.path.exists(outdir):
#     os.mkdir(outdir)
# cr2c_lr.summarize_tables(
#     end_dt_str, 
#     365, 
#     add_time_el = True, 
#     outdir = outdir
# )
# # Field Data
# fld.process_data(pydir = pydir, table_name = 'DailyLogResponsesV2')


# #===========================> Update with latest operational data <============================

# # Initialize HMI class
# op_run = op.opdata_agg(
#     hmi_start_dt_str, # Start of date range we want summary data for 
#     end_dt_str, # End of date range we want summary data for)
#     ip_path = hmi_path
# )

# Hourly averages
# op_run.run_agg(
#     ['COND']*4 + 
#     ['PH']*4 + 
#     ['WATER']*10 + 
#     ['GAS']*3 + 
#     ['TEMP']*3 + 
#     ['TMP']*1 +
#     ['DPI']*3 + 
#     ['PRESSURE']*3 + 
#     ['LEVEL']*5
#     ,
#     ['AT201','AT303','AT306','AT309'] + 
#     ['AT203','AT305','AT308','AT311'] + 
#     ['FT200','FT201','FT202','FT300','FT301','FT302','FT303','FT304','FT305','FIT600'] + 
#     ['FT700','FT702','FT704'] + 
#     ['AT202','AT304','AT310'] +
#     ['AIT302'] + 
#     ['DPIT300','DPIT301','DPIT302'] +
#     ['PIT205','PIT700','PIT702'] + 
#     ['LT100','LT200','LT201','LIT300','LIT301']
#     ,
#     [1]*36,
#     ['HOUR']*36
# )

# # Minute averages (for sensors we are validating and membrane parameters)
# op_run.run_agg(
#     ['water'] +
#     ['tmp'] +
#     ['ph']*2 + 
#     ['dpi']*2 +
#     ['pressure']*1
#     , # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
#     ['FT305'] +
#     ['AIT302'] +
#     ['AT203','AT305'] + 
#     ['DPIT300','DPIT301'] +
#     ['PIT700']
#     , # Sensor ids that you want summary data for (have to be in HMI data file obviously)    
#     [1]*7, # Number of hours/minutes we want to average over
#     ['minute']*7, # Type of time period (can be "hour" or "minute")
# )

#==============================> Update with latest validation data <==============================
    
# # Initialize validation class
# cr2c_vl = val.cr2c_validation()

# # Get COD balance and biotech parameters
# cr2c_vl.get_biotech_params(end_dt_str, 4)

# # Get validation plots
# cr2c_vl.instr_val(
#     valtypes = ['PH','PH','PH','PH'],
#     op_sids = ['AT203','AT305','AT308','AT311'],
#     ltypes = ['PH','PH','PH','PH'],
#     lstages = ['Microscreen','AFBR','Research AFMBR MLSS','Duty AFMBR MLSS'],
# )
# cr2c_vl.instr_val(
#     valtypes = ['DPI','DPI','DPI','PRESSURE','PRESSURE','PRESSURE'],
#     op_sids = ['DPIT300','DPIT301','DPIT302','PIT700','PIT702','PIT704'],
#     fld_varnames = [
#         ('Before Pump: R300','After Pump: R300'),
#         ('Before Pump: R301','After Pump: R301'),
#         ('Before Pump: R302','After Pump: R302'),
#         'Manometer Pressure: R300',
#         'Manometer Pressure: R301',
#         'Manometer Pressure: R302'
#     ]
# )


lt100data = cut.get_data('opdata',['LEVEL_LT100_1_HOUR_AVERAGES'])['LEVEL_LT100_1_HOUR_AVERAGES']
lt100data.to_csv('/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/LEVEL_LT100_1_HOUR_AVERAGES.csv')

import plotly.plotly as py
import plotly.graph_objs as go

# Create a trace
trace = go.Scatter(
    x = lt100data['Time'],
    y = lt100data['Value']
)
layout = dict(
    xaxis = dict(title = 'Time'),
    yaxis = dict(title = 'Av. LT100 Reading (in.)'),
)
data = [trace]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='lt100-plot')


