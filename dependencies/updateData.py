# Utilities
import os 
from datetime import datetime as dt
from datetime import timedelta

# CR2C
import cr2c_labdata as pld
import cr2c_opdata as op
import cr2c_fielddata as fld
import cr2c_validation as val

#======================================> Input Arguments <=====================================

outdir_root = '/Users/josebolorinos/Google Drive/Codiga Center/Charts and Data/Monitoring Reports'
hmi_path = '/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Most Plant Parameters_20181128022807.csv'
start_dt_str = '9-9-18'
hmi_start_dt_str = '11-10-18'
end_dt_str = '11-27-18'

#====================================> Create directories <====================================

outdir = os.path.join(outdir_root,'Monitoring Report ' + end_dt_str)

if not os.path.exists(outdir): 
    
    os.mkdir(outdir)
    
labOutdir = os.path.join(outdir,'Lab Data')

if not os.path.exists(labOutdir): 
    
	os.mkdir(labOutdir)
    
HMIOutdir = os.path.join(outdir,'Operational Data')

if not os.path.exists(HMIOutdir): 
    
	os.mkdir(HMIOutdir)
    
valOutdir = os.path.join(outdir,'Validation')

if not os.path.exists(valOutdir): 
    
	os.mkdir(valOutdir)

#=====================================> Set date variables <===================================

mo6_start_dt = dt.strptime(end_dt_str,'%m-%d-%y') - timedelta(days = 180)
mo6_start_dt_str = dt.strftime(mo6_start_dt,'%m-%d-%y')
mo1_start_dt = dt.strptime(end_dt_str,'%m-%d-%y') - timedelta(days = 30)
mo1_start_dt_str = dt.strftime(mo1_start_dt,'%m-%d-%y')


#==========================> Update with latest lab and field data <===========================

# Initialize lab data class
cr2c_lr = pld.labrun() 
# Lab Data
cr2c_lr.process_data(local = True)
# Field Data
fld.process_data('DailyLogResponsesV2')

#===========================> Update with latest operational data <============================

# Initialize HMI class
op_run = op.opdata_agg(
    hmi_start_dt_str, # Start of date range we want summary data for 
    end_dt_str, # End of date range we want summary data for)
    ip_path = hmi_path
)

# Hourly averages
op_run.run_agg(
    ['COND']*4 + 
    ['PH']*4 + 
    ['WATER']*10 + 
    ['GAS']*2 + 
    ['TEMP']*3 + 
    ['TMP']*1 +
    ['DPI']*3 + 
    ['PRESSURE']*3 + 
    ['LEVEL']*5
    ,
    ['AT201','AT303','AT306','AT309'] + 
    ['AT203','AT305','AT308','AT311'] + 
    ['FT200','FT201','FT202','FT300','FT301','FT302','FT303','FT304','FT305','FIT600'] + 
    ['FT700','FT704'] + 
    ['AT202','AT304','AT310'] +
    ['AIT302'] + 
    ['DPIT300','DPIT301','DPIT302'] +
    ['PIT205','PIT700','PIT702','PIT704'] + 
    ['LT200','LT201','LIT300','LIT301','LIT302']
    ,
    [1]*35,
    ['HOUR']*35
)

# Minute averages (for sensors we are validating and membrane parameters)
op_run.run_agg(
    ['water'] +
    ['tmp'] +
    ['ph']*2 + 
    ['dpi']*2 +
    ['pressure']*2
    , # Type of sensor (case insensitive, can be water, gas, pH, conductivity, temp, or tmp
    ['FT305'] +
    ['AIT302'] +
    ['AT203','AT305'] + 
    ['DPIT300','DPIT301'] +
    ['PIT700','PIT704']
    , # Sensor ids that you want summary data for (have to be in HMI data file obviously)    
    [1]*8, # Number of hours/minutes we want to average over
    ['minute']*8, # Type of time period (can be "hour" or "minute")
)

#==============================> Get operational tables and plots<==============================

# # Initialize validation class
# cr2c_vl = val.cr2c_validation(ip_path = hmi_path)

# # Get COD balance and biotech parameters (with plots)
# cr2c_vl.get_cod_bal(end_dt_str, 8, plot = True, outdir = valOutdir)
# cr2c_vl.get_biotech_params(end_dt_str, 8, plot = True, outdir = valOutdir)

# # Get validation plots
# cr2c_vl.instr_val(
#     valtypes = ['PH','PH'],
#     start_dt_str = mo1_start_dt_str,
#     end_dt_str = end_dt_str,
#     op_sids = ['AT203','AT305'],
#     ltypes = ['PH','PH'],
#     lstages = ['Microscreen','AFBR'],
#     outdir = valOutdir
# )
# cr2c_vl.instr_val(
#     valtypes = ['DPI','DPI','PRESSURE','PRESSURE'],
#     start_dt_str = mo1_start_dt_str,
#     end_dt_str = end_dt_str,
#     op_sids = ['DPIT300','DPIT301','PIT700','PIT704'],
#     fld_varnames = [('Before Pump: R300','After Pump: R300'),('Before Pump: R301','After Pump: R301'),'Manometer Pressure: R300','Manometer Pressure: R301'],
#     outdir = valOutdir
# )