
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


def update_data(
    pydir, 
    lab_update,
    fld_update,
    op_update,
        hmi_path, 
        hour_sids,
        minute_sids,
        op_start_dt_str, 
        op_end_dt_str,
    val_update,
        biotech_params,
        val_sids,
        val_end_dt_str,
        nweeks_back,
):

    # Dictionary of sensor types (stypes) and sensor ids (sids)
    sids_stypes = {
        'COND': ['AT201','AT303','AT306','AT309'],
        'PH': ['AT203','AT305','AT308','AT311'],
        'WATER': ['FT200','FT201','FT202','FT300','FT301','FT302','FT303','FT304','FT305','FIT600'],
        'GAS': ['FT700','FT702','FT704'],
        'TEMP': ['AT202','AT304','AT307','AT310'],
        'TMP': ['AIT301','AIT302','AIT304','AIT305','AIT306','AIT307'],
        'DPI': ['DPIT300','DPIT301','DPIT302'],
        'PRESSURE': ['PIT205','PIT700','PIT702','PIT704'],
        'LEVEL': ['LT100','LT200','LT201','LIT300','LIT301','LIT302'],
        'POWER': ['B801_KW','C200_KW','MainFeed_KW','MS200_KW','P100_KW','P101_KW','P304_KW'],
    }

    sids_val_relation = {
        'lab': {
            'AT203':'Microscreen',
            'AT305':'AFBR',
            'AT308':'Research AFMBR MLSS',
            'AT311':'Duty AFMBR MLSS'
        },
        'fld': {
            'DPIT300':('Before Pump: R300','After Pump: R300'),
            'DPIT301':('Before Pump: R301','After Pump: R301'),
            'DPIT302':('Before Pump: R302','After Pump: R302'),
            'PIT700':'Manometer Pressure: R300',
            'PIT702':'Manometer Pressure: R301',
            'PIT704':'Manometer Pressure: R302'
        }
    }

    # Update lab Data
    if lab_update:
        # Initialize lab data class
        cr2c_lr = pld.labrun() 
        cr2c_lr.process_data(pydir = pydir)

    # Update field data
    if fld_update:
        # Field Data
        fld.process_data(pydir = pydir, table_name = 'DailyLogResponsesV3', if_exists = 'replace')

    # Update operational data
    if op_update:
        # Initialize HMI class
        op_run = op.opdata_agg(
            op_start_dt_str, 
            op_end_dt_str, 
            ip_path = hmi_path
        )

        for ttype in ['HOUR','MINUTE']:

            for stype in sids_stypes.keys():

                if ttype == 'HOUR':
                    sids = hour_sids
                else:
                    sids = minute_sids

                sids_stype = [sid for sid in sids if sid in sids_stypes[stype]]
                
                if sids_stype:

                    nsids = len(sids_stype)
                    op_run.run_agg(
                        [stype]*nsids,
                        sids_stype,
                        [1]*nsids,
                        ['HOUR']*nsids
                    )         

    # Update validation data
    if val_update:

        # Initialize validation class
        cr2c_vl = val.cr2c_validation()

        # Get COD balance and biotech parameters
        if biotech_params:

            cr2c_vl.get_biotech_params(val_end_dt_str, nweeks_back)

        # Run sensor validation
        if val_sids:
            # Loop through val_methods ("lab" for laboratory validation and "fld" for field measurement validation)
            for val_method in sids_val_relation.keys():

                sids   = [sid for sid in val_sids if sid in sids_val_relation[val_method]]
                stypes = [stype for stype in sids_stypes.keys() for sid in sids if sid in sids_stypes[stype]]
                val_tags   = [sids_val_relation[val_method][sid] for sid in sids]

                if sids:
                    cr2c_vl.get_sval(
                        val_method = val_method,
                        stypes = stypes,
                        sids = sids,
                        val_tags = val_tags,
                    )


update_data(
    pydir = '/Volumes/GoogleDrive/Shared drives/CR2C.Box/Monitoring Data and Procedures/Python/GoogleProjectsAdmin',
    lab_update = False,
    fld_update = False,
    op_update = False,
        hmi_path = '/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/cr2c_opdata_edna_template_20190811050838.csv',
        hour_sids = 
            ['AT201','AT303','AT306','AT309'] + 
            ['AT203','AT305','AT308','AT311'] + 
            ['FT200','FT201','FT202','FT300','FT301','FT302','FT303','FT304','FT305','FIT600'] + 
            ['FT700','FT702','FT704'] + 
            ['AT202','AT304','AT310'] +
            ['AIT302'] + 
            ['DPIT300','DPIT301','DPIT302'] +
            ['PIT205','PIT700','PIT702'] + 
            ['LT100','LT200','LT201','LIT300','LIT301']
        ,
        minute_sids = 
            ['AT203','AT305'] +
            ['FT305'] +
            ['AIT302'] + 
            ['DPIT300','DPIT301'] +
            ['PIT700']
        ,
        op_start_dt_str = '7-27-19',
        op_end_dt_str = '8-9-19',
    val_update = True,
        biotech_params = False,
        # val_sids = ['DPIT300','DPIT301','DPIT302','PIT700','PIT702','PIT704'],
        val_sids = ['AT203','AT305','AT308','AT311','DPIT300','DPIT301','DPIT302','PIT700','PIT702','PIT704'],
        val_end_dt_str = '8-9-19',
        nweeks_back = 8

)
