
## Utilities
import os 
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import numpy as np
import json
import sys
import warnings
    
# Suppress Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore") 

## CR2C
from dependencies import cr2c_labdata as lab
from dependencies import cr2c_opdata as op
from dependencies import cr2c_fielddata as fld
from dependencies import cr2c_validation as val
from dependencies import cr2c_utils as cut

## Dash/Plotly
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event
import flask
from flask import Flask, send_file, jsonify
import urllib
from zipfile import ZipFile
import io

# Initialize dash app
server = Flask(__name__)
app = dash.Dash(__name__, server = server)
app.config['suppress_callback_exceptions'] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

#================= Create datetimea layout map from existing data =================#

lab_types = [dtype for dtype in cut.get_table_names('labdata') if dtype != 'WASTED_SOLIDS']
op_types = [dtype.split('_')[0] for dtype in cut.get_table_names('opdata')]
val_types = cut.get_table_names('valdata')
selection_vars = ['Stage','Type','Sensor ID']
selectionID, selection, click, history = [None]*4
cr2c_ddict = {'Lab Data': {},'Operational Data': {},'Validation': {}}

# Load data
lab_data = cut.get_data('labdata',lab_types)
val_data  = cut.get_data('valdata',val_types)
op_tables = cut.get_table_names('opdata', local = False)

# Load lab_type variables and their stages/types
for lab_type in lab_types:

    if lab_type in ['TSS_VSS','COD','VFA','BOD']:

        cr2c_ddict['Lab Data'][lab_type] = {
            'Stage': list(lab_data[lab_type]['Stage'].unique()),
            'Type': list(lab_data[lab_type]['Type'].unique())
        }

    elif lab_type == 'GASCOMP':

        cr2c_ddict['Lab Data'][lab_type] = {
            'Type': list(lab_data[lab_type]['Type'].unique())
        }   

    else:

        cr2c_ddict['Lab Data'][lab_type] = {
            'Stage': list(lab_data[lab_type]['Stage'].unique())
        }

for op_type in op_types:

    sids    = [table_name.split('_')[1] for table_name in op_tables if table_name.split('_')[0] == op_type]
    sids    = list(set(sids))
    ttypes  = [table_name.split('_')[3] for table_name in op_tables if table_name.split('_')[0] == op_type]
    cr2c_ddict['Operational Data'][op_type] = {'Sensor ID': sids}


val_type_descs = {'cod_balance': 'COD Balance','vss_params': 'Process Parameters','instr_validation': 'Instrument Validation'}
for val_type in val_types:

    val_type_desc = val_type_descs[val_type]
    cr2c_ddict['Validation'][val_type_desc] = {
        'Type': list(val_data[val_type]['Type'].unique())
    }    

    if val_type == 'instr_validation':

        cr2c_ddict['Validation'][val_type_desc] = {
            'Sensor ID': list(val_data[val_type]['Sensor_ID'].unique())
        } 


#### Create a nested dictionary of dynamic controls from data layout
cr2c_objects = {'Lab Data': {},'Operational Data': {},'Validation': {}}

tab_style={'color':'#0f2540','backgroundColor':'#9fabbe','borderBottom':'1px solid #d6d6d6','padding':'6px','height':'32px'}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#0f2540',
    'color': '#d0d0e1',
    'padding': '6px','height':'32px'
}

for dclass in cr2c_ddict:

    # Create nested Div to put in dtype-tab-container
    cr2c_objects[dclass]['tab'] = \
        html.Div([
            dcc.Tabs(
                id = '{}-dtype-tab'.format(dclass),
                children = [dcc.Tab(label = dtype, value = dtype, style = tab_style, selected_style = tab_selected_style) for dtype in cr2c_ddict[dclass]],
                value = list(cr2c_ddict[dclass].keys())[0],style = {'verticalAlign':'middle'}
            ),
            html.Div(
                id = '{}-selection-container'.format(dclass),
                style = {'marginTop':'0'}                
            )
        ])

    for dtype in cr2c_ddict[dclass]:

        cr2c_objects[dclass][dtype] = {}
        cr2c_objects[dclass][dtype]['tab'] = html.Div([
            dcc.Tabs(
                id = '{}-{}-vtype-tab'.format(dclass, dtype),
                children = [dcc.Tab(label = vtype, value = vtype, style = tab_style, selected_style = tab_selected_style) for vtype in cr2c_ddict[dclass][dtype]],
                value = list(cr2c_ddict[dclass][dtype].keys())[0]
            ),
            html.Div(id = '{}-{}-selection-container'.format(dclass, dtype))
        ])

        for vtype in cr2c_ddict[dclass][dtype]:

            cr2c_objects[dclass][dtype][vtype] = html.Div([
                dcc.Checklist(
                    id = '{}-{}-{}-selection'.format(dclass, dtype, vtype),
                    options = [{'label': value, 'value': value} for value in cr2c_ddict[dclass][dtype][vtype] if value],
                    values = [],labelStyle={'display': 'inline-block',"marginTop": "10"}
                )
            ],
            style={'textAlign':'center','backgroundColor':'#2e3f5d','color': '#d0d0e1'}
            )

layoutChildren = [
    dcc.Location(id='url', refresh=False),
    
    html.Div([
        html.Span(
            'Treatment Monitoring Dashboard', 
            className='app-title', 
            style = {'font-size':'36px','backgroundColor':'#2e3f5d','color':'#d0d0e1','whiteSpace':'pre'}
        ),
        html.Div(html.Img(src = 'https://i.imgur.com/t3QZAfp.png',height = '100%'),style = {'float':'right','height':'100%'})
    ],
        className = 'header',style = {'textAlign':'center','verticalAlign':'middle'}
    ),
    html.Div(
        dcc.Tabs(
            id = 'dclass-tab', 
            value = 'Lab Data',
            style = {'height':'55','color':'#d0d0e1','textAlign':'center','verticalAlign':'middle','font-size':'16px'},
            children = [dcc.Tab(label = dclass, value = dclass, selected_style = {'backgroundColor':'#9fabbe','color':'#0f2540'}) for dclass in cr2c_ddict],
            colors = {"primary": "#2e3f5d"}
        ),
        style = {'backgroundColor':'#0f2540','color':'#d0d0e1'}
    ),
    html.Div(id = 'selection-container'),
    html.Div([
        html.Label(
            'Date Filter',
            style = {'height':'30','textAlign':'center','verticalAlign':'middle','font-size':'20px'}
        ),
        dcc.DatePickerRange(
            id = 'date-picker-range',
            min_date_allowed = dt(2017, 5, 10),
            max_date_allowed = dt.today(),
            initial_visible_month = dt.today(),
            clearable = True
        )
    ],
    className='four columns',
    style={'textAlign':'left'}
    ),
    html.Div([    
        html.Div([
            html.Label(
                'Temporal Resolution', 
                style = {'height':'30','textAlign':'center','verticalAlign':'middle','font-size':'20px'}
            ),
            dcc.Dropdown(
                id = 'time-resolution-radio-item',
                options = [{'label': value, 'value': value} for value in ['Minute','Hourly','Daily','Weekly','Monthly']],
                value='Hourly',
                clearable=False
            )
        ],
        className='four columns'
        ),

        html.Div([
            html.Label(
                'Plot Order',
                style = {'height':'30','textAlign':'center','verticalAlign':'middle','font-size':'20px'}
            ),
            dcc.Dropdown(
                id = 'time-order-radio-item',
                options = [{'label': value, 'value': value} for value in ['Chronological','By Hour','By Weekday','By Month']],
                value='Chronological',
                clearable=False
            )
        ],
        className = 'four columns'
        )
    ],
    className = 'row',style = {'backgroundColor':'white'}  
    ),
    html.Div(
        [html.Button(
            'Clear Selection', 
            id = 'reset-selection-button', 
            n_clicks = 0, className='button button-primary',
            style = {'height': '45px', 'width': '220px','font-size': '15px'}
        )],
        style = {'padding': '1px','backgroundColor':'white','textAlign':'right'},
    ),
    html.Div(
        [html.A(
            'Download',
            id='download-zip',
            download = 'data.zip',
            href="/download_csv/",
            target="_blank",
            n_clicks = 0, className='button button-primary',
            style = {'height': '45px', 'width': '220px','font-size': '15px'}
        )],
        style = {'padding': '1px','backgroundColor':'white','textAlign':'right'},
    ),
    html.Div([dcc.Graph(id = 'graph-object')],style={'backgroundColor':'white'}),
    html.Div(id = 'output-container', children = '{}', style = {'display': 'none'}),
    html.Div(id = 'data-container', children = '{}')#, style = {'display': 'none'})
]

for dclass in cr2c_ddict:

    for dtype in cr2c_ddict[dclass]:

        for vtype in cr2c_ddict[dclass][dtype]:

            layoutChildren.append(html.Div(id = '{}-{}-{}-history'.format(dclass, dtype, vtype), style = {'display': 'none'}))
            layoutChildren.append(html.Div(id = '{}-{}-{}-reset-selection'.format(dclass, dtype, vtype), style = {'display': 'none'}))

 
app.layout = html.Div(id = 'page-content', children = layoutChildren, style = {'fontFamily':'sans-serif','backgroundColor':'white'})

#================= Callback functions defining interactions =================#

# Loads lightweight lab and validation data to a hidden div (avoids latency issues and keeps data updated)
@app.callback(
    Output('data-container','children'),
    [Input('url','pathname')]
)
def preload_lightweightData(pathname):

    outdict = {}

    outdict['labdata'] = {dtype: cut.get_data('labdata', lab_types)[dtype].to_json() for dtype in lab_types}
    outdict['valdata'] = {dtype: cut.get_data('valdata', val_types)[dtype].to_json() for dtype in val_types}

    return json.dumps(outdict)

@app.callback(
    Output('selection-container','children'),
    [Input('dclass-tab','value')]
)
def dclass_tab(dclass):
    return cr2c_objects[dclass]['tab']


@app.callback(
    Output('page-content','children'),
    events = [Event('reset-selection-button','click')]
)
def reset_selection():
    return layoutChildren


def generate_dclass_dtype_tab(dclass, dtype):

    def dclass_dtype_tab(dclass, dtype):
        try:
            return cr2c_objects[dclass][dtype]['tab']
        except:
            return

    return dclass_dtype_tab


def generate_dclass_dtype_vtype_selection(dclass, dtype, vtype):

    def dclass_dtype_vtype_selection(dclass, dtype, vtype):
        try:
            return cr2c_objects[dclass][dtype][vtype]
        except:
            return 

    return dclass_dtype_vtype_selection


def generate_update_selection_history(selectionID, selection):

    def update_selection_history(selectionID, selection): 
        return json.dumps({selectionID: selection})

    return update_selection_history


def generate_load_selection_value(selectionID, jhistory):

    def load_selection_value(selectionID, jhistory):
        if jhistory:
            return json.loads(jhistory)[selectionID]

    return load_selection_value


for dclass in cr2c_ddict:

    # Create selection containers for each dclass and dtype
    app.callback(
        Output('{}-selection-container'.format(dclass),'children'),
        [Input('dclass-tab','value'), Input('{}-dtype-tab'.format(dclass),'value')]
    )(generate_dclass_dtype_tab(dclass, dtype))

    for dtype in cr2c_ddict[dclass]:

        # Create selection containers for each dclass and dtype
        app.callback(
            Output('{}-{}-selection-container'.format(dclass, dtype),'children'),
            [
            Input('dclass-tab','value'), Input('{}-dtype-tab'.format(dclass),'value'),
            Input('{}-{}-vtype-tab'.format(dclass, dtype),'value')
            ]
        )(generate_dclass_dtype_vtype_selection(dclass, dtype, vtype))

        for vtype in cr2c_ddict[dclass][dtype]:

            # Update selection history div with current selection
            app.callback(
                Output('{}-{}-{}-history'.format(dclass, dtype, vtype),'children'),
                [
                Input('{}-{}-{}-selection'.format(dclass, dtype, vtype),'id'),
                Input('{}-{}-{}-selection'.format(dclass, dtype, vtype),'values')
                ]
            )(generate_update_selection_history(selectionID, selection))

            # Load previous data selected for the given dclass, dtype and vtype
            app.callback(
                Output('{}-{}-{}-selection'.format(dclass, dtype, vtype),'values'),
                [Input('{}-{}-{}-selection'.format(dclass, dtype, vtype),'id')],
                [State('{}-{}-{}-history'.format(dclass, dtype, vtype),'children')],
            )(generate_load_selection_value(selectionID, history))

# Get a list of all hidden div inputs sending output to the output container
op_cont_inputs = [
    Input('{}-{}-{}-history'.format(dclass, dtype, vtype), 'children')
        for dclass in cr2c_ddict
            for dtype in cr2c_ddict[dclass] 
                for vtype in cr2c_ddict[dclass][dtype] 
]

@app.callback(Output('output-container','children'), op_cont_inputs)


def load_data_selection(*args):

    selections = [json.loads(selection) for selection in args if selection]
    dataSelected = {}

    for selection in selections:

        dclass, dtype, vtype = list(selection.keys())[0].split('-')[0:3]
        selectionVals = list(selection.values())[0]

        if dclass in dataSelected:

            if dtype in dataSelected[dclass]:
                dataSelected[dclass][dtype][vtype] = selectionVals

            else:
                dataSelected[dclass][dtype] = {vtype: selectionVals}

        else:
            dataSelected[dclass] = {dtype: {vtype: selectionVals}}

    return json.dumps(dataSelected)


@app.callback(
    Output('graph-object','figure'),
    [
    Input('output-container','children'),
    Input('data-container','children'),
    Input('time-resolution-radio-item','value'),
    Input('time-order-radio-item','value'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range','end_date')
    ]
)

def render_plot(dataSelected, lightweightData, time_resolution, time_order, start_date, end_date):

    dataSelected = json.loads(dataSelected)
    lightweightData = json.loads(lightweightData)
    df = pd.DataFrame([])
    seriesList = []
    plotFormat = {}
    stages, types, sids = [None]*3
    seriesNo = 1
    size = 6
    dashTypes = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")

    # Number of distinct axis types being plotted     
    sel_dtypes = []
    for dclass in dataSelected:
        for dtype in dataSelected[dclass]:
            sel_dtypes.append(dtype)

    # Limit to unique data types 
    sel_dtypes = list(set(sel_dtypes))
    axes_dict = {}
    for req_dtype_ind,req_dtype in enumerate(sel_dtypes):
        axes_dict[req_dtype] = str(req_dtype_ind + 1) 

    for dclass in dataSelected:

        for dtype in dataSelected[dclass]:

            nseries = get_nseries(dataSelected)
            if nseries == 1:
                seriesNamePrefix = ''
            else:
                seriesNamePrefix = dtype + ': '

            for vtype in dataSelected[dclass][dtype]:

                stages = retrieve_value(dataSelected[dclass][dtype],'Stage')
                types = retrieve_value(dataSelected[dclass][dtype],'Type')
                sids = retrieve_value(dataSelected[dclass][dtype],'Sensor ID')

                if dclass == 'Lab Data':
                    mode = 'lines+markers'
                    # Larger size if more than one dclass is being plotted
                    if len(list(dataSelected.keys())) > 1:
                        size = 10

                if dclass == 'Operational Data':
                    mode = 'lines'

                if dclass == 'Validation':
                    mode = 'markers'

                plotFormat['size'] = size
                plotFormat['mode'] = mode
                plotFormat['seriesNamePrefix'] = seriesNamePrefix
                plotFormat['yaxis'] = 'y' +  axes_dict[dtype] 
                plotFormat['symbol'] = seriesNo
                plotFormat['dash'] = dashTypes[(seriesNo - 1) % len(dashTypes)]

            seriesList += get_data_objs(
                lightweightData,
                dclass, dtype, 
                time_resolution, time_order, start_date, end_date, 
                stages, types, sids, 
                plot = True,
                plotFormat = plotFormat
            )
            seriesNo += 1

    layout = get_layout(dataSelected, axes_dict, time_resolution, time_order)
    return {'data': seriesList  , 'layout': layout}


def get_nseries(dataSelected):

    nseries = 0
    for dclass in dataSelected:
        for dtype in dataSelected[dclass]:
            nseries += 1
    return nseries


def retrieve_value(dictionary, key):

    if key in dictionary:
        return dictionary[key]


def pad_na(df, time_var):

    df.loc[:,'Date'] = df[time_var].dt.date
    start_time = df['Date'].min()
    end_time = df['Date'].max()
    n_days = end_time - start_time
    n_days = int(n_days/timedelta(days = 1))
    time_steps = np.array([start_time + timedelta(days = d) for d in range(n_days)])
    empty_df = pd.DataFrame({'Date': time_steps})
    padded_df = df.merge(empty_df, on = 'Date', how = 'outer')
    # Replace NaT values for time_variable with date:00:00:00
    padded_df.loc[np.isnat(padded_df[time_var]),time_var] = \
        pd.to_datetime(padded_df.loc[np.isnat(padded_df[time_var]),'Date'])
    padded_df.sort_values('Date', inplace = True)
    padded_df.drop(['Date'] , axis = 1, inplace = True)
    padded_df.reset_index(inplace = True)

    return padded_df

# Queries operational data (according to availability of hourly vs minute data)
def query_opdata(dtype, sids):

    try: # Try querying hourly data
        
        table_names = ['{}_{}_1_HOUR_AVERAGES'.format(dtype.upper(), sid) for sid in sids]
        if len(sids) == 1:
            out = cut.get_data('opdata', table_names)[table_names[0]]
        else:
            out = cut.get_data('opdata', table_names)
        
    except: # Otherwise only available as minute data, needs to be aggregated to hourly
        
        for sid in sids:
            # Load minute data
            table_name = '{}_{}_1_MINUTE_AVERAGES'.format(dtype.upper(), sid)
            df = cut.get_data('opdata', [table_name])[table_name]
            # Group to hourly data
            df.loc[:,'Time'] = op_data['Time'].values.astype('datetime64[h]')
            df = df.groupby('Time').mean()
            df.reset_index(inplace = True)
            # If just one sid, output it directly, else output will be a dictionary of dfs
            if len(sids) > 1:
                out = df.copy()
            else:
                out[sid] = df

    return out


def unjson_dfs(dict_):
    return {key: pd.read_json(value) for key,value in dict_.items()}


# Gets data either for plotting or for download to a csv
def get_data_objs(
    lightweightData,
    dclass, dtype, 
    time_resolution, time_order, start_date, end_date, 
    stages, types, sids,
    plot = True,
    plotFormat = None
):
    
    # Un-json data 
    lightweightData = {key:unjson_dfs(value) for key,value in lightweightData}
    groupvars = ['Time']
    dflist = []

    if plotFormat:
        seriesNamePrefix = plotFormat['seriesNamePrefix']
    else:
        seriesNamePrefix = ''

    if dclass == 'Lab Data':

        df = lightweightData['labdata'][dtype]
        df.loc[:,'Time'] = df['Date_Time']
        df.loc[:,'yvar'] = df['Value']
        
        if stages:
            df = df.loc[df['Stage'].isin(stages),:]
            groupvars.append('Stage')
        else:
            stages = [None]

        if types:
            df = df.loc[df['Type'].isin(types),:]
            groupvars.append('Type') 
        else:
            types = [None]

        # Average all measurements taken for a given sample
        df = df.groupby(groupvars).mean()
        df.reset_index(inplace = True) 

        if plot:

            for stage in stages:

                for type_ in types:

                    if stage and type_:                
                        dfsub = df[(df['Type'] == type_) & (df['Stage'] == stage)]
                        seriesName = seriesNamePrefix + type_ + '-' + stage
                    elif stage:
                        dfsub = df[df['Stage'] == stage]
                        seriesName = seriesNamePrefix + stage
                    elif type_:
                        dfsub = df[df['Type'] == type_]
                        seriesName = seriesNamePrefix + type_
                    else:  
                        continue

                    subSeries = {'seriesName': seriesName}
                    subSeries['data'] = filter_resolve_time(dfsub, groupvars, dtype, time_resolution, time_order, start_date, end_date)
                    dflist += [subSeries]

        else:

            df = filter_resolve_time(df, groupvars, dtype, time_resolution, time_order, start_date, end_date, plot = False)

    if dclass == 'Operational Data':
        
        # Loop through sids if trying to get series for plot
        if plot:

            for sid in sids:

                dfsub = query_opdata(dtype, [sid])
                dfsub.loc[:,'yvar'] = dfsub['Value']
                seriesName = seriesNamePrefix + sid
                subSeries = {'seriesName': seriesName}
                subSeries['data'] = filter_resolve_time(dfsub, groupvars, dtype, time_resolution, time_order, start_date, end_date)
                dflist += [subSeries]

        # Otherwise, just query all sids at once, which returns a dictionary of pandas dataframes
        else:

            df_dict = query_opdata(dtype, sids)
            df = cut.merge_tables(df_dict, ['Time'], measure_varnames = ['Value']*len(sids), merged_varnames = sids)
            df = filter_resolve_time(df, groupvars, dtype, time_resolution, time_order, start_date, end_date, plot = False)

    if dclass == 'Validation':

        val_type_abbrevs = {'COD Balance':'cod_balance','Process Parameters':'vss_params','Instrument Validation':'instr_validation'}
        df = lightweightData['valdata'][val_type_abbrevs[dtype]]
        df.loc[:,'Time'] = df['Date_Time']
        df.loc[:,'yvar'] = df['Value']

        if dtype == 'Instrument Validation' and sids:
            types = ['Sensor Value','Validated Measurement','Error']
            df = df.loc[df['Sensor_ID'].isin(sids),:]
            groupvars.append('Sensor_ID')

        else:
            sids = [None]

        if types:
            df = df.loc[df['Type'].isin(types),:]
            groupvars.append('Type')
        else:
            types = [None]

        if plot:

            for type_ in types:

                for sid in sids:

                    if type_ and sid:
                        dfsub = df[(df['Type'] == type_) & (df['Sensor_ID'] == sid)]
                        seriesName = seriesNamePrefix + type_ + '-' + sid
                    elif type_:
                        dfsub = df[df['Type'] == type_]
                        seriesName = seriesNamePrefix + type_
                    elif sid:
                        dfsub = df[df['Sensor_ID'] == sid]
                        seriesName = seriesNamePrefix + sid
                    else:
                        continue

                    subSeries = {'seriesName': seriesName}
                    subSeries['data'] = filter_resolve_time(df, groupvars, dtype, time_resolution, time_order, start_date, end_date, plot = plot)
                    dflist += [subSeries] 

        else:

            df = filter_resolve_time(df, groupvars, dtype, time_resolution, time_order, start_date, end_date, plot = False)

    if plot:

        out_obj = []

        for df in dflist:

            for dfsub in df['data']:

                if dtype == 'COD Balance' and df['seriesName'] not in ['COD In','COD Balance: COD In']:

                        out_obj.append(
                            go.Bar(
                                x = dfsub['data']['Time'],
                                y = dfsub['data']['yvar'],
                                opacity = 0.8,  
                                name = df['seriesName'] + dfsub['timeSuffix'],
                                xaxis = 'x1',   
                                yaxis = plotFormat['yaxis']
                            )
                        )                    

                else:

                    if dtype == 'Process Parameters':
                        plotFormat['mode'] = 'lines+markers'

                    out_obj.append(
                        go.Scatter(
                            x = dfsub['data']['Time'],
                            y = dfsub['data']['yvar'],
                            mode = plotFormat['mode'],
                            opacity = 0.8,  
                            marker = {
                                'size': plotFormat['size'], 
                                'line': {'width': 0.5, 'color': 'white'},
                                'symbol': plotFormat['symbol'],
                            },
                            line = {'dash': plotFormat['dash']},
                            name = df['seriesName'] + dfsub['timeSuffix'],
                            xaxis = 'x1',   
                            yaxis = plotFormat['yaxis']
                        )
                    )  

    # If not plotting (i.e. if downloading data) convert to a long dataframe
    else:

        if dclass == 'Lab Data':
            out_obj = df.loc[:, groupvars + ['Value']]

        elif dclass == 'Operational Data':
            out_obj = df.loc[:, groupvars + sids]

        elif dclass == 'Validation':
            out_obj = df.loc[:, groupvars + ['Value']]


        else:

            return

        out_obj.to_csv('/Volumes/GoogleDrive/My Drive/Old Google Drive/Codiga Center/out_obj.csv')


    return out_obj


# Takes a data frame and filters, orders and groups into a desired time, splits into a list of time bins if plotting
def filter_resolve_time(dfsub, groupvars, dtype, time_resolution, time_order, start_date, end_date, plot = True):

    # Initialize empty list of output dataframes
    dflist = []
    # Add missing values to every day (if no observation)
    dfsub = pad_na(dfsub, 'Time')

    # Filter data by dates
    if start_date:
        start_date = dt.strptime(start_date,'%Y-%m-%d')
    else:
        start_date = dt(2017, 5, 10)

    if end_date:
        end_date = dt.strptime(end_date,'%Y-%m-%d')
    else:
        end_date = dt.now()

    dfsub = dfsub[dfsub['Time'] >= start_date]
    dfsub = dfsub[dfsub['Time'] <= end_date]

    # Set time resolution of data
    if time_resolution == 'Minute':

        dfsub.loc[:,'Time'] = dfsub['Time'].values.astype('datetime64[m]')
        dfsub.loc[:,'secondsMult'] = 60

    if time_resolution == 'Hourly':

        dfsub.loc[:,'Time'] = dfsub['Time'].values.astype('datetime64[h]')
        dfsub.loc[:,'secondsMult'] = 3600

    if time_resolution == 'Daily':

        dfsub.loc[:,'Time'] = pd.to_datetime(dfsub['Time'].dt.date)
        dfsub.loc[:,'secondsMult'] = 3600*24

    if time_resolution == 'Weekly':

        dfsub.loc[:,'week'] = dfsub['Time'].dt.week
        dfsub.loc[:,'year'] = dfsub['Time'].dt.year
        dfsub = dfsub.groupby(groupvars + ['year','week']).mean()
        dfsub.reset_index(inplace = True)
        dfsub.loc[:,'month'] = 1
        dfsub.loc[:,'day'] = 1
        dfsub.loc[:,'Time'] = pd.to_datetime(pd.DataFrame(dfsub[['year','month','day']]))
        dfsub.loc[:,'week'] = pd.to_timedelta(dfsub['week']*7,unit = 'd')
        dfsub.loc[:,'Time'] = dfsub['Time'] + dfsub['week']
        dfsub['secondsMult'] = 3600*24*7

    if time_resolution == 'Monthly':

        dfsub.loc[:,'month'] = dfsub['Time'].dt.month
        dfsub.loc[:,'year'] = dfsub['Time'].dt.year 
        dfsub = dfsub.groupby(groupvars + ['year','month']).mean()
        dfsub.reset_index(inplace = True)    
        dfsub.loc[:,'day'] = 1
        dfsub.loc[:,'Time'] = pd.to_datetime(pd.DataFrame(dfsub[['year','month','day']]))
        dfsub.loc[:,'Days in Month'] = dfsub['Time'].dt.daysinmonth
        dfsub['secondsMult'] = dfsub['Days in Month']*3600*24

        # If a flow, get the total (otherwise stay with average)
        if dtype in ['GAS','WATER']:
            
            dfsub.loc[:,'yvar'] = dfsub['yvar']*dfsub['Days in Month']*24

    # Group data by time ordering
    if time_order == 'By Hour':

        dfsub.loc[:,'TimeBin'] = dfsub['Time'].dt.hour
        if time_resolution in ['Daily','Weekly','Monthly']:
            return
        else:
            dfsub.loc[:,'Time'] = dfsub['TimeBin']

    if time_order == 'By Weekday':

        dfsub.loc[:,'TimeBin'] = dfsub['Time'].dt.weekday
        if time_resolution in ['Weekly','Monthly']:
            return  
        elif time_resolution == 'Daily':
            dfsub.loc[:,'Time'] = dfsub['TimeBin']
        else:
            dfsub.loc[:,'Time'] = dfsub['Time'].dt.hour

    if time_order == 'By Month':

        dfsub.loc[:,'TimeBin'] = dfsub['Time'].dt.month
        if time_resolution == 'Monthly':
            dfsub.loc[:,'Time'] = dfsub['TimeBin']
        elif time_resolution == 'Hourly':
            dfsub.loc[:,'Time'] = dfsub['Time'].dt.hour
        elif time_resolution == 'Daily':
            dfsub.loc[:,'Time'] = dfsub['Time'].dt.weekday
        else:   
            dfsub.loc[:,'year'] = dfsub['Time'].dt.year
            dfsub.loc[:,'month'] = dfsub['Time'].dt.month
            dfsub.loc[:,'day'] = 1
            dfsub.loc[:,'Time'] = dfsub['Time'] - pd.to_datetime(dfsub[['year','month','day']])

    # If the time resolution is equivalent to the ordering, just output one series
    if (
        time_order == 'Chronological' or
        time_resolution == 'Hourly' and time_order == 'By Hour' or 
        time_resolution == 'Daily' and time_order == 'By Weekday' or
        time_resolution == 'Monthly' and time_order == 'By Month' or
        plot == False
    ):
        dfsub = dfsub.groupby(groupvars).mean()
        dfsub.reset_index(inplace = True)

        if plot:

            return [{'data': dfsub,'timeSuffix': ''}]

        else:

            return dfsub

    # Otherwise output a list of series corresonding to each timebin
    else:  

        if time_order == 'By Month' and time_resolution == 'Weekly': 
            dfsub.loc[:,'Time'] = dfsub['Time'].dt.total_seconds()/dfsub['secondsMult'] 

        dfsub = dfsub.groupby(['TimeBin','Time']).mean()
        dfsub.reset_index(inplace = True)
        timebins = list(dfsub['TimeBin'].sort_values().unique())

        for timebin in timebins:

            dfsubTime = dfsub.loc[dfsub['TimeBin'] == timebin,:]
            timeSuffix = '-' + str(timebin)
            # Add to series output
            dflist.append({'data': dfsubTime, 'timeSuffix': timeSuffix})

        return dflist


# Gets plotly plot layout according to data selection and time ordering/resolution 
def get_layout(dataSelected, axes_dict, time_resolution, time_order):

    layoutItems = {'height': 700}
    position = 0.5
    xrangeList = [np.datetime64('2017-05-10'), dt.today()]
    xaxisTitles = {
        'Minute': 'Minute',
        'Hourly': 'Hour',
        'Daily': 'Day',
        'Weekly': 'Week',
        'Monthly': 'Month'
    }
    yaxisTitles =  {
        'COD' : 'COD (mg/L)',
        'BOD' : 'BOD (mg/L)',
        'OD' : 'OD (mg/L)',
        'TSS_VSS' : 'Suspended Solids (mg/L)',
        'PH' : 'pH',
        'ALKALINITY' : 'Alkalinity mg/L as CaCO3',
        'VFA' : 'VFAs as mgCOD/L',
        'AMMONIA' : 'NH3 (mg/L as N)',
        'TKN' : 'mgTKN/L',
        'SULFATE' : 'mg/L SO4',
        'GASCOMP' : 'Biogas %',
        'WATER' : 'Flow (Gal)',
        'GAS': 'Biogas Production (liters)',
        'TMP': 'Trans-Membrane Pressure (psia)',
        'PRESSURE': 'Pressure (psig)',
        'TEMP': 'Temperature (°C)',
        'DPI': 'Differential Pressure (psia)',
        'COND': 'Conductivity (S/m)',
        'LEVEL': 'Water Level (in.)',
        'COD Balance': 'COD (kg)',
        'Instrument Validation': '',
        'Process Parameters': ''
    }   

    layoutItems['xaxis'] = {
        'title': xaxisTitles[time_resolution],
        'ticks': 'inside'
    }
    if time_order == 'Chronological':
        layoutItems['xaxis']['title'] = 'Time'
        layoutItems['xaxis']['type'] = 'date'
        layoutItems['xaxis']['rangeselector'] = {
            'buttons': list([
                dict(
                    count = 7,
                    label = '1w',
                    step = 'day',
                    stepmode = 'backward'
                ),
                dict(
                    count = 1,
                    label = '1m',
                    step = 'month',
                    stepmode = 'backward'
                ),
                dict(
                    count = 6,
                    label = '6m',
                    step = 'month',
                    stepmode = 'backward'
                ),
                dict(step = 'all')
            ])
        }

    else:

        layoutItems['xaxis']['type'] = 'linear'

    for dclass in dataSelected:

        for dtype in dataSelected[dclass]:

            if dtype == 'COD Balance':
                layoutItems['barmode'] = 'stack'

            axisNo = int(axes_dict[dtype])
            if axisNo == 1 and max(list(axes_dict.values())) == 1:
                yaxisKey = 'yaxis'
            else:
                yaxisKey = 'yaxis' + str(axes_dict[dtype])

            if axisNo < 3:
                anchor = 'x'
                position = 0
            else:
                anchor = 'free'
                position = (axisNo + 1) % 2 + [-1,1][axisNo % 2]*np.floor((axisNo/2))*0.05

            axisSide = ['left','right'][axisNo % 2]
            layoutItems[yaxisKey] = {
                'title': yaxisTitles[dtype],
                'anchor': anchor,
                'side': axisSide,
                'position': position
            }

            if axisNo > 1:
                layoutItems[yaxisKey]['overlaying'] = 'y'

    if position > 0.5:
        domainLeft = 1 - position + 0.05
        domainRight = position - 0.05
    else:
        domainLeft = position + 0.05
        domainRight = 1 - position - 0.05

    layoutItems['xaxis']['domain'] = [domainLeft, domainRight]

    return go.Layout(layoutItems)


@app.callback(
    Output('download-zip', 'href'),
    [Input('output-container','children'),
    Input('time-resolution-radio-item','value'),
    Input('time-order-radio-item','value'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range','end_date')]
)
def undate_link(dataSelected, time_resolution, time_order, start_date, end_date):
    return '/download_csv?value={}${}${}${}${}'.format(dataSelected, time_resolution, time_order, start_date, end_date)


@app.server.route('/download_csv')
def download_csv():

    input = flask.request.args.get('value')
    input = input.split("$")
    dataSelected, time_resolution, time_order, start_date, end_date = input[0], input[1], input[2], input[3], input[4]

    if start_date == 'None':
        start_date = False
    if end_date == 'None':
        end_date = False

    dataSelected = json.loads(dataSelected)
    df_all = {}

    for dclass in dataSelected:
        for dtype in dataSelected[dclass]:

            stages = retrieve_value(dataSelected[dclass][dtype],'Stage')
            types = retrieve_value(dataSelected[dclass][dtype],'Type')
            sids = retrieve_value(dataSelected[dclass][dtype],'Sensor ID')

            df_all[dclass + '-' + dtype] = get_data_objs(
                dclass, dtype, 
                time_resolution, time_order, start_date, end_date, 
                stages, types, sids, 
                plot = False
            )

    create_zipfile(df_all)

    return send_file('data.zip',
                     attachment_filename='data.zip',
                     as_attachment=True)


def create_zipfile(df_selected):

    # Create a zipfile object
    zip_object = ZipFile('data.zip', 'w')

    for fileName in df_selected:

        data_sub = df_selected[fileName]
        file_name = fileName
        data_sub.to_csv('{}.csv'.format(file_name))
        zip_object.write('{}.csv'.format(file_name))

    zip_object.close()

    return zip_object



if __name__ == '__main__':

    app.run_server(debug = True, host = '0.0.0.0', port = 8080)



