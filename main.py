
## Utilities
import os 
from datetime import datetime as dt
import pandas as pd
import numpy as np
import json

## CR2C
import cr2c_labdata as lab
import cr2c_opdata as op
import cr2c_fielddata as fld
import cr2c_validation as val
import cr2c_utils as cut

## Dash/Plotly
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event

# Initialize dash app
app = dash.Dash(__name__)
app.config['suppress_callback_exceptions'] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

#================= Create datetimea layout map from existing data =================#

lab_types = ['PH','COD','TSS_VSS','ALKALINITY','VFA','GASCOMP','AMMONIA','SULFATE','TKN','BOD']
op_types = ['WATER','GAS','TMP','PRESSURE','PH','TEMP','DPI','LEVEL'] 
val_types = ['COD Balance','Process Parameters','Instrument Validation']
selection_vars = ['Stage','Type','Sensor ID', 'Validation']
selectionID, selection, click, history = [None]*4
cr2c_dtypes = {'Lab Data': {},'Operational Data': {},'Validation Data': {}}

# Load data
lab_data = lab.get_data(lab_types)
op_tables = op.get_table_names()

# Load lab_type variables and their stages/types
for lab_type in lab_types:

    if lab_type in ['TSS_VSS','COD','VFA','BOD']:

        cr2c_dtypes['Lab Data'][lab_type] = {
            'Stage': list(lab_data[lab_type]['Stage'].unique()),
            'Type': list(lab_data[lab_type]['Type'].unique())
        }

    elif lab_type == 'GASCOMP':

        cr2c_dtypes['Lab Data'][lab_type] = {
            'Type': list(lab_data[lab_type]['Type'].unique())
        }   

    else:

        cr2c_dtypes['Lab Data'][lab_type] = {
            'Stage': list(lab_data[lab_type]['Stage'].unique())
        }

for op_type in op_types:

    sids = [table_name.split('_')[1] for table_name in op_tables if table_name.split('_')[0] == op_type]
    sids = list(set(sids))
    ttypes = [table_name.split('_')[3] for table_name in op_tables if table_name.split('_')[0] == op_type]
    cr2c_dtypes['Operational Data'][op_type] = {'Sensor ID': sids}


for val_type in val_types:

    cr2c_dtypes['Validation Data'][val_type] = {'Validation': [val_type]}

#### Create a nested dictionary of dynamic controls from data layout
cr2c_objects = {'Lab Data': {},'Operational Data': {},'Validation Data': {}}

for dclass in cr2c_dtypes:

    # Create nested Div to put in dtype-tab-container
    cr2c_objects[dclass]['tab'] = \
        html.Div([
            dcc.Tabs(
                id = '{}-dtype-tab'.format(dclass),
                children = [dcc.Tab(label = dtype, value = dtype) for dtype in cr2c_dtypes[dclass]],
                value = list(cr2c_dtypes[dclass].keys())[0]
            ), 
            html.Div(
                id = '{}-selection-container'.format(dclass),
                style = {
                    'border': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'padding': '5px 2px'
                }                
            )
        ])

    for dtype in cr2c_dtypes[dclass]:

        cr2c_objects[dclass][dtype] = {}
        cr2c_objects[dclass][dtype]['tab'] = html.Div([
            dcc.Tabs(
                id = '{}-{}-vtype-tab'.format(dclass, dtype),
                children = [dcc.Tab(label = vtype, value = vtype) for vtype in cr2c_dtypes[dclass][dtype]],
                value = list(cr2c_dtypes[dclass][dtype].keys())[0]
            ),
            html.Div(id = '{}-{}-selection-container'.format(dclass, dtype))
        ])

        for vtype in cr2c_dtypes[dclass][dtype]:

            cr2c_objects[dclass][dtype][vtype] = html.Div([
                dcc.Checklist(
                    id = '{}-{}-{}-selection'.format(dclass, dtype, vtype),
                    options = [{'label': value, 'value': value} for value in cr2c_dtypes[dclass][dtype][vtype] if value],
                    values = []
                )
            ])


layoutChildren = [
    dcc.Location(id='url', refresh=False),
    html.Hr(),
    html.H1('CR2C-Monitoring Dashboard', style = {'textAlign':'center'}),
    dcc.Tabs(
        id = 'dclass-tab', 
        value = 'Lab Data', 
        children = [dcc.Tab(label = dclass, value = dclass) for dclass in cr2c_dtypes]
    ),
    html.Div(id = 'selection-container'),
    html.Hr(),
    html.Div([
        dcc.DatePickerRange(
            id = 'date-picker-range',
            min_date_allowed = dt(2017, 5, 10),
            max_date_allowed = dt.today(),
            initial_visible_month = dt.today(),
            clearable = True
        ),
        dcc.RadioItems(
            id = 'time-resolution-radio-item',
            options = [{'label': value, 'value': value} for value in ['Minute','Hourly','Daily','Weekly','Monthly']],
            value = 'Hourly'
        ),
        dcc.RadioItems(
            id = 'time-order-radio-item',
            options = [{'label': value, 'value': value} for value in ['Chronological','By Hour','By Weekday','By Month']],
            value = 'Chronological'
        ),
        html.Hr(),
        html.Div(
            [html.Button(
                'Clear Selection', 
                id = 'reset-selection-button', 
                n_clicks = 0,
                style = {'height': '50px', 'width': '200px','font-size': '20px'}
            )],
            style = {'padding': '5px 2px'}
        ),
    ]),
    html.Div([dcc.Graph(id = 'graph-object')]),
    html.Div(id = 'output-container', style = {'display': 'none'})
]

for dclass in cr2c_dtypes:

    for dtype in cr2c_dtypes[dclass]:

        for vtype in cr2c_dtypes[dclass][dtype]:

            layoutChildren.append(html.Div(id = '{}-{}-{}-history'.format(dclass, dtype, vtype), style = {'display': 'none'}))
            layoutChildren.append(html.Div(id = '{}-{}-{}-reset-selection'.format(dclass, dtype, vtype), style = {'display': 'none'}))

app.layout = html.Div(id = 'page-content', children = layoutChildren)

#================= Callback functions defining interactions =================#

@app.callback(
    Output('selection-container','children'),
    [Input('dclass-tab','value')]
)
def display_tab(dclass):
    return cr2c_objects[dclass]['tab']


@app.callback(
    Output('page-content','children'),
    events = [Event('reset-selection-button','click')]
)
def reset_selection():
    return layoutChildren


def generate_dclass_dtype_tab(dclass, dtype):

    def dclass_dtype_tab(dclass, dtype):
        return cr2c_objects[dclass][dtype]['tab']

    return dclass_dtype_tab


def generate_dclass_dtype_vtype_tab(dclass, dtype, vtype):

    def dclass_dtype_vtype_tab(dclass, dtype, vtype):
        return cr2c_objects[dclass][dtype][vtype]

    return dclass_dtype_vtype_tab


def generate_update_selection_history(selectionID, selection):

    def update_selection_history(selectionID, selection): 
        return json.dumps({selectionID: selection})

    return update_selection_history


def generate_load_selection_value(selectionID, jhistory):

    def load_selection_value(selectionID, jhistory):
        if jhistory:
            return json.loads(jhistory)[selectionID]

    return load_selection_value


for dclass in cr2c_dtypes:

    # Create selection containers for each dclass and dtype
    app.callback(
        Output('{}-selection-container'.format(dclass),'children'),
        [Input('dclass-tab','value'), Input('{}-dtype-tab'.format(dclass),'value')]
    )(generate_dclass_dtype_tab(dclass, dtype))

    for dtype in cr2c_dtypes[dclass]:

        # Create selection containers for each dclass and dtype
        app.callback(
            Output('{}-{}-selection-container'.format(dclass, dtype),'children'),
            [
            Input('dclass-tab','value'), Input('{}-dtype-tab'.format(dclass),'value'),
            Input('{}-{}-vtype-tab'.format(dclass, dtype),'value')
            ]
        )(generate_dclass_dtype_vtype_tab(dclass, dtype, vtype))

        for vtype in cr2c_dtypes[dclass][dtype]:

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


@app.callback(
    Output('output-container','children'),
    [
        Input('Lab Data-PH-Stage-history','children'),
        Input('Lab Data-COD-Stage-history','children'),
        Input('Lab Data-COD-Type-history','children'),
        Input('Lab Data-TSS_VSS-Stage-history','children'),
        Input('Lab Data-TSS_VSS-Type-history','children'),
        Input('Lab Data-ALKALINITY-Stage-history','children'),
        Input('Lab Data-VFA-Stage-history','children'),
        Input('Lab Data-VFA-Type-history','children'),
        Input('Lab Data-GASCOMP-Type-history','children'),
        Input('Lab Data-AMMONIA-Stage-history','children'),
        Input('Lab Data-SULFATE-Stage-history','children'),
        Input('Lab Data-TKN-Stage-history','children'),
        Input('Lab Data-BOD-Stage-history','children'),
        Input('Operational Data-WATER-Sensor ID-history','children'),
        Input('Operational Data-GAS-Sensor ID-history','children'),
        Input('Operational Data-TMP-Sensor ID-history','children'),  
        Input('Operational Data-PRESSURE-Sensor ID-history','children'),
        Input('Operational Data-PH-Sensor ID-history','children'),
        Input('Operational Data-TEMP-Sensor ID-history','children'),
        Input('Operational Data-DPI-Sensor ID-history','children')
    ]
)

def load_data_selection(
    sel1, sel2, sel3, sel4, sel5, 
    sel6, sel7, sel8, sel9, sel10, 
    sel11, sel12, sel13, sel14, sel15, 
    sel16, sel17, sel18, sel19, sel20
):

    selections = [json.loads(selection) for selection in locals().values() if selection]
    dataRequested = {}

    for selection in selections:

        dclass, dtype, vtype = list(selection.keys())[0].split('-')[0:3]
        selectionVals = list(selection.values())[0]

        if dclass in dataRequested:

            if dtype in dataRequested[dclass]:
                dataRequested[dclass][dtype][vtype] = selectionVals

            else:
                dataRequested[dclass][dtype] = {vtype: selectionVals}

        else:
            dataRequested[dclass] = {dtype: {vtype: selectionVals}}

    return json.dumps(dataRequested)


@app.callback(
    Output('graph-object','figure'),
    [
    Input('output-container','children'),
    Input('time-resolution-radio-item','value'),
    Input('time-order-radio-item','value'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range','end_date')
    ]
)

def render_plot(dataRequested, time_resolution, time_order, start_date, end_date):

    dataRequested = json.loads(dataRequested)
    df = pd.DataFrame([])
    seriesList = []
    plotFormat = {}
    stages, types, sids = [None]*3
    seriesNo = 1
    size = 6
    dashTypes = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")
    
    # Number of data classes are being plotted
    nclasses = len(list(dataRequested.keys()))

    # Number of distinct axis types being plotted     
    # Need to reset dclass  variable (list comprehension doesn't work otherwise)
    dclass = list(dataRequested.keys())[0]
    req_dtypes = [dtype for dtype in dataRequested[dclass] for dclass in dataRequested]
    axes_dict = {}
    for req_dtype_ind,req_dtype in enumerate(req_dtypes):
        axes_dict[req_dtype] = str(req_dtype_ind + 1)
    
    for dclass in dataRequested:

        for dtype in dataRequested[dclass]:

            nseries = get_nseries(dataRequested)
            if nseries == 1:
                seriesNamePrefix = ''
            else:
                seriesNamePrefix = dtype + ': '

            for vtype in dataRequested[dclass][dtype]:

                if dclass == 'Lab Data':

                    stages = retrieve_value(dataRequested[dclass][dtype],'Stage')
                    types = retrieve_value(dataRequested[dclass][dtype],'Type')

                    mode = 'lines+markers'
                    if nclasses > 1:
                        size = 10

                if dclass == 'Operational Data':

                    sids = dataRequested[dclass][dtype]['Sensor ID']
                    mode = 'lines'

                plotFormat['size'] = size
                plotFormat['mode'] = mode
                plotFormat['seriesNamePrefix'] = seriesNamePrefix
                plotFormat['yaxis'] = 'y' +  axes_dict[dtype] 
                plotFormat['symbol'] = seriesNo
                plotFormat['dash'] = dashTypes[(seriesNo - 1) % len(dashTypes)]

            seriesList += get_series(
                dclass, dtype, 
                time_resolution, time_order, start_date, end_date, 
                stages, types, sids, 
                plotFormat
            )
            seriesNo += 1


    layout = get_layout(dataRequested, axes_dict, time_resolution, time_order)
    return {'data': seriesList  , 'layout': layout}


def get_nseries(dataRequested):

    nseries = 0
    for dclass in dataRequested:
        for dtype in dataRequested[dclass]:
            nseries += 1
    return nseries


def retrieve_value(dictionary, key):

    if key in dictionary:
        return dictionary[key]


def get_series(
    dclass, dtype, 
    time_resolution, time_order, start_date, end_date, 
    stages, types, sids,
    plotFormat
):
    
    groupVars = ['Time']
    series = []
    dflist = []
    seriesNamePrefix = plotFormat['seriesNamePrefix']

    if dclass == 'Lab Data':

        df = lab_data[dtype]
        df.loc[:,'Time'] = df['Date_Time']
        df.loc[:,'yvar'] = df['Value']

        if stages:
            df = df[df['Stage'].isin(stages)]
            groupVars.append('Stage')
        else:
            stages = [None]

        if types:
            df = df[df['Type'].isin(types)]
            groupVars.append('Type') 
        else:
            types = [None]

        # Average all measurements taken for a given sample
        df = df.groupby(groupVars).mean()
        df.reset_index(inplace = True) 

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
                subSeries['data'] = filter_resolve_time(dfsub, dtype, time_resolution, time_order, start_date, end_date)
                dflist += [subSeries]


    if dclass == 'Operational Data':
        
        for sind, sid in enumerate(sids):

            # Retrieve data
            try: # Try querying hourly data
                
                dfsub = op.get_data([dtype], [sid], [1], ['HOUR'])
                
            except: # Otherwise only available as minute data
                
                # Load minute data
                dfsub = op.get_data([dtype], [sid], [1],['MINUTE'])
                # Group to hourly data
                dfsub.loc[:,'Time'] = op_data['Time'].values.astype('datetime64[h]')
                dfsub = dfsub.groupby('Time').mean()
                dfsub.reset_index(inplace = True)

            if dtype in ['GAS','WATER']:
                
                dfsub.loc[:,sid] = dfsub[sid]*60

            dfsub.loc[:,'yvar'] = dfsub[sid]
            seriesName = seriesNamePrefix + sid

            subSeries = {'seriesName': seriesName}
            subSeries['data'] = filter_resolve_time(dfsub, dtype, time_resolution, time_order, start_date, end_date)
            dflist += [subSeries]

    for df in dflist:

        for dfsub in df['data']:

            series.append(
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

    return series


def filter_resolve_time(dfsub, dtype, time_resolution, time_order, start_date, end_date):

    # Initialize empty list of output dataframes
    dflist = []
    # Filter data by dates
    if start_date:

        start_date = dt.strptime(start_date,'%Y-%m-%d')
        dfsub = dfsub[dfsub['Time'] >= start_date]

    if end_date:

        end_date = dt.strptime(end_date,'%Y-%m-%d')
        dfsub = dfsub[dfsub['Time'] <= end_date]

    # Set time resolution of data
    if time_resolution == 'Minute':

        dfsub.loc[:,'Time'] = dfsub['Time'].values.astype('datetime64[m]')
        dfsub['secondsMult'] = 60

    if time_resolution == 'Hourly':

        dfsub.loc[:,'Time'] = dfsub['Time'].values.astype('datetime64[h]')
        dfsub['secondsMult'] = 3600

    if time_resolution == 'Daily':

        dfsub.loc[:,'Time'] = pd.to_datetime(dfsub['Time'].dt.date)
        dfsub['secondsMult'] = 3600*24

    if time_resolution == 'Weekly':

        dfsub.loc[:,'week'] = dfsub['Time'].dt.week
        dfsub.loc[:,'year'] = dfsub['Time'].dt.year
        dfsub = dfsub.groupby(['year','week']).mean()
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
        dfsub = dfsub.groupby(['year','month']).mean()
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
        time_resolution == 'Monthly' and time_order == 'By Month' 
    ):
        dfsub = dfsub.groupby('Time').mean()
        dfsub.reset_index(inplace = True)
        return [{'data': dfsub,'timeSuffix': ''}]

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


def get_layout(dataRequested, axes_dict, time_resolution, time_order):

    layoutItems = {'height': 700}
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
        'ALKALINITY' : r'$\\text{Alkalinity (mg/L as } CaCO_3 \\text{)}$',
        'VFA' : 'VFAs as mgCOD/L',
        'AMMONIA' : r'$NH_3\\text{ (mg/L as N)}$',
        'TKN' : 'mgTKN/L',
        'SULFATE' : r'$\\text{mg/L }SO_4$',
        'GASCOMP' : 'Biogas %',
        'WATER' : 'Flow (Gal)',
        'GAS': 'Biogas Production (liters)',
        'TMP': 'Trans-Membrane Pressure (psia)',
        'PRESSURE': 'Pressure (psig)',
        'TEMP': 'Temperature (°C)',
        'DPI': 'Differential Pressure (psia)',
        'LEVEL': 'Water Level (in.)'
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

    axisSides = ['right','left']
    axisSigns = [-1,1]

    for dclass in dataRequested:

        for dtype in dataRequested[dclass]:

            axisNo = int(axes_dict[dtype])
            if axisNo == 1:
                yaxisKey = 'yaxis'
            else:
                yaxisKey = 'yaxis' + str(axes_dict[dtype])

            if axisNo < 3:
                anchor = 'x'
                position = 0
            else:
                anchor = 'free'
                position = (axisNo + 1) % 2 + axisSigns[axisNo % 2]*np.floor((axisNo/2))*0.05

            axisSide = axisSides[axisNo % 2]
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



if __name__ == '__main__':

    app.run_server(debug = True)

