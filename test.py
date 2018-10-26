import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import itertools
from datetime import datetime as dt
from datetime import timedelta
import json


import cr2c_labdata as lab
import cr2c_opdata as op
import cr2c_utils as cut
import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions'] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True



if __name__ == '__main__':

    app.run_server(debug = True)