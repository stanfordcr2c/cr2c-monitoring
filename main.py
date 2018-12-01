
## Utilities
import os 
from datetime import datetime as dt
import pandas as pd
import numpy as np
import json
import sys
import warnings

# Suppress Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Google Cloud Debugger (need to activate if using Python3)
try:
    import googleclouddebugger
    googleclouddebugger.enable()
except ImportError:
    pass

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
from flask import Flask

# Initialize dash app
server = Flask(__name__)
app = dash.Dash(__name__, server = server)
app.config['suppress_callback_exceptions'] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

 
app.layout = html.Div(id = 'page-content', children = ['Hello CR2C!'])


if __name__ == '__main__':

    app.run_server(debug = True)

