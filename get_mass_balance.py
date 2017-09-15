'''
	Computes a mass balance for water for any two Flowmeters 
	or for COD-CH4 in the reactor area for any range of dates
	takes dates as inputs and outputs a summary file with mass balance info
'''

from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg",force=True) 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as dates
import pylab as pl
import numpy as np
import scipy as sp
from scipy import interpolate as ip
import pandas as pd
import datetime as datetime
from datetime import datetime as dt
from datetime import timedelta as tdelt
from pandas import read_excel
import os
import sys
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory
import get_lab_data as gld

class get_mass_balance:

	def __init__(self, start_dt, end_dt):



