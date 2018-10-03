## Synopsis

The cr2c-monitoring project manages the data systems for the Bill & Cloy Codiga Resource Recovery Center (CR2C). CR2C produces three principal streams of data: laboratory data from water quality and other testing, operational data from the facility's automated sensors, and field data collected by the facility's operators in their daily checks of the plant's performance. The scripts in this repository process and output these data streams to a single data store, perform various analyses to monitor and validate the plant's performance on a day to day basis, and integrates important information from all data into a single, online visualization and data query platform. 

## Contributors

The contributors to this project are managers and operators of the Codiga Center:

Sebastien Tilmans (Director of Operations)
Jose Bolorinos (Operator)
Yinuo Yao (Operator)
Andrew Hyunwoo Kim (Operator)

## Table of Contents

[Prerequisites](#prerequisites)
[Data Systems and Structure](#data-systems-and-structure)
	[CR2C Data Systems](#cr2c-data-systems)
	[Laboratory Data](#laboratory-data)
	[Field Data](#field-data)
	[Operational Data](#operational-data)
[Documentation](#documentation)
	[cr2c-utils](#cr2c-utils)
	[cr2c-labdata](#cr2c-labdata)
	[cr2c-hmidata](#cr2c-hmidata)
	[cr2c-fielddata](#cr2c-fielddata)
	[cr2c-validation](#cr2c-validation)
	[masterCaller](#mastercaller)

## Prerequisites

To use this repository on your machine, open a terminal window, change to the directory where you would like to place the repository, and clone the repository:

```
cd "/mydir"
git clone https://github.com/stanfordcr2c/cr2c-monitoring
```

This project uses Python's data management modules, including Numpy, Pandas, and sqlite3. In addition, all interactive plotting is done with [Dash](https://dash.plot.ly/), a web application module developed by the Plotly Project, built on top of the [Flask](http://flask.pocoo.org/docs/1.0/) framework. We also make extensive use of the Google Cloud Services platform, including:

[Google App Engine](https://developers.google.com/api-client-library/python/apis/appengine/v1)
[Google BigQuery](https://cloud.google.com/bigquery/docs/reference/libraries)
[Google Sheets API](https://developers.google.com/sheets/api/guides/concepts)

All dependencies are listed in the "cr2c-dependencies.txt" document in the repository. 

With Anaconda, these can be installed in a new Conda environment:

`conda create --name cr2c-monitoring --file cr2c-dependencies.txt`

or in an existing Conda environment: 

`conda install --name myenv --file cr2c-dependencies.txt`

Alternatively, all dependencies can be installed with pip:

`pip install -r cr2c-dependencies.txt`

## Data Systems and Structure

### CR2C Data Systems

### Laboratory Data

### Field Data

### Operational Data

## Documentation

### cr2c-labdata

### cr2c-hmidata

### cr2c-fielddata

### cr2c-validation

### masterCaller


