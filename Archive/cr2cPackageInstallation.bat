@echo off 
setlocal enableDelayedExpansion 
set MYDIR=C:\Users\jbolorin\Documents\cr2c-monitoring\cr2c-dependencies
for /F %%x in ('dir /B/D %MYDIR%') do (
  set FILENAME=%MYDIR%\%%x
  echo ===========================  Search in !FILENAME! ===========================
)