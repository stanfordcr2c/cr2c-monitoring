
# Unix

source activate cr2c-monitoring
conda list --explicit > cr2c-dependencies.txt
mkdir cr2c-dependencies
wget -i cr2c-dependencies.txt -P cr2c-dependencies/
tar -cvvzf cr2c-dependencies.tar.bz2 cr2c-dependencies

# Windows

@echo off 
setlocal enableDelayedExpansion 

set MYDIR=C:\users\user\jbolorin\Documents\cr2c-monitoring\cr2c-dependencies
for /F %%x in ('dir /B/D %MYDIR%') do (
  set FILENAME=%MYDIR%\%%x
  echo ===========================  Search in !FILENAME! ===========================
)