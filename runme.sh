#!/bin/bash

# This script runs the "main.py" python script, which produces the vertical cross-section figures.
# The python code takes 4 command line arguments: 3 for input data, 1 as path to output figures.


# Path to data and figures:
PATH_DATA='/home/alve/Desktop/NWPvis/data'
PATH_FIGS='/home/alve/Desktop/NWPvis/figures/'

# Model level data
ML_DATA="${PATH_DATA}/ML.nc"
# Logarithm of surface pressure
SFC_LNSP="${PATH_DATA}/SFC_LNSP.nc"
# Surface geopotential data
Z_DATA="${PATH_DATA}/TOPO.nc"

# Call the python script: save figs in the figures folder
python main_bash.py ${ML_DATA} ${SFC_LNSP} ${Z_DATA} ${PATH_FIGS}
