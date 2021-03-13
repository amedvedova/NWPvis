#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#
# Author : Alzbeta Medvedova
#
# This script takes netCDF files as input and produces plots of vertical
# cross-sections of multiple atmospheric variables relevant for weather
# forecasting.
#
# STEPS:
# 1. Combine input files into one dataset (load_cut_nc_files)
# 2. Add geopotential and pressure to the dataset (geopotential_calculation)
# 3. Add all calculated variables (calculations)
# 4. Select time and slices/cross-sections (load_cut_nc_files)
# 5. Make desired plots for the selected cross-section of data (plotting)
#
#
# TODO LIST:
# - re-write the geopotential calculation code to work with selected slices
#   instead of whole 4D fields: for speed-up of the code
# - write tests to see if everything works, run those when changes are made
#
##############################################################################

import pandas as pd

# import: local dependencies
from load_cut_nc_files import get_input_data
from geopotential_calculation import get_geopotential
from calculations import calculate_all_vars
from load_cut_nc_files import slice_lat, slice_lon, slice_diag

# definitions of plotting functions/classes
from plotting import Wind_plot, Temperature_plot, RH_plot, \
                     Stability_plot, Precipitation_plot
from plotting import plot_topography


# bool option to save all the produced figures (now used for )
savefigs = False
path_figs = '/home/alve/Desktop/NWPvis/figures/'

# %% Load all data, get geopotential

# path to netCDF files containing data for visualization
PATH = '/home/alve/Desktop/NWPvis/data/'

ML_DATA = PATH + 'ML.nc'                             # model level data
SFC_LNSP = PATH + 'SFC_LNSP.nc'                      # log of surface pressure data
GEOPOTENTIAL_DATA = PATH + 'TOPO.nc'                 # surface geopotential data

# load all model data, combine into one dataset for convenience
ds = get_input_data(filename_model_level=ML_DATA,
                    filename_sfc_lnsp=SFC_LNSP,
                    filename_sfc_geopotential=GEOPOTENTIAL_DATA)

# Calculate geopotential and pressure: do this BEFORE choosing slices!
ds = get_geopotential(PATH, ds)

# add "initial time" attribute to calculate time differences later
ds.attrs['init_time'] = pd.to_datetime(ds.time[0].values)


# %% Choose data slices: add all variables to them

# slices along chosen latitudes/longitudes
#   THESE ARE EXAMPLE VALUES: to be deterined which cross-sections are going to
#   be "standardized" for the weather briefing
ds_lat = slice_lat(ds, [46.0, 47.3, 48.0, 50.0])
ds_lon = slice_lon(ds, [5.5, 6.6, 7.6, 11.4, 12.4, 13.3])

# Add calculated variables to the data set
ds_lat = calculate_all_vars(ds_lat)
ds_lon = calculate_all_vars(ds_lon)

# %% Plot top view of where the cross-sections are now
# this is useful for visualizing where the cross-sections are
# values of the plotted lines must be chosen directly in the function

plot_topo = False
if plot_topo:
    fig, ax = plot_topography(ds)


# %% Trial cross-section plotting: EXAMPLE CASES
# uncomment whatever line(s) to get different examples
# with the limited example files, select time between 0 and 5

# EXAMPLES: LAT / LON CROSS-SECTIONS
# data_out = ds_lat.sel(latitude=47.3, method='nearest').isel(time=2)
data_out = ds_lon.sel(longitude=11.4, method='nearest').isel(time=4)

# EXAMPLES: DIAGONAL CROSS-SECTIONS
# # plotting along diagonals takes time because of interpolation:
# # IMPORTANT: select time step BEFORE calling slice_diag

# ds_diag = slice_diag(ds.isel(time=2), 5.5, 47.3, 17.5, 47.8)
# data_out = calculate_all_vars(ds_diag)

# ds_diag = slice_diag(ds.isel(time=2), 5.5, 46.0, 17.5, 55.0)
# data_out = calculate_all_vars(ds_diag)


fig_t, ax_t = Temperature_plot(data_out).make_figure()
fig_wind, ax_wind = Wind_plot(data_out).make_figure()
fig_rh, ax_rh = RH_plot(data_out).make_figure()
# fig_prcp, ax_prcp = Precipitation_plot(data_out).make_figure()
# fig_Nm, axNm = Stability_plot(data_out, meshgrid, axis).make_figure()

if savefigs:
    time = str(data_out.time.dt.strftime("%Y%m%d_%H").values)
    fig_t.savefig(path_figs+time+'_T.png',
                  dpi=200, bbox_inches='tight', format='png')
    fig_wind.savefig(path_figs+time+'_wspd.png',
                     dpi=200, bbox_inches='tight', format='png')
    fig_rh.savefig(path_figs+time+'_RH.png',
                   dpi=200, bbox_inches='tight', format='png')
