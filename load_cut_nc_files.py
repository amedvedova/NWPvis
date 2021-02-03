#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##############################################################################
#
# Author : Alzbeta Medvedova
#
# This module has two functions:
# 1. Load all the needed .nc files, combine them into one dataset.
#    This makes it easier to work with the data later.
#    Inputs: ML_DATA, SFC_LNSP, GEOPOTENTIAL_DATA
#                    paths to netCDF4 files containing:
#                    1. temperature and humidity on
#                    model levels (contains other variables
#                    as well)
#                    2. logarithm of surface pressure
#                    3. surface geopotential
# 2. Select slices of data to be visualized:
#    - along constant latitude
#    - along constant longitude
#    - along a defined diagonal
#    When slicing, some metadata is added to the created slices
#
##############################################################################


import xarray as xr
import numpy as np

# local imports
from calculations import angle, diag_wind


# %% Load and combine files into one dataset

def get_input_data(filename_model_level=None,
                   filename_sfc_lnsp=None,
                   filename_sfc_geopotential=None,
                   filename_allData=None):
    '''
    Loading and combining .nc files needed for the visualizaton of vertical
    cross-sections.

    The minimum of four quantities is needed:
        surface geopotential (z),
        logarithm of surface pressure (lnsp),
        temeprature (t) and
        humidity (q)
    at model levels.
    This information can either be contained in one file (allData) or three
    separate files (sfc_geopotential, sfc_lnsp, model_level).

    This function either loads three separate files and combines them into one
    dataset, or loads one dataset containing all needed data.

    Parameters
    ----------
    filename_model_level : str
        path to the .nc file containing data on model levels.
    filename_sfc_lnsp : str
        path to the .nc file containing log of sfc pressure.
    filename_sfc_geopotential : str
        path to the .nc file containing surface geopotential.
    filename_allData : str
        path to the .nc file containing all data.  The default is None.
        This is basically an artifact from the original version of the code
        (J. Horak + D. Morgenstern) - if the data is always going to be
        provided in three files, this can be removed.

    Raises
    ------
    Exception
        If files are not provided correctly (either one file with all needed
        data or three separate files), an exception is raised.

    Returns
    -------
    data_combined : xr.Dataset
        combined dataset containing all variables needed for
        further calculations
    '''

    # if one file with all data is provided, load this file
    if filename_allData is not None:
        data_combined = xr.open_dataset(filename_allData)
    # if one file is not given, check if three data files are specified
    elif any(x is None for x in [filename_model_level,
                                 filename_sfc_lnsp,
                                 filename_sfc_geopotential]):
        raise Exception("Give path to data: either one file containing all"
                        "data or three files containing parts of data.")
    # if three files are all provided, load and combine them
    else:
        tq = xr.open_dataset(filename_model_level)
        ln_sp = xr.open_dataset(filename_sfc_lnsp)
        # select only first time step to avoid time conflicts with other files
        z = xr.open_dataset(filename_sfc_geopotential).isel(time=0)

        data_combined = xr.merge([tq, ln_sp, z], join="exact")

    return data_combined


# %% Select slices along various dimensions

def slice_lat(ds, lats):
    '''
    Selects a slice of data from the dataset along lines of constant latitude

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced
    lats : list
        List of latitiude values which we want to keep for plotting

    Returns
    -------
    ds : xr.Dataset
        Subset of input data - selected slices to be visualized

    '''
    # select slices
    ds_lat = ds.sel(latitude=lats, method='nearest').copy()

    # add metadata: x-axis properties and title
    ds_lat.attrs['x_mesh'] = np.tile(ds.longitude, (len(ds.level), 1))
    ds_lat.attrs['x_axis'] = ds.longitude
    ds_lat.attrs['x_ticklabels'] = ds.longitude
    ds_lat.attrs['xlab'] = 'Longitude [°E]'
    ds_lat.attrs['title'] = 'Cross-section: {} along {:.1f}°N\n'

    # used for filling the title text later
    ds_lat.attrs['cross_section_style'] = 'straight'

    # determine which wind is transect or perpendicular
    # E/W wind:
    ds_lat['transect_wind'] = ds_lat.u
    # S/N wind - minus sign to make southerly (out of page) positive:
    ds_lat['perp_wind'] = -1*ds_lat.v

    return ds_lat


def slice_lon(ds, lons):
    '''
    Selects a slice of data from the dataset along lines of constant longitude

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced
    lons : list
        List of longitiude values which we want to keep for plotting

    Returns
    -------
    ds : xr.Dataset
        Subset of input data - selected slices to be visualized

    '''
    # select slices
    ds_lon = ds.sel(longitude=lons, method='nearest').copy()

    # add metadata: x-axis properties and title
    ds_lon.attrs['x_mesh'] = np.tile(ds.latitude, (len(ds.level), 1))
    ds_lon.attrs['x_axis'] = ds.latitude
    ds_lon.attrs['x_ticklabels'] = ds.latitude
    ds_lon.attrs['xlab'] = 'Latitude [°N]'
    ds_lon.attrs['title'] = 'Cross-section: {} along {:.1f}°E\n'

    # used for filling the title text later
    ds_lon.attrs['cross_section_style'] = 'straight'

    # determine which wind is transect or perpendicular
    ds_lon['transect_wind'] = ds_lon.v  # E/W wind
    ds_lon['perp_wind'] = ds_lon.u      # S/N wind

    return ds_lon


def slice_diag(ds, lon0, lat0, lon1, lat1):
    '''
    Selects a slice of data from the dataset along a diagonal defined by two
    points - their longitude and latitude

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced
    lon0, lat0, lon1, lat1 : float
        lat/lon coordinates of the first (0) and last (1) point of the
        diagonal cross-section

    Returns
    -------
    ds : xr.Dataset
        a diagonal cross-section with a new dimension in the data

    '''
    num = 100  # number of horizontal points for the diagonal cross-section

    # Make sure that longitude in the cross-sections always increases
    #   (if it doesn't, exchange starting points)
    # This is necessary for correct re-mapping of winds and plotting, and
    #   it also limits cross-section bearing angle from 0 to 180 deg
    if lon1 < lon0:
        [lon1, lon0] = [lon0, lon1]
        [lat1, lat0] = [lat0, lat1]

    # Define cross-section locations: arrays need the same number of points!
    lat_np = np.linspace(lat0, lat1, num)
    lon_np = np.linspace(lon0, lon1, num)

    # Get an array lat/lon pairs to be used as plot labels
    latlon_pairs = np.column_stack((lat_np, lon_np))
    xlabels = list(map(tuple, np.round(latlon_pairs, 2)))

    # Get cross-section locations as xr.DataArrays: used for interpolation
    lat_xr = xr.DataArray(lat_np, dims='diag')
    lon_xr = xr.DataArray(lon_np, dims='diag')

    # Interpolate along the defined line
    ds_diag = ds.interp(latitude=lat_xr, longitude=lon_xr).copy()

    # Index for plotting
    idx = np.linspace(0, num, num)

    # add metadata: x-axis properties and title
    ds_diag.attrs['x_axis'] = idx
    ds_diag.attrs['x_ticklabels'] = xlabels
    ds_diag.attrs['x_mesh'] = np.tile(idx, (len(ds.level), 1))
    ds_diag.attrs['xlab'] = '(Latitude [°N], Longitude [°E])'
    ds_diag.attrs['title'] = 'View from south: diagonal cross-section of {}\n'

    # used for filling the title text later
    ds_diag.attrs['cross_section_style'] = 'diagonal'

    # determine which wind is transect or perpendicular
    diag_angle = angle(lon0, lat0, lon1, lat1)

    tw, pw = diag_wind(ds_diag.u, ds_diag.v, diag_angle)
    ds_diag['transect_wind'] = tw
    ds_diag['perp_wind'] = pw

    return ds_diag
