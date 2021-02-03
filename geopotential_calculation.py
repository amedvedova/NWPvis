#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#
# Author : Alzbeta Medvedova
#
# This script calculates pressure and geopotential height on model levels of
# a xr.Dataset.
#
# Based on the script of Johannes Horak and later by Deborah Morgenstern
# Improvements and changes to their code:
# - necessary data already combined into one dataset is provided to this script
# - all possible calculations are vectorized (and therefore faster)
# - the functions within this module can be easily called from other scripts
# - all variables passed to functions directly, only constants are imported
# - some TODOs are left here as suggested future improvements
#
#
# All formulas are taken and referenced to IFS Documentation Cy41r1,
# Part III: Dynamics and numerical procedures by ECMWF, May 2015:
#
# https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
#
#
# Inputs:
# ds: xr.Dataset containing:
#   1. temperature and humidity on model levels
#      (optionally also other variables)
#   2. logarithm of surface pressure
#   3. surface geopotefinal
# ecmwf_ab_coeffs.pkl (optional):
#   pickle file containing level definitions a and b - can also be downloaded
#   in this script if not yet available
#
# Output:
# ds: xr.Dataset, which now contains also pressure fields, geopotential and
#   geopotential height on all model levels
#
##############################################################################


import pandas as pd
import xarray as xr
import numpy as np
import requests
import os

# local dependencies
from constants import g, Rd, Rvap


# %% MATH FROM THE ECMWF DOCUMENTATION
#
# t, q, tv: temperature, relative humidity, virtual temperature
# tv = t * [1 + (R_vap/R_dry - 1) * q]      (no Eq. number)
# code can be vectorized - no dependency on neighboring values
#
#
# p, p_S: pressure, surface pressure
# a, b: coefficients on half levels, constants from ECMWF
# p[k+1/2] = a[k+1/2] + b[k+1/2] * p_s      (Eq. 2.11)
# p[k] = (p[k+1/2] - p[k+1/2])/2            (no Eq. number)
# code can be vectorized - no dependency on neighboring values
#
#
# dp: pressure differential
# dp[k] = p[k+1/2] - p[k-1/2]               (Eq. 2.13)
# code can be vectorized - no dependency on neighboring values
#
# alpha: function of pressure, pressure gradient
# alpha[1] = ln(2)
# alpha[k] = 1 - p[k-1/2]/dp[k] * ln(p[k+1/2]/p[k-1/2])  (for k > 1, Eq. 2.23)
# code can be vectorized - no dependency on neighboring values
#
#
# psi: geopotential
# psi[k+1/2] = psi[sfc] + (sum_{j=k+1}^NLEV R_dry* tv[j] *
#                           ln(p[j+1/2]/p[j-1/2]))       (Eq. 2.21)
# psi[k] = psi[k+1/2] + alpha[k] * R_dry * tv[k]
# code CANNOT be vectorized - psi has to be calculated iteratively
#  - start at the ground where k = 137, repeat up to low k's
#  - at this point, we will have all tv and p calculated


# %% FUNCTIONS FOR INPUT DATA

def get_model_level_definition(path):
    '''
    Function loading the coefficients a, b needed for calculating pressure
    at half levels. If the file containing the coefficients does not exist yet,
    it will be downloaded.

    Parameters
    ----------
    path : str
        check it the pickle file containing the coefficients exists
        at this path - if not, download it from ECMWF and save
        as pickle (.pkl) to this path

    Returns
    -------
    a, b : pd.Series
        arrays containing the model coefficients

    '''

    level_definition_file = "ecmwf_ab_coeffs.pkl"

    # If the file with coeffs does not exist in the file directory, download it
    if level_definition_file in os.listdir(path):
        download = False
    else:
        download = True

    if download:
        url = "https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels"
        html = requests.get(url).content
        L137ml_list = pd.read_html(html)

        L137ml_df = L137ml_list[0]
        L137ml_df.to_pickle(os.path.join(path, level_definition_file))

    levDef = pd.read_pickle(os.path.join(path, level_definition_file))
    a = levDef['a [Pa]']
    b = levDef['b']

    # TODO: write checks to see if this process was successful
    return a, b

# %% Define functions for calculating


def get_geopotential(path, ds):
    '''
    Calculates pressure, geopotential, and geopotential height at all full
    model levels - to be used as vertical coordinates.

    Coefficients a, b needed for conversion are either loaded from a pickle
    file or downloaded in the process

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that has to contain the following variables:
            ln sfc pressure, temperature, rh, sfc geopotential (topography)
        The data set is expected to be 4-dimensional:
            time, level, latitude, longitude

    Returns
    -------
    ds : xr.Dataset
        Same dataset, now containing also pressure, geopotential and
        geopotential height at all vertical levels
    '''

    # TODO: rewrite code to be more flexible - check number of
    # dimensions, make it work with 3 dimensions as well
    # (if the files contain only one time step)?

    # Get max and min level values
    level_max = ds.level.max().values  # sfc level: 137 in ECMWF
    level_min = ds.level.min().values  # uppermost level in data

    # Initialize an empty storage array for geopotential in the data set:
    #   same shape and dimensions as other arrays (e.g. teperature)
    ds['geopotential'] = xr.DataArray(np.zeros(ds.t.shape), dims=ds.t.dims)

    # Get model level definitions. Either load data or download it.
    all_a, all_b = get_model_level_definition(path)

    # Get virtual temperature
    t_virtual = ds['t'] * (1 + (Rvap / Rd - 1.0) * ds['q'])

    # Get pressure and alpha
    ds['pressure'], alpha, pressure_ratio = get_pressure_and_alpha(ds,
                                                                   all_a,
                                                                   all_b)

    # Calculate geopotential iteratively, starting at surface (level_max)
    for k in range(level_max, level_min - 1, -1):

        # at the surface, input geopotential is used
        if k == level_max:

            # get values for this iteration of for-loop
            phi_k_plus_half = ds.z.values
            # prepare values for next iteration
            phi_k_plus_one_and_half = phi_k_plus_half

        # for all other levels, phi at half level below has to be calculated
        else:
            # get values for this iteration
            phi_k_plus_half = (phi_k_plus_one_and_half +
                               (Rd * t_virtual.sel(level=k+1)) *
                               np.log(pressure_ratio.sel(level=k+1)))
            # prepare values for next iteration
            phi_k_plus_one_and_half = phi_k_plus_half

        t_virtual_k = t_virtual.sel(level=k)
        alpha_k = alpha.sel(level=k)

        # formula 2.22
        phi_k = phi_k_plus_half + alpha_k * Rd * t_virtual_k

        # save results
        ds['geopotential'].loc[dict(level=k)] = phi_k

    # Once geopotential is calculated, convert it to geopotential height
    ds['geopotential_height'] = ds['geopotential'] / g

    # Add metadata
    ds.geopotential.attrs['units'] = 'm**2 s**-2'
    ds.geopotential.attrs['long_name'] = 'Geopotential on full model levels'
    ds.geopotential_height.attrs['units'] = 'm'
    ds.geopotential_height.attrs['long_name'] = \
        'Geopotential height on full model levels'
    ds.pressure.attrs['units'] = 'Pa'
    ds.pressure.attrs['long_name'] = 'Pressure on full model levels'

    return ds


def get_pressure_and_alpha(ds, all_a, all_b):
    '''
    Calculate pressure and alpha on full model levels based on Eqns. 2.11, 2.23
    from the ECMWF documentation

    Parameters
    ----------
    ds : xr.Dataset
        needs to contain log of sfc pressure (lnsp)
    a, b : pd.Series
        arrays with coefficients needed to calculate pressure on half levels

    Returns
    -------
    pressure : xr.DataArray
        Pressure [Pa] on full model levels
    alpha: xr.DataArray
        Defined at full levels, needed for geopotential calculation. Does
        NOT have to be calculated iteratively - use a vectorized calculation
    pressure_ratio: xr.DataArray
        Ratio on pressures on galf-levels, needed for calculating
        geopotential in Eq. 2.21:    p[k+1/2]/p[k-1/2]
    '''
    # Cut to size of other data: a and b coeffs on half levels
    level_min = ds.level.min().values
    a = all_a.values[level_min-1:]
    b = all_b.values[level_min-1:]

    # Get sfc pressure from log of sfc pressure
    sp = np.exp(ds['lnsp']).values

    # Get pressure on all half levels: a + b*sp, change order of axis
    # TODO: a nice way to reorder axis without hard-coded indices?
    p_half_levels = (np.multiply.outer(a, np.ones(sp.shape)) +
                     np.multiply.outer(b, sp)).transpose([1, 0, 2, 3])

    p_minus = p_half_levels[:, 0:-1, :, :]
    p_plus = p_half_levels[:, 1:, :, :]

    # Get pressure on full model levels
    pressure = xr.DataArray((p_minus + p_plus)/2,
                            dims=('time', 'level', 'latitude', 'longitude'),
                            coords=[ds.time, ds.level,
                                    ds.latitude, ds.longitude])

    # Get pressure ratio on full levels for use in alpha and Eq. 2.21
    pressure_ratio = xr.DataArray(p_plus/p_minus,
                                  dims=('time', 'level',
                                        'latitude', 'longitude'),
                                  coords=[ds.time, ds.level,
                                          ds.latitude, ds.longitude])

    # Get alpha on full model levels
    dp = p_plus - p_minus
    a_full_levels = 1 - (p_minus/dp * np.log(pressure_ratio))  # formula 2.23

    alpha = xr.DataArray(a_full_levels,
                         dims=('time', 'level', 'latitude', 'longitude'),
                         coords=[ds.time, ds.level,
                                 ds.latitude, ds.longitude])

    return pressure, alpha, pressure_ratio
