#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#
# Author : Alzbeta Medvedova
#
# This module contains functions to calculate derived variables based on the
# variables originally contained in the loaded dataset.
#
##############################################################################

# numerical libraries
import numpy as np
import pandas as pd

# local dependencies
from constants import g, t_0, p0, Rd, Rvap, c_p, c_l, rcp, gamma_d


# %% REFERENCES

# Bolton [1980]:
# Bolton, D., 1980: The computation of equivalent potential temperature.
# Mon. Wea. Rev., 108, 1046-1053

# Hobbs [2006]:
# Hobbs, P. V., and J. M. Wallace, 2006: Atmospheric Science: An Introductory
# Survey. 2nd ed. Academic Press, 504 pp.

# Stull [2011]:
# Stull, R., 2011: "Meteorology for Scientists & Engineers, 3rd Edition.
#  Univ. of British Columbia.  938 pages.  isbn 978-0-88865-178-5


# %%

# In the input files, we have the following variables:
# t, q, u, v
# w [Pa/s]
# vo: relative vorticity
# cc: cloud fraction
# specific rain water/snow water, cloud liquid water/cloud ice water content
# pressure, geopotential, geopotential height

# we need:
# theta_es (saturation equivalent potential temperature)
# vorticity (relative? probably...)
# N_m^2: brunt-vaisala frequency I think? (?)
# total cloud water
# hydrometeors, suspended and precipitating
# vertical vellocity: have in Pa/s, want in m/s? Need Pa/m for conversion
# CIN, CAPE
# horizontal convergence

# calculation pipeline:
# es: saturation vapor pressure
# rh: relative humidity. needs es
# theta: needs p, t
# T_lcl: needs t, rh
# theta_e: needs t, p, q, T_lcl
# vertical velocity in m/s: needs t, p, rh, es


# %% DRY THERMODYNAMICS

def theta_from_t_p(data):
    '''
    Potential temperature theta [K] at all hybrid levels

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature (data.t), and pressure (data.pressure)

    Returns
    -------
    theta : xr.DataArray
        potential temperature in C (for plotting)
    '''

    # https://en.wikipedia.org/wiki/Potential_temperature

    # Check: if temperature is in [C], convert to [K] for the calculation
    if (data.t < 100).any():
        data.t += t_0

    # calculate potential temperature [K]
    theta = data.t * (p0 / data.pressure)**rcp

    return theta


def N_dry_from_p_theta(data):
    # TODO
    # Calculate Brunt Vaisala frequency... need T, p or something?
    # What do I need here? MetPY has sigma... What's sigma?
    # sigma = -RT/p * d(ln theta)/dp... huh?
    # I want N_m

    # dry: N = sqrt(g/theta * d(theta)/dz)
    return


def windspeed(data):
    '''
    Calculate total windspeed from separate components

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing u [m/s], v [m/s], and omega [Pa/s] as vector
        components of wind

    Returns
    -------
    wspd : xr.DataArray
        Scalar windspeed [ms]
    '''

    # Check: if vertical velocity [m/s] is not calculated yet, calculate it.
    if 'w_ms' not in data.keys():
        w = w_from_omega(data)
    else:
        w = data.w_ms

    wspd = np.sqrt(data.u**2 + data.v**2 + w**2)

    return wspd


# %% MOIST THERMODYNAMICS

def es_from_t(data):
    '''
    Saturation vapor pressure based on Bolton [1980], Eq. 10. Here we use
    T[K] unlike T[C] in Bolton. Needs only temperature as input.

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature (data.t)

    Returns
    -------sat_pressure_0c
    es : xr.DataArray
        Saturation vapor pressure [Pa]
    '''
    # Check: if temperature is in C, convert to K for the calculation
    if (data.t < 100).any():
        data.t += t_0

    es = 611.2*np.exp(17.67 * (data.t-t_0) / (data.t-29.65))
    return es


def rh_from_t_q_p(data):
    '''
    Calculates relative humidity [%] at all model levels from absolute
    temperature (t), mixing ratio (q) and pressure

    Eqn. and constants from
    https://earthscience.stackexchange.com/questions/2360/
    TODO find another more reliable reference?

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature (data.t), mixing ratio (data.q) and
        pressure (data.pressure)

    Returns
    -------
    rh : xr.DataArray
        relative humidity at model levels [%]
    '''

    # Check: if temperature is in C, convert to K for the calculation
    if (data.t < 100).any():
        data.t += t_0

    # Get saturation vapor pressure
    es = es_from_t(data)

    # Get relative humidity
    rh = 100 * data.q * data.pressure * (0.622*es)**(-1)
    return rh


def w_from_omega(data):
    '''
    Convert vertical velocity from [Pa/s] to [m/s].
    We assume hydrostatic balance: dp/dz = -rho*g
    We need temperature, mixing ratio (q), pressure

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature (data.t), pressure (data.pressure),
        and mixing ratio (data.q)

    Returns
    -------
    w_ms : xr.DataArray
        Vertical wind component: positive = up

    '''

    # Check: if relative humidity is not calculated yet, calculate it.
    if 'rh' not in data.keys():
        data['rh'] = rh_from_t_q_p(data)

    # Get partial pressure of water vapor e:
    e = data.rh / 100 * es_from_t(data)

    # Get density (Wallace & Hobbs [2006], pg. 67, above Eq. 3.15)
    rho_dry = (data.pressure - e)/(Rd*data.t)
    rho_moist = e/(Rvap*data.t)
    rho = rho_dry + rho_moist

    # Get vertical velocity in [m/s]: Hobbs [2006], Eq. 7.33
    w_ms = -rho*g*data.w

    return w_ms


def T_lcl_from_T_rh(data):
    '''
    Absolute temperature at the lifting condensation level according to
    Bolton [1980], Eq. 22. Needs temperature [K] and relative humidity [%].

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # Check: if temperature is in C, convert to K for the calculation
    if (data.t < 100).any():
        data.t += t_0

    # Check: if relative humidity is not calculated yet, calculate it.
    if 'rh' not in data.keys():
        data['rh'] = rh_from_t_q_p(data)

    denominator = (1 / (data.t-55)) + (np.log(data.rh/100) / 2840)
    T_lcl = 1/denominator + 55
    return T_lcl


def theta_e_from_t_p_q_Tlcl(data):
    '''
    Equivalent potential temperature according to Bolton [1980], Eq. 43.
    Function of absolute temperature T [K], pressure p [Pa], mixing
    ratio q [kg/kg] (q is called "r" in Bolton and has unist [g/kg], we have
    [kg/kg]), and absolute temperature at the lifting condensation
    level T_lcl [K].

    Parameters
    ----------
    data : xr.Dataset
        dataset

    Returns
    -------
    theta_e : xr.DataArray
        Equivalent potential temperature

    '''

    # Get T_lcl
    T_lcl = T_lcl_from_T_rh(data)

    # Define exponents
    exp_1 = rcp * (1 - 0.28*data.q)
    exp_2 = (3.376/T_lcl - 0.00254) * 1e3*data.q*(1+0.81*data.q)

    # Get theta_e
    theta_e = data.t * (p0/data.pressure)**exp_1 * np.exp(exp_2)

    return theta_e


def theta_es_from_t_p_q(data):
    '''
    Saturated quivalent potential temperature from Bolton [1980], Eq. 43.
    Function of absolute temperature T [K], pressure p [Pa], mixing
    ratio q [kg/kg] (q is called "r" in Bolton and has unist [g/kg], we have
    [kg/kg]). Since we assume saturation, simply replace T_lcl by temperature

    Parameters
    ----------
    data : xr.Dataset
        dataset

    Returns
    -------
    theta_es : xr.DataArray
        Saturation equivalent potential temperature

    '''

    # Define exponents
    exp_1 = rcp * (1 - 0.28*data.q)
    exp_2 = (3.376/data.t - 0.00254) * 1e3*data.q*(1+0.81*data.q)

    # Get theta_e
    theta_es = data.t * (p0/data.pressure)**exp_1 * np.exp(exp_2)

    return theta_es


def N_moist_squared(data):
    '''
    Moist Brunt-Vaisala frequency. Based on Kirshbaum [2004] eq. 6, or
    Schreiner [2011], eq. 3.1. TODO REF
    Derivatives with numpy: central differences in the interior of the array,
    forward/backward differences at the boundary points

    Parameters
    ----------
    data : xr.Dataset
        dataset

    Returns
    -------
    N_m_squared : xr.DataArray
        Moist Brunt-Vaisala frequency (squared)

    '''

    # CHECK: correct to add liquid/cloud water and ice to w to get total water?
    # Total water mixing ratio calculation:
    # first get mixing ratios of all components (common denominator),
    # then sum them: r_tot = m_water_total / m_dry_air

    r_vap = data.q / (1 - data.q)            # water vapor / dry air
    r_w_cloud = data.clwc / (1 - data.clwc)  # cloud liquid water / dry air
    r_i_cloud = data.ciwc / (1 - data.ciwc)  # cloud ice water / dry air
    r_rain = data.crwc / (1 - data.crwc)     # rain / dry air
    r_snow = data.cswc / (1 - data.cswc)     # snow / dry air
    r_tot = r_vap + r_w_cloud + r_i_cloud + r_rain + r_snow  # total water

    # Moist adiabatic lapse rate, after Stull [2011], eq. 4.37b
    a = 8711    # [K]
    b = 1.35e7  # [K^2]

    # TODO: calculating gamma, use total water as well or just vapor?
    gamma_m = gamma_d * (1+(a*data.q/data.t)) / (1+(b*data.q/(data.t**2)))

    # numpy can't derive on a non-uniform meshgrid: get df and dz separately
    df = np.gradient((c_p + c_l*r_tot) * np.log(data.theta_e), axis=0)
    dz = np.gradient(data.geopotential_height, axis=0)
    dq = np.gradient(r_tot, axis=0)

    term_1 = gamma_m * (df/dz)
    term_2 = (c_l * gamma_m * np.log(data.t) + g) * (dq/dz)

    frac = 1/(1 + r_tot)
    N_m_squared = frac * (term_1 - term_2)

    return N_m_squared


# %% Rotation of wind coordinates

def bearing(lon0, lat0, lon1, lat1):
    '''
    Calculate the bearing angle (measured clockwise from the north direction)
    in RADIANS. Used for re-calculating the wind direction in the transect
    plane and in/out of page for the diagonal cross-sections.

    The diagonal cross-sections does NOT have a constant bearing - the first
    point gives just an approximation.

    Parameters
    ----------
    lon0, lat0, lon1, lat1 : float
        Location of the initial and final points of the cross-section

    Returns
    -------
    bearing : float
        bearing angle

    '''

    # [lon0, lat0, lon1, lat1] = [5.5, 46.0, 17.3, 52.0]
    [lon0, lat0, lon1, lat1] = np.deg2rad([lon0, lat0, lon1, lat1])
    dLon = lon1 - lon0

    y = np.sin(dLon)*np.cos(lat1)
    x = np.cos(lat0)*np.sin(lat1) - np.sin(lat0)*np.cos(lat1)*np.cos(dLon)

    # arctan2 chooses quadrant correctly: resulting angle will lie between
    #   -pi/2 and pi/2 since dLon > 0 by definition of the cross-sections
    #   this angle is based on the unit circle, i.e. w.r.t. east
    #   and corresponds to bearing of 0 to 180 deg w.r.t. north
    bearing = np.arctan2(y, x)

    # however, we want to keep sign of v for the view from the south:
    # limit bearing from -90 to 90 deg (-pi/2 to pi/2) around north
    if bearing > np.pi/2:
        bearing = bearing - np.pi

    return bearing


def angle(lon0, lat0, lon1, lat1):
    '''
    Calculate the angle (measured clockwise from the north direction)
    in RADIANS. Used for re-calculating the wind direction in the transect
    plane and in/out of page for the diagonal cross-sections.

    Although this is technically trigonometry on a sphere, we neglect that:
    the data is projected on a rectangular grid - the cross-sections are not
    really "straight lines" anyway.

    Parameters
    ----------
    lon0, lat0, lon1, lat1 : float
        Location of the initial and final points of the cross-section

    Returns
    -------
    bearing : float
        bearing angle

    '''

    # [lon0, lat0, lon1, lat1] = [5.5, 46.0, 17.3, 52.0]  # for trial purposes
    dLon = lon1 - lon0
    dLat = lat1 - lat0

    # arctan2 chooses quadrant correctly: resulting angle will lie between
    #   -pi/2 and pi/2 since dLon > 0 by definition of the cross-sections
    #   this angle is based on the unit circle, i.e. w.r.t. east
    #   and corresponds to an angle of 0 to 180 deg w.r.t. north
    angle = np.arctan2(dLat, dLon)  # equivalent to "atan(y/x)"

    # however, we want to keep sign of v for the view from the south:
    # limit angle from -90 to 90 deg (-pi/2 to pi/2) around north
    if angle > np.pi/2:
        angle = angle - np.pi

    # angle = np.rad2deg(angle)  # uncomment if we want degrees, not radians
    return angle


def diag_wind(u, v, angle):
    '''
    Calculates the transect/perpendicular wind components for diagonal
    cross-sections, based on the angle

    TODO check if this works correctly!

    Parameters
    ----------
    u, v : xr.DataArray
        original u-wind and v-wind components component
    angle : float
        angle in radians

    Returns
    -------
    out_of_page_wind, transect_plane_wind : xr.DataArray
        new wind components. This works when the diagonal cross-sectiones are
        viewed "from the south", i.e. with longitude increasing left to right
        of the figure

    '''

    transect_plane_wind = -v*np.cos(angle) + u*np.sin(angle)
    out_of_page_wind = v*np.sin(angle) + u*np.cos(angle)

    return out_of_page_wind, transect_plane_wind


# %% Function to call on a given input dataset to add all derived variables

def calculate_all_vars(ds):
    '''
    A function to add all the derived variables to the original input dataset

    Parameters
    ----------
    ds : xr.Dataaset
        Input dataset: contains t, q, u, v, w, vo, crwc, cswc, clwc, ciwc, cc,
        geopotential, geopotential height, pressure...

    Returns
    -------
    ds : xr.Dataset
        Output dataset: contains all the input variables plus all the derived
        quantities

    '''
    # add "initial time" attribute to calculate time differences later
    ds.attrs['init_time'] = pd.to_datetime(ds.time[0].values)

    # add all calculated variables
    # TODO check calculations with MetPy / atmos packages?
    ds['rh'] = rh_from_t_q_p(ds)       # Needs temperature still in [K]
    ds['theta'] = theta_from_t_p(ds)   # Potential temperature [K]
    ds['theta_e'] = theta_e_from_t_p_q_Tlcl(ds)  # Equivalent pot. temp. [K]
    ds['theta_es'] = theta_es_from_t_p_q(ds)  # Satur. equiv. pot. temp. [K]
    ds['w_ms'] = w_from_omega(ds)      # Vertical velocity [m/s]
    ds['wspd'] = windspeed(ds)         # Total scalar wind speed [m/s]
    ds['N_m'] = N_moist_squared(ds)    # Moist Brunt-Vaisala frequency [1/s^2]
    return ds
