#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#
# Author : Alzbeta Medvedova
#
# This module contains physical constant necessary for calculations which are
# needed to derive physical quantities not contained in the original dataset.
#
##############################################################################



# Gravity taken from: http://glossary.ametsoc.org/wiki/Geopotential_height
g = 9.80665     # [m/s^2] globally averaged gravity at sea leve


# Gas constants for the equation of state are taken from:
# P. Markowski and Y. Richardson, 2016: Mesoscale Meteorology in Midlatitudes
# Wiley-Blackwell, Royal Meteorological Society, ISBN: 978-0-470-74213-6
Rd = 287.04     # [J/kg/K] gas constant, dry air
Rvap = 461.51   # [J/kg/K] gas constant, water vapor

t_0 = 273.15    # [K] temperature
p0 = 1e5        # [Pa] reference pressure (100 hPa)
c_p = 1005.7    # [J/kg/K] specific heat of dry air at consant pressure
c_l = 4190      # [J/kg/K] specific heat of liquid water at consant pressure
rcp = 0.2854    # [-] Rd/c_p

# TODO: REF and exact value for specific heat of liquid water


# Stull, R., 2011: "Meteorology for Scientists & Engineers, 3rd Edition
gamma_d = g/c_p  # [K/m] dry adiabatic lapse rate
