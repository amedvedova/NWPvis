#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#
# Author : Alzbeta Medvedova
#
# This script contains functions and classes necessary to create the final
# product of this project: the plots of vertical cross-sections.
#
##############################################################################


# numerical libraries
import numpy as np
import pandas as pd

# plotting libraries
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cmocean.cm as cmo

# local dependencies
from constants import g, t_0


# %% Plot topography: view of the topography and cross-sections from above

def plot_topography(ds):
    # initiate figure
    fig, ax = plt.subplots(figsize=[12, 8],)
    # change axis projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # plot topography (surface geopotential divided by g): shows dataset extent
    topo = ds.z/g
    topo.plot(ax=ax, cmap='terrain', vmin=0)
    # add borders + coastline for easier orientation
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)

    # add lines of constant latitude
    ax.hlines(y=[46, 47.3, 48.6],
              xmin=ds.longitude.min(),
              xmax=ds.longitude.max(),
              colors='r')
    # add lines of cconstant longitude (position of IBK)
    ax.vlines(x=[11.4],
              ymin=ds.latitude.min()-2.5,
              ymax=ds.latitude.max(),
              colors='r')
    # add diagonal lines, calculated to cross through IBK
    ax.plot([5.5, 17.3], [42.6, 52.0], color='r')
    ax.plot([5.5, 17.3], [52.0, 42.6], color='r')

    return fig, ax


# %% "Plot" class definition: parent class of all plots

class Plot:
    """  Defines features that can be added to all plots and customized
    as needed  """

    def __init__(self, data):
        self.data = data
        self.x = data.x_mesh
        # self.axis = axis

        self.lat = self.data.latitude.values
        self.lon = self.data.longitude.values
        self.topo = self.data.z.values/g

    def finish_figure_settings(self, varname):
        """
        - Adds topography to the vertical cross-section
        - Adds title, x- and y-labels, sets x-ticks and their labels
        - Sets axis limits
        - Adds text to the plot: Time since model run etc.

        """
        # add topography
        self.ax.fill_between(self.data.x_axis, self.topo, 0, color='k')

        # different settings for normal/diagonal cross-sections
        if self.data.cross_section_style == 'diagonal':
            # x-tickes + labels
            self.ax.set_xticks(self.data.x_axis[::11])
            self.ax.set_xticklabels(self.data.x_ticklabels[::11], rotation=20)
            # title format
            self.title = self.data.title.format(varname)
        elif self.data.cross_section_style == 'straight':
            # title format
            if "Latitude" in self.data.xlab:
                self.title = self.data.title.format(varname,
                                                    self.data.longitude.values)
            elif "Longitude" in self.data.xlab:
                self.title = self.data.title.format(varname,
                                                    self.data.latitude.values)

        # x-axis settings: x-label
        self.ax.set_xlabel(self.data.xlab, fontsize=14)

        # y-axis settings: y-label and axis limits
        self.ax.set_ylabel('Altitude [m]', fontsize=14)
        self.ax.set_ylim(0, 12000)

        # figure title: add new empty line for text with dates + time
        self.ax.set_title(self.title, fontsize=16)

        it = self.data.init_time                    # Initial time of model run
        ft = pd.to_datetime(self.data.time.values)  # Figure time
        dt = int((ft - it).seconds / 3600)          # Time difference [h]

        # Figure texts: e.g. 00 UTC Run: 2000JAN01 +12...
        txt_left = (str(it.hour).zfill(2) + ' UTC Run: ' +
                    str(it.year)+it.month_name()[0:3].upper()+str(it.day) +
                    '  +' + str(dt))
        txt_right = (ft.day_name()[0:3] + ' ' +
                     str(ft.year)+ft.month_name()[0:3].upper()+str(ft.day) +
                     ' ' + str(ft.hour).zfill(2) + ' UTC')

        self.ax.text(0.0, 1.01, txt_left, ha='left',
                     fontsize=14, transform=self.ax.transAxes)
        self.ax.text(1.0, 1.01, txt_right, ha='right',
                     fontsize=14, transform=self.ax.transAxes)

        # set figure size
        self.fig.set_figwidth(12)
        self.fig.set_figheight(8)
        return

    def plot_theta_contours(self, color='k'):
        """ Method to add theta contour lines to the plot"""
        isentrope = self.ax.contour(self.x,
                                    self.data.geopotential_height,
                                    self.data.theta-t_0,
                                    levels=np.arange(-60, 100, 4),
                                    colors=color,
                                    linewidths=0.7)
        isentrope.clabel(fmt='%1.0f', fontsize=12)
        return

    def plot_theta_e_contours(self, color='k'):
        """ Method to add theta_e contour lines to the plot"""
        isentrope = self.ax.contour(self.x,
                                    self.data.geopotential_height,
                                    self.data.theta_e-t_0,
                                    levels=np.arange(-60, 100, 4),
                                    colors=color,
                                    linewidths=0.7)
        isentrope.clabel(fmt='%1.0f', fontsize=12)
        return

    def plot_transect_wind(self):
        """ Method to plot transect wind (in the plane of the cross-section)
        as quivers. """
        # get vertical wind
        w = self.data.w_ms.values
        # choose every p-th point along x/y axis
        [py, px] = [2, 4]

        # get positions of points - a meshgrid to plot on
        x = self.x[::py, ::px]
        z = self.data.geopotential_height.values[::py, ::px]
        w = w[::py, ::px]
        # this is determined when cross-sections are chosen based on their
        # orientation, i.e. if they're along constant lat, lon, or diagonal
        page_plane = self.data.transect_wind[::py, ::px]

        # quiver plot: arrows in the transect plane
        q = self.ax.quiver(x, z,            # plotting coords: meshgrid
                           page_plane, w,   # x- and y- components of vectors
                           # pivot='mid',   # arrows anchored in their midpoint
                           color='black',   # color of arrows
                           width=0.002)
        # "legend" for the quiver plot
        self.ax.quiverkey(q, 1.08, 1.01, 20, label='20 m/s', labelpos='N')
        return

    def plot_out_of_page_wind_contour(self):
        # choose every p-th point along x/y axis
        [py, px] = [2, 4]

        # get positions of points - a meshgrid to plot on
        x = self.x[::py, ::px]
        z = self.data.geopotential_height.values[::py, ::px]
        # this is determined when cross-sections are chosen based on their
        # orientation, i.e. if they're along constant lat, lon, or diagonal
        out_of_page = self.data.perp_wind[::py, ::px]

        windcontour = self.ax.contour(x, z, out_of_page,
                                      levels=np.arange(-200, 200, 5),
                                      linewidths=1,
                                      colors='k')
        windcontour.monochrome = True   # Makes negative values dashed
        windcontour.clabel(fmt='%1.0f', fontsize=12)
        return

    def plot_zero_temp_line(self, color='white'):
        """ Method to add the line of 0°C to the plot"""
        zero_temp_line = self.ax.contour(self.x,
                                         self.data.geopotential_height,
                                         self.data.t-t_0,
                                         levels=[0],
                                         colors=color,
                                         linewidths=2)
        # # add label
        # zero_temp_line.clabel(fmt='%1.0f', fontsize=12)
        return


# %% Separate classes from each plot, inheriting from the class above

class Wind_plot(Plot):
    """ Inherits methods and attributes from the 'Plot' class """

    # Class attributes: specific for wind plot
    varname = 'total wind speed'
    units = '[m/s]'

    # Background specific for the wind figure
    def plot_background(self):
        bcg = self.ax.contourf(self.x,
                               self.data.geopotential_height,
                               self.data.wspd,
                               levels=20,
                               cmap=cmo.haline_r,
                               extend='neither',
                               alpha=0.9,
                               antialiased=True)

        cbar = self.fig.colorbar(bcg)
        cbar.ax.set_ylabel(self.varname.capitalize()+' '+self.units,
                           fontsize=14)
        return

    # Specify which other variables should be overlaid on the plot
    def make_figure(self):
        # initiate figure
        self.fig, self.ax = plt.subplots()
        # plot background and its colorbar
        self.plot_background()
        # add theta contours
        self.plot_theta_contours('white')
        # add transect wind
        self.plot_transect_wind()
        # add out of page wind contours
        self.plot_out_of_page_wind_contour()
        # finish figure layout settings and labels
        self.finish_figure_settings(self.varname)
        # add figure explanation below
        self.fig.tight_layout()
        ax_loc = self.fig.axes[0].get_position()
        figtext = 'ECMWF forecast: wind speed [m/s, vectors (transect plane) and black contours (full lines out of page, \ndashed into the page); shading (scalar wind speed)], potential temperature [C, white contours]'
        self.fig.text(ax_loc.xmin, 0.00, figtext,
                      ha='left', va='top', fontsize=12, wrap=True)

        return self.fig, self.ax


class Temperature_plot(Plot):
    """ Inherits methods and attributes from the 'Plot' class """

    # Class attributes: specific for wind plot
    varname = 'temperature'
    units = '[K]'

    # Background specific for the temperature figure
    def plot_background(self):
        bcg = self.ax.contourf(self.x,
                               self.data.geopotential_height,
                               self.data.t - t_0,
                               levels=20,
                               cmap=cmo.thermal,
                               extend='neither',
                               alpha=0.9,
                               antialiased=True)

        cbar = self.fig.colorbar(bcg)
        cbar.ax.set_ylabel(self.varname.capitalize()+' '+self.units,
                           fontsize=14)
        return

    # Specify which other variables should be overlaid on the plot
    def make_figure(self):
        # initiate figure
        self.fig, self.ax = plt.subplots()
        # plot background and its colorbar
        self.plot_background()
        # add theta contours
        self.plot_theta_contours('white')
        # add transect wind
        self.plot_transect_wind()
        # add out of page wind contours
        self.plot_out_of_page_wind_contour()
        # add zero temperature line
        self.plot_zero_temp_line(color='blue')
        # finish figure layout settings and labels
        self.finish_figure_settings(self.varname)
        # add figure explanation below
        self.fig.tight_layout()
        ax_loc = self.fig.axes[0].get_position()
        figtext = 'ECMWF forecast: temperature [C, shading], 0°C line (blue), wind [m/s, vectors (transect plane), black \ncontours (full lines out of page, dashed into the page)], potential temperature [C, white contours]'
        self.fig.text(ax_loc.xmin, 0.00, figtext,
                      ha='left', va='top', fontsize=12, wrap=True)
        return self.fig, self.ax


class RH_plot(Plot):
    """ Inherits methods and attributes from the 'Plot' class """

    # Class attributes: specific for wind plot
    varname = 'relative humidity'
    units = '[%]'

    # Background specific for the relative humidity figure
    def plot_background(self):
        bcg = self.ax.contourf(self.x,
                               self.data.geopotential_height,
                               self.data.rh,
                               levels=np.arange(0, 100, step=10),
                               # levels=[60, 75, 90],
                               cmap='Greens',
                               extend='max',
                               alpha=0.8,
                               antialiased=True)

        cbar = self.fig.colorbar(bcg)
        cbar.ax.set_ylabel(self.varname.capitalize()+' '+self.units,
                           fontsize=14)
        return

    # Specify which other variables should be overlaid on the plot
    def make_figure(self):
        # initiate figure
        self.fig, self.ax = plt.subplots()
        # plot background and its colorbar
        self.plot_background()
        # add theta contours
        self.plot_theta_e_contours('black')
        # add transect wind
        self.plot_transect_wind()
        # add out of page wind contours
        self.plot_out_of_page_wind_contour()
        # finish figure layout settings and labels
        self.finish_figure_settings(self.varname)
        # add figure explanation below
        self.fig.tight_layout()
        ax_loc = self.fig.axes[0].get_position()
        figtext = 'ECMWF forecast: relative humidity (shading), wind speed [m/s, vectors (transect plane), black contours \n(full lines out of page, dashed into the page), equivalent potential temperature [C, white contours]'
        self.fig.text(ax_loc.xmin, 0.00, figtext,
                      ha='left', va='top', fontsize=12, wrap=True)
        return self.fig, self.ax


class Stability_plot(Plot):
    """ Inherits methods and attributes from the 'Plot' class """
    # TODO: This still doesn't work I'm pretty sure!
    # Some mistake in the calculation probably? CHECK!

    # Class attributes: specific for stability plot
    varname = 'moist Brunt-Väisälä frequency'
    units = '[$s^{-2}$]'

    # Background specific for the relative humidity figure
    def plot_background(self):
        bcg = self.ax.contourf(self.x,
                               self.data.geopotential_height,
                               self.data.N_m,
                               levels=np.arange(-4e-4, 4e-4, 4e-5),
                               cmap='RdGy',
                               extend='both',
                               alpha=0.8,
                               antialiased=True)

        cbar = self.fig.colorbar(bcg)
        cbar.ax.set_ylabel('$N_m^2$'+' '+self.units, fontsize=14)
        return

    # Specify which other variables should be overlaid on the plot
    def make_figure(self):
        # initiate figure
        self.fig, self.ax = plt.subplots()
        # plot background and its colorbar
        self.plot_background()
        # add theta contours
        self.plot_theta_e_contours('black')
        # add transect wind
        self.plot_transect_wind()
        # add out of page wind contours
        self.plot_out_of_page_wind_contour()
        # finish figure layout settings and labels
        self.finish_figure_settings(self.varname)
        return self.fig, self.ax


class Vorticity_plot(Plot):
    """ Dataset has relative vorticity [s^-1]"""
    # TODO: not implemented
    pass


class Precipitation_plot(Plot):
    """ Dataset has suspended/precipitating water/ice"""
    # TODO: not implemented... or at least not finalized as I'd like it to be.
    # Add figtext.
    # Move cbars closer to each other to make more space for the plot itself

    # Class attributes: specific for stability plot
    varname = 'suspended and precipitating water'
    units = '[$kg~kg^{-1}$]'

    def plot_suspended_water_ice(self):
        '''
        Plots specific cloud liquid / ice water content as filled contours
        '''
        water = self.ax.contourf(self.x,
                                 self.data.geopotential_height,
                                 self.data.clwc,
                                 levels=np.array([0.05, 0.1, 0.2, 0.5])*1e-3,
                                 cmap='Greys',
                                 extend='max',
                                 alpha=0.8,
                                 antialiased=True)

        ice = self.ax.contourf(self.x,
                               self.data.geopotential_height,
                               self.data.ciwc,
                               levels=np.array([0.05, 0.1, 0.2, 0.5])*1e-3,
                               cmap='Blues',
                               extend='max',
                               alpha=0.8,
                               antialiased=True)

        # TODO 
        # to make more space for the figure itself
        cbar_w = self.fig.colorbar(water)
        cbar_i = self.fig.colorbar(ice).set_ticks([])
        cbar_w.ax.set_ylabel(self.varname.capitalize()+' '+self.units,
                             fontsize=14)
        return

    # Specify which other variables should be overlaid on the plot
    def make_figure(self):
        # initiate figure
        self.fig, self.ax = plt.subplots()
        # plot background and its colorbar
        self.plot_suspended_water_ice()
        # add theta contours
        self.plot_theta_e_contours('black')
        # add zero temperature line
        self.plot_zero_temp_line(color='blue')
        # finish figure layout settings and labels
        self.finish_figure_settings(self.varname)
        return self.fig, self.ax
