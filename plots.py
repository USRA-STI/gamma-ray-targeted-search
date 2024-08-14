# Copyright 2017-2022 by Universities Space Research Association (USRA). All rights reserved.
#
# Developed by: William Cleveland and Adam Goldstein
#               Universities Space Research Association
#               Science and Technology Institute
#               https://sti.usra.edu
#
# Developed by: Daniel Kocevski and Joshua Wood
#               National Aeronautics and Space Administration (NASA)
#               Marshall Space Flight Center
#               Astrophysics Branch (ST-12)
#
# Developed by: Lorenzo Scotton
#               University of Alabama in Huntsville
#               Center for Space Plasma and Aeronomic Research
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations under the
# License.
#
import os
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as colormaps
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.colorbar as mcolorbar
import matplotlib.ticker as ticker
from astropy import units as u

from gdt.core.plot.earthplot import EarthPlot, EarthPoints

from gdt.core.plot.lightcurve import Lightcurve
from gdt.core.binning.binned import rebin_by_time
from gdt.core.binning.binned import combine_by_factor

from gdt.core.plot.plot import Histo
from gdt.core.plot.plot import HistoErrorbars

from gdt.core.background.primitives import BackgroundRates
from gdt.core.data_primitives import Ebounds

def plot_orbit(spacecraft_frames, t0, filename, saa=None):
    """Plot the orbital position of a spacecraft. Shows the orbit, position of the spacecraft,
    the SAA, and the McIlwain L gradient.
    
    Args:
        spacecraft_frames (SpacecraftFrame): spacecraft frame information over time
        t0 (Time): The time of interest
        filename (str): The filename for the image
        saa (SouthAtlanticAnomaly): class with polygon border definition for the South Atlantic Anomaly.
    """

    # Initialize the plot
    orbit_plot = EarthPlot(interactive=False, saa=saa)

    # Limit to 45 min on either side of t0
    obstime = spacecraft_frames.obstime
    tstart = t0 - 2700.0 * u.second
    tstop = t0 + 2700.0 * u.second

    # Slice the obstime object
    obstime_subset = obstime[(obstime >= tstart) & (obstime <= tstop)]
    spacecraft_frames_subset = spacecraft_frames.at(obstime_subset)

    # Make the plot
    orbit_plot.add_spacecraft_frame(spacecraft_frames_subset, trigtime=t0, sizes=[100], color='green', zorder=1000)

    orbit_plot.orbit.color = 'darkblue'
    orbit_plot.orbit.alpha = 1
    orbit_plot.orbit.linewidth = 2

    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    except ValueError as err:
        print(err)
    plt.close()

class Waterfall():
    """Class to make waterfall plots of variables from the targeted search.
    These are made as a function of candidate start time and duration. We call
    them 'waterfall' plots because real transients create an image that resembles
    a waterfall when using a blue color scale.
    """
    
    def __init__(self, results, t0, figsize=(12,6), fontsize=12):
        """ Class constructor

        Args:
            results (Results): The targeted search results object
            figsize (tuple(2), optional): The figure size in inches. Default is (12, 6)
            fontsize (int, optional): The fontsize of the plot axes labels.  Default is 14
        """
        self._t0 = t0
        self._results = results
        self._fig = None
        self._ax = None
        self._fontsize=fontsize
        self._figsize = figsize
        self._x = np.sqrt(2.0)
        self.dpi=150
    
    def plot_loglr(self, filename=None, spectra=False, **kwargs):
        """Waterfall plot for the log-likelihood ratio

        Note: Need to consolidate plot_loglr, plot_coinclr, etc into a single function
        that has arguments for the variable name and plot title.

        Args:
            filename (str): The filename to save to. Using None will display the image.
            spectra (bool, optional): If true, plot different colors for the spectra. Default is False
            val_min (float, optional): Minimum value to plot. Defaults to the lowest positive value
            val_max (float, optional): Maximum value to plot. Defaults to the highest value
            log_color (bool, optional): If True, scales the colors logarithmically, otherwise scales
                                        linearly.  Default is True.
            cmap (str, optional): The color map to use. Default is 'Blues'
            cmaps (list, optional): For multi-spectra plot, the color maps to use. Default is
                                    ('Purples', 'Blues', 'Greens')
        """        
        self._results.sort(loglr=True)
        vals = self._results.loglr
        title = 'Marginalized log-likelihood ratio'
        if not spectra:
            self._plot_one(filename, vals, title, **kwargs)
        else:
            self._plot_multi(filename, vals, title, **kwargs)
    
    def plot_coinclr(self, filename, spectra=False, **kwargs):
        """Waterfall plot for the 'coincident' log-likelihood ratio which is defined
        by using a localization from another instrument as the prior marginalizing the
        log-likelihood ratio over the sky.

        Args:
            See plot_logr for list of parameters.
        """
        self._results.sort(coinclr=True)
        vals = self._results.coinclr
        title = 'Coincident Marginalized log-likelihood ratio'
        if not spectra:
            self._plot_one(filename, vals, title, **kwargs)
        else:
            self._plot_multi(filename, vals, title, **kwargs)
    
    def plot_snr(self, filename, spectra=False, **kwargs):
        """Waterfall plot for the signal-to-noise ratio.

        Args:
            See plot_logr for list of parameters.
        """
        self._results.sort(snr=True)
        vals = self._results.snr[:,0]
        title = 'S/N Ratio'
        if not spectra:
            self._plot_one(filename, vals, title, **kwargs)
        else:
            self._plot_multi(filename, vals, title, **kwargs)
    
    def plot_pflux(self, filename, spectra=False, **kwargs):
        """Waterfall plot for photon flux.

        Args:
            See plot_logr for list of parameters.
        """
        self._results.sort(amplitude=True)
        vals = self._results.amplitudes
        title = r"Photon Flux (erg s$^{-1}$ cm$^{-2}$)"    
        if not spectra:
            self._plot_one(filename, vals, title, **kwargs)
        else:
            self._plot_multi(filename, vals, title, **kwargs)
    
    def _plot_multi(self, filename, vals, title, val_max=None, val_min=None, 
                   log_color=True, cmaps=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']):
        """ Internal method for plotting multiple waterfall plots on the same axes

        Args:
            filename (str): The filename to save to. Using None will display the image.
            vals (np.narray): Values to plot
            val_min (float, optional): Minimum value to plot. Defaults to the lowest positive value
            val_max (float, optional): Maximum value to plot. Defaults to the highest value
            log_color (bool, optional): If True, scales the colors logarithmically, otherwise scales
                                        linearly.  Default is True.
            cmaps (list, optional): For multi-spectra plot, the color maps to use. Default is
                                    ('Purples', 'Blues', 'Greens')
 
        """
        spectra = np.unique(self._results.templates) 
        nspec = spectra.size

        if len(cmaps) < nspec:
            raise ValueError("Too few color maps defined")
        else:
            cmaps = np.array(cmaps)
            cmap_index = np.arange(nspec).astype(int)
            cmaps = cmaps[cmap_index]            
        
        # the rectangle dimensions
        dims = self._rect_dims(self._results.times_relative, self._results.durations)
        
        if val_max is None:
            val_max = np.nanmax(vals)
        if val_min is None:
            if log_color:
                # this appears to be a typo. val should be vals?
                val_min = np.nanmin(val[val > 0.0])
            else:
                val_min = np.nanmin(vals)
        
        # get the colors for each colormap, scaling by color fractions
        cmaps = [colormaps.get_cmap(cmap) for cmap in cmaps]
        color_fracs = self._scale_color(vals, val_min, val_max, log_color) 
        colors = np.empty((self._results.size, 4))
        for i in range(nspec):
            mask = (self._results.templates == spectra[i])
            colors[mask] = cmaps[i](color_fracs[mask], alpha=1.0)

        # Create the rectangle patch with a specific color
        # now using PatchCollection because its ~2x faster than adding
        # each patch individually
        self._init_fig()
        patches = np.empty(self._results.size, dtype=object)
        for i in range(self._results.size):
            rect = mpatches.Rectangle(*dims[i], facecolor=colors[i,:], edgecolor=None)
            patches[i] = rect
        self._ax.add_collection(PatchCollection(patches, match_original=True))
    
        # add the colorbar
        self._multi_colorbar(cmaps, spectra, log_color, (val_min, val_max), title)              
    
        try:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        except ValueError as err:
            print(err)
        plt.close()
        
    def _plot_one(self, filename, vals, title, val_max=None, val_min=None, 
              log_color=True, cmap='Blues'):            
        """ Internal method for plotting a single waterfall plot

        Note: is this function really needed? Can't we just add an argument to plot_multi to allow
        a case where all spectra are combined?

        Args:
            filename (str): The filename to save to. Using None will display the image.
            vals (np.narray): Values to plot
            val_min (float, optional): Minimum value to plot. Defaults to the lowest positive value
            val_max (float, optional): Maximum value to plot. Defaults to the highest value
            log_color (bool, optional): If True, scales the colors logarithmically, otherwise scales
                                        linearly.  Default is True.
            cmap (str, optional): The color map to use. Default is 'Blues'
        """
        # the rectangle dimensions
        dims = self._rect_dims(self._results.times_relative, self._results.durations)

        if val_max is None:
            val_max = np.nanmax(vals)
        if val_min is None:
            if log_color:
                val_min = np.nanmin(val[val > 0.0])
            else:
                val_min = np.nanmin(vals)
        
        # Get the color map, calculate color fraction, and map to rgb(a)
        cmap = colormaps.get_cmap(cmap)
        color_fracs = self._scale_color(vals, val_min, val_max, log_color)        
        colors = cmap(color_fracs, alpha=1.0)
        
        # Create the rectangle patch with a specific color
        # now using PatchCollection because its ~2x faster than adding
        # each patch individually
        self._init_fig()
        patches = np.empty(self._results.size, dtype=object)
        for i in range(self._results.size):
            rect = mpatches.Rectangle(*dims[i], facecolor=colors[i,:], edgecolor=None)
            patches[i] = rect
        self._ax.add_collection(PatchCollection(patches, match_original=True))
    
        # add the colorbar
        self._one_colorbar(cmap, log_color, (val_min, val_max), title=title)        

        if filename is None:
            plt.show()
            return

        try:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        except ValueError as err:
            print(err)
        plt.close()

    def _init_fig(self):
        """ Internal method to intialize the plot figure"""
        self._fig = plt.figure(figsize=self._figsize)
        self._ax = self._fig.gca()

        # set the x axis
        self._ax.set_xlim(np.min(self._results.times_relative), 
                          np.max(self._results.times_relative))
        self._ax.set_xlabel('Time (s) - {}'.format(self._t0), 
                            fontsize=self._fontsize) 

        # Set the ylimits and scale
        self._ax.set_ylim(self._results.timescales[0]/self._x, 
                          self._results.timescales[-1]*self._x)
        self._ax.set_yscale('log')
        # Set the yticks and ylabel
        self._ax.set_yticks(self._results.timescales)
        self._ax.set_yticklabels(['{:.3f}'.format(t) for t in self._results.timescales])
        self._ax.set_ylabel('Timescale (s)', fontsize=self._fontsize)

        # Turn on the minor ticks, but remove them on the y-axis
        self._ax.minorticks_on()
        self._ax.tick_params(axis='y',which='minor',left='off')
        self._ax.tick_params(axis='y',which='minor',right='off')
     
    def _rect_dims(self, tcents, durs):
        """ Internal method to set the rectangular dimensions in the plot for each of search bins

        Args:
            tcents (np.ndarray): start time of the search bins
            durs (np.ndarray): durations of the search bins
        """
        n = len(tcents)
        dims = [((tcents[i]-durs[i]/2.0, durs[i]/self._x), durs[i], 
                 durs[i]*(self._x-1.0/self._x)) for i in range(n)]
        return dims
    
    def _scale_color(self, vals, min_val, max_val, log_color):
        """ Method to calculate the fraction of the min-max allowable range.
        This will represent the color to use on the colorbar given the 
        data value and min-max range of colorbar

        Args:
            vals (np.ndarray): array of values to be plotted
            val_min (float, optional): Minimum value to plot. Defaults to the lowest positive value
            val_max (float, optional): Maximum value to plot. Defaults to the highest value
            log_color (bool, optional): If True, scales the colors logarithmically, otherwise scales linearly.
        """
        if log_color:
            color_fracs = np.log(vals-min_val)/np.log(max_val-min_val)
        else:
            color_fracs = (vals-min_val)/(max_val-min_val)
        color_fracs[np.isnan(color_fracs)] = 0.0
        color_fracs[color_fracs < 0.0] = 0.0
        return color_fracs
    
    def _one_colorbar(self, cmap, log_color, val_range, title):
        """ Method to plot the colorbar for a single spectrum

        Args:
            cmap (str): name of the color map to use
            log_color (bool, optional): If True, scales the colors logarithmically, otherwise scales linearly.
            val_range (tuple(2)): the minimum and maximum values of the color range
            title (str): title to apply to this color bar
        """
        if log_color:
            norm = LogNorm(vmin=val_range[0], vmax=val_range[1])
        else:
            norm = Normalize(vmin=val_range[0], vmax=val_range[1])
        
        cb_ax = self._fig.add_axes([0.905, 0.11, 0.015, 0.77])
        colorbar = mcolorbar.ColorbarBase(cb_ax, cmap=cmap, norm=norm,
                                          orientation='vertical',
                                          format='%3.0f')
        colorbar.set_label(title, fontsize=self._fontsize)
    
    def _multi_colorbar(self, cmaps, names, log_color, val_range, title):
        """ Method to plot the colorbars for multiple spectra

        Args:
            cmaps (list): the color maps to use.
            names (list): names to apply to each color bar.
            log_color (bool, optional): If True, scales the colors logarithmically, otherwise scales linearly.
            val_range (tuple(2)): the minimum and maximum values of the color range
            title (str): title to apply to this color bar
        """
        if log_color:
            norm = LogNorm(vmin=val_range[0], vmax=val_range[1])
        else:
            norm = Normalize(vmin=val_range[0], vmax=val_range[1])
        
        nbars = len(names)
        cbs = np.empty(nbars, dtype=object)
        for i in range(nbars):
            cb_ax = self._fig.add_axes([0.895+(i+1)*0.01, 0.11, 0.01, 0.77/1.0])
            colorbar = mcolorbar.ColorbarBase(cb_ax, cmap=cmaps[i], norm=norm,
                                              orientation='vertical',
                                              format='%3.0f')
            if i < nbars-1:
                colorbar.set_ticks([])
                colorbar.ax.set_yticklabels([])
            cbs[i] = colorbar
        cbs[-1].set_label(title, fontsize=self._fontsize)                             


class plotElement():
    """A base class representing a plot element.  A plot element can be a 
    complex collection of more primitive matplotlib plot elements, but are 
    treated as a single element.
    
    Note:
        This class is not intended to be instantiated directly by the user, 
        rather it is inherited by child plot element objects.

    Note: 
        Inherited classes may have sub-elements that can have 
        different colors (and alphas). For those classes, refer to 
        their specifications as to the definition of :attr:`color`.
    """

    def __init__(self, color=None, alpha=None):
        """ Class constructor

        Args:
            alpha (float): The alpha opacity value, between 0 and 1.
            color (str): The color of the plot element
        """
        self._artists = []
        self._visible = True
        self._color = color
        self._alpha = alpha
        self._kwargs = None

    def __del__(self):
        """ Class destructor"""
        self.remove()

    @property
    def visible(self):
        """ (bool): True if the element is shown on the plot, False otherwise"""
        return self._visible

    @property
    def color(self):
        """ (str): color of the plot element"""
        return self._color

    @color.setter
    def color(self, color):
        """ Method for setting the plot element color

        Args:
            color (str): color of the plot element
        """
        [artist.set_color(color) for artist in self._artists]
        self._color = color

    @property
    def alpha(self):
        """ alpha (float): The alpha opacity value, between 0 and 1."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """ Method for setting the plot element color

        Args:
            alpha (float): The alpha opacity value, between 0 and 1.
        """
        [artist.set_alpha(alpha) for artist in self._artists]
        self._alpha = alpha

    def hide(self):
        """Hide the plot element"""
        self._change_visibility(False)

    def show(self):
        """Show the plot element"""
        self._change_visibility(True)

    def toggle(self):
        """Toggle the visibility of the plot element"""
        self._change_visibility(not self._visible)

    def remove(self):
        """Remove the plot element"""
        for artist in self._artists:
            try:
                artist.remove()
            except:
                pass

    def _sanitize_artists(self, old_artists):
        """Each artist collection is a collection of artists, and each artist
        is a collection of elements.  Matplotlib isn't exactly consistent
        on how each of these artists are organized for different artist 
        classes, so we have to have some sanitizing

        Args:
            old_artists (list): existing matplotlib artists
        """
        artists = []
        for artist in old_artists:
            if artist is None:
                continue
            elif isinstance(artist, (tuple, list)):
                if len(artist) == 0:
                    continue
                artists.extend(self._sanitize_artists(artist))
            else:
                artists.append(artist)
        return artists

    def _set_visible(self, artist, visible):
        """In some cases, the artist is a collection of sub-artists and the 
        set_visible() function has not been exposed to the artists, so 
        we must iterate.

        Args:
            artist (matplotlib.artist): artist class
            visible (bool): True if the element is shown on the plot, False otherwise.
        """
        try:
            for subartist in artist.collections:
                subartist.set_visible(visible)
        except:
            pass

    def _change_visibility(self, visible):
        """ Internal metho for updating visibility

        Args:
            visible (bool): True if the element is shown on the plot, False otherwise.
        """
        for artist in self._artists:
            try:
                artist.set_visible(visible)
            except:
                self._set_visible(artist, visible)
        self._visible = visible

class LightcurveBackground(plotElement):
    """Plot a lightcurve background model with an error band.

    Parameters:
        backrates (:class:`~gbm.background.BackgroundRates`):
            The background rates object integrated over energy.  If there is 
            more than one remaining energy channel, the background will be 
            integrated over the remaining energy channels.
        ax (:class:`matplotlib.axes`): The axis on which to plot
        cent_alpha (float, optional): The alpha of the background centroid line. 
                                      Default is 1
        cent_color (str, optional): The color of the background centroid line
        err_alpha (float, optional): The alpha of the background uncertainty. 
                                     Default is 1
        err_color (str, optional): The color of the background uncertainty
        
        color (str, optional): The color of the background. If set, overrides 
                               ``cent_color`` and ``err_color``.
        alpha (float, optional): The alpha of the background. If set, overrides 
                                 ``cent_alpha`` and ``err_alpha``
        **kwargs: Other plotting options

    Attributes:
        alpha (float): The alpha opacity value
        cent_alpha (float): The opacity of the centroid line
        cent_color (str): The color of the centroid line
        color (str): The color of the plot element.
        err_alpha (float): The opacity of the uncertainty band
        err_color (str): The color of the uncertainty band
        visible (bool): True if the element is shown on the plot, 
                        False otherwise    
    """

    def __init__(self, backrates, ax, color=None, alpha=None, cent_alpha=None,
                 err_alpha=None, cent_color=None, err_color=None, **kwargs):
        """ Class constructor

        Args:
            backrates (:class:`~gbm.background.BackgroundRates`):
                The background rates object integrated over energy.  If there is 
                more than one remaining energy channel, the background will be 
                integrated over the remaining energy channels.
            ax (:class:`matplotlib.axes`): The axis on which to plot
            color (str, optional): The color of the background. If set, overrides 
                                   ``cent_color`` and ``err_color``.
            alpha (float, optional): The alpha of the background. If set, overrides 
                                     ``cent_alpha`` and ``err_alpha``
            cent_alpha (float, optional): The alpha of the background centroid line. 
                                          Default is 1
            err_alpha (float, optional): The alpha of the background uncertainty. 
            cent_color (str, optional): The color of the background centroid line
                                        Default is 1
            err_color (str, optional): The color of the background uncertainty
            **kwargs: Other plotting options
        """
        super().__init__(color=color, alpha=alpha)
        self._kwargs = kwargs

        # color and alpha act as setting the global color values for the object
        if color is not None:
            cent_color = color
            err_color = color
        if alpha is not None:
            cent_alpha = alpha
            err_alpha = alpha

        self._cent_alpha = cent_alpha
        self._err_alpha = err_alpha
        self._cent_color = cent_color
        self._err_color = err_color
        artists = self._create(backrates, ax)
        self._artists = self._sanitize_artists(artists)

    @property
    def color(self):
        """ (str): color of the background"""
        return self._color

    @color.setter
    def color(self, color):
        """ color (str): color of the background"""
        [artist.set_color(color) for artist in self._artists]
        self._color = color
        self._cent_color = color
        self._err_color = color

    @property
    def alpha(self):
        """ (float): alpha of the background"""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """ alpha (float): alpha of the background"""
        [artist.set_alpha(alpha) for artist in self._artists]
        self._alpha = alpha
        self._cent_alpha = alpha
        self._err_alpha = alpha

    @property
    def cent_alpha(self):
        """ (float): alpha of the background centroid line"""
        return self._cent_alpha

    @cent_alpha.setter
    def cent_alpha(self, alpha):
        """ alpha (float): alpha of the background centroid line"""
        [artist.set_alpha(alpha) for artist in self._artists \
         if artist.__class__.__name__ == 'Line2D']
        self._cent_alpha = alpha

    @property
    def err_alpha(self):
        """ (float): alpha of the background uncertainty"""
        return self._err_alpha

    @err_alpha.setter
    def err_alpha(self, alpha):
        """ alpha (float): alpha of the background uncertainty"""
        [artist.set_alpha(alpha) for artist in self._artists \
         if artist.__class__.__name__ == 'PolyCollection']
        self._err_alpha = alpha

    @property
    def cent_color(self):
        """ (float): color of the background centroid line"""
        return self._cent_color

    @cent_color.setter
    def cent_color(self, color):
        """ color (float): color of the background centroid line"""
        [artist.set_color(color) for artist in self._artists \
         if artist.__class__.__name__ == 'Line2D']
        self._cent_color = color

    @property
    def err_color(self):
        """ (float): color of the background uncertainty"""
        return self._err_color

    @err_color.setter
    def err_color(self, color):
        """ color (float): color of the background uncertainty"""
        [artist.set_color(color) for artist in self._artists \
         if artist.__class__.__name__ == 'PolyCollection']
        self._err_color = color

    def _create(self, backrates, ax):
        """ Internal method to create the lightcurve plot

        Args:
            backrates (BackgroundRates): background rates object to plot
            ax (matplotlib.axes): axes where the plot is created
        """
        return lightcurve_background(backrates, ax,
                                     cent_color=self._cent_color,
                                     err_color=self._err_color,
                                     cent_alpha=self._cent_alpha,
                                     err_alpha=self._err_alpha, **self._kwargs)


def errorband(x, y_upper, y_lower, ax, **kwargs):
    """Plot an error band
    
    Args:
        x (np.array): The x values
        y_upper (np.array): The upper y values of the error band
        y_lower (np.array): The lower y values of the error band
        ax (:class:`matplotlib.axes`): The axis on which to plot
        **kwargs: Other plotting options
    
    Returns:
        list: The reference to the lower and upper selection    
    """
    refs = ax.fill_between(x, y_upper.squeeze(), y_lower.squeeze(), **kwargs)
    return refs

def lightcurve_background(backrates, ax, cent_color=None, err_color=None,
                          cent_alpha=None, err_alpha=None, **kwargs):
    """Plot a lightcurve background model with an error band
    
    Args:
        backrates (:class:`~gbm.background.BackgroundRates`):
            The background rates object integrated over energy. If there is more
            than one remaining energy channel, the background will be integrated
            over the remaining energy channels.
        ax (:class:`matplotlib.axes`): The axis on which to plot
        cent_color (str): Color of the centroid line
        err_color (str): Color of the errorband
        cent_alpha (float): Alpha of the centroid line
        err_alpha (fl): Alpha of the errorband
        **kwargs: Other plotting options
    
    Returns:
        list: The reference to the lower and upper selection    
    """
    times = backrates.time_centroids
    rates = backrates.rates
    uncert = backrates.rate_uncertainty
    p2 = errorband(times, rates + uncert, rates - uncert, ax, alpha=err_alpha,
                   color=err_color, linestyle='-', **kwargs)
    p1 = ax.plot(times, rates, color=cent_color, alpha=cent_alpha,
                 **kwargs)
    refs = [p1, p2]
    return refs


class TargetedLightcurves():
    """Class to make lightcurves for the targeted search
    
    Parameters:
    -----------
    data_dir: str
        The directory containing the BTTE/background data
    min_res: float, optional
        The minimum resolution of the data. Default is 64 ms
    lc_color: str, optional
        The color of the lightcurve. Default is #394264 (a dark blue)
    bkgd_color: str, optional   
        The color of the background. Default is firebrick.
    selection_color: str, optional
        The color of the event selection highlight. Default is #9a4e0e (orange).        
    fontsize: int, optional 
        The font size of the labels. Default is 12
    
    Public Methods:
    ---------------
    plot_channels:
        Multi-panel plot, each panel showing a channel, summed over detectors
    plot_detectors:
        Multi-panel plot, each panel showing a detector, summed over channels
    plot_summed:
        Single-panel plot, summed over channels and detectors
    search_plots:
        Create the full spread of search plots
    """
    def __init__(self, pha2_data, background_rates, t0, min_res=0.064, lc_color='#394264', 
                bkgd_color='firebrick', selection_color='#9a4e0e', fontsize=12):
        """ Class constructor

        Args:
            pha2_data (list): PHAII data for each detector
            background_rates (list): background rates for each detector
            min_res (float, optional): The minimum resolution of the data. Default is 64 ms
            lc_color (str, optional): The color of the lightcurve. Default is #394264 (a dark blue)
            bkgd_color (str, optional): The color of the background. Default is firebrick.
            selection_color (str, optional): The color of the event selection highlight. Default is #9a4e0e (orange).        
            fontsize (int, optional): The font size of the labels. Default is 12
        """ 
        self._t0 = t0
        self._lc_color=lc_color
        self._bkgd_color=bkgd_color
        self._sel_color=selection_color
        self._fontsize=fontsize
        self._fig = None
        self._axes = None
        self._min_res = min_res
        self.dpi = 150
        
        # load up data
        self._btte = pha2_data
        self._bkgd = background_rates
    
    def plot_detectors(self, time_res, out_file, event_time,
                       time_range=None, **kwargs):
        """Multi-panel plot, each panel showing a detector, summed over channels

        Args:
            time_res (float):
                Time resolution of the lightcurve.
                Must be a multiple of the resolution of the data
            out_file (str): The filename to be written to
            event_time (float, optional): The time of an event of interest
            time_range (tuple(2), optional):
                The time range of the data to be plotted.  If set, this overrides
                the automatically-determined time range.
            **kwargs:
                channel_range (tuple(2), optional):
                    The channel range of the data to be plotted
                energy_range (tuple(2), optional):
                    The energy range of the data to be plotted
        """        
        time_range = self._time_bounds(event_time, time_res, time_range)

        # initialize figure
        numdets = len(self._btte)
        self._init_fig(numdets, time_range)
        
        # plot each detector
        lcplots = np.empty(numdets, dtype=object)
        ebars = np.empty(numdets, dtype=object)
        bplots = np.empty(numdets, dtype=object)
        selects = np.empty(numdets, dtype=object)

        for i in range(numdets):

            # rebin the BTTE data, plot the lightcurve and errorbars
            lc = self._rebin_lc(self._btte[i], time_res, event_time, time_range,
                                **kwargs)
            lcplots[i] = Histo(lc, self._axes[i], color=self._lc_color)
            ebars[i] = HistoErrorbars(lc, self._axes[i], color=self._lc_color,
                                      alpha=0.5)

            # integrate the background over the energy range and plot
            b = self._integrate_bkgd(self._btte[i], self._bkgd[i], time_range=time_range, 
                                     **kwargs)
            bplots[i] = LightcurveBackground(b, self._axes[i], zorder=1000,
                                             cent_alpha=0.85, err_alpha=0.5, 
                                             color=self._bkgd_color)
            # if there is an event time, plot the highlight
            if event_time is not None:
                event = (event_time, event_time + time_res)
                selects[i] = self._axes[i].axvspan(*event, color=self._sel_color, 
                                                   alpha=0.2)

            # Set the y-axis limit
            lc = lc.slice(*time_range)
            self._axes[i].set_ylim(0.8*np.min(lc.rates), 1.2*np.max(lc.rates))

            # Get the detector name or create one if none exists
            if len(self._btte[i].detector) == 0:
                detector_name = 'Detector %s' % i
            else:
                detector_name = self._btte[i].detector

            # Annotate the plot
            self._annotate(self._axes[i], detector_name, (b.emin[0], b.emax[0]))
       
        if out_file is None:
            plt.show()
            return

        # save the figure
        try:
            plt.savefig(out_file, dpi=self.dpi, bbox_inches='tight')
        except ValueError as err:
            print(err)
        plt.close()
    
    def plot_channels(self, time_res, out_file, event_time, time_range=None, detector_subset=None, **kwargs):
        """Multi-panel plot, each panel showing a channel, summed over detectors

        Args:
            time_res (float):
                Time resolution of the lightcurve.
                Must be a multiple of the resolution of the data
            out_file (str): The filename to be written to
            event_time (float, optional): The time of an event of interest
            time_range (tuple(2), optional):
                The time range of the data to be plotted.  If set, this overrides
                the automatically-determined time range.
            detector_subset (list, optional):
                 A list of indices to select a subset of detectors to be plotted   
            **kwargs:
                channel_range (tuple(2), optional):
                    The channel range of the data to be plotted
                energy_range (tuple(2), optional):
                    The energy range of the data to be plotted
        """        
        time_range = self._time_bounds(event_time, time_res, time_range)

        btte = self._btte
        bkgd = self._bkgd

        # Select a subset of detectors
        if detector_subset is not None:
            btte = btte[detector_subset]
            bkgd = bkgd[detector_subset]
 
        # initialize figure
        spec = btte[0].to_spectrum(**kwargs)
        numchans = spec.size
        numdets = len(self._btte)
        chans = spec.centroids
        lo_edges = spec.lo_edges
        hi_edges = spec.hi_edges
        self._init_fig(numchans, time_range)
        
        # plot each channel, summing over detectors
        lcplots = np.empty(numchans, dtype=object)
        ebars = np.empty(numchans, dtype=object)
        bplots = np.empty(numchans, dtype=object)
        selects = np.empty(numchans, dtype=object)

        # Sum the background rates per channel from all the detectors
        bkgd_summed = self.sum_bkgds(bkgd)
        
        for i in range(numchans):

            # rebin the BTTE data, sum, then plot the lightcurve and errorbars
            lcs = [self._rebin_lc(one_btte, time_res, event_time, time_range, 
                   energy_range=(chans[i], chans[i])) for one_btte in btte]
            lc = lcs[0].sum(lcs)
            lcplots[i] = Histo(lc, self._axes[i], color=self._lc_color)
            ebars[i] = HistoErrorbars(lc, self._axes[i], color=self._lc_color,
                                      alpha=0.5)
            
            # Integrate the background over the energy range
            b_channel = bkgd_summed.integrate_energy(emin=chans[i]+1, emax=chans[i]-1)
        
            # Plot the channel specific background
            bplots[i] = LightcurveBackground(b_channel, self._axes[i], zorder=1000,
                                             cent_alpha=0.85, err_alpha=0.5, 
                                             color=self._bkgd_color) 
                                           
            # if there is an event time, plot the highlight
            if event_time is not None:
                event = (event_time, event_time + time_res)
                selects[i] = self._axes[i].axvspan(*event, color=self._sel_color, 
                                                   alpha=0.2)

            # Set the y-axis limit
            lc = lc.slice(*time_range)
            self._axes[i].set_ylim(0.8*np.min(lc.rates), 1.2*np.max(lc.rates))

            # Get the detector name or create one if none exists
            if len(btte[0].detector) == 0:
                detector_range = 'Detector %s - %s' % (0,numdets-1)
            else:
                detector_range = '%s - %s' % (btte[0].detector, btte[-1].detector)

            # Annotate the plot
            self._annotate(self._axes[i], detector_range, (b_channel.emin[0], b_channel.emax[0]))
        
        if out_file is None:
            plt.show()
            return

        # Save the figure
        try:
            plt.savefig(out_file, dpi=self.dpi*1.333, bbox_inches='tight')
        except ValueError as err:
            print(err)
        plt.close()
        
    def _init_fig(self, numplots, time_range, figsize=None):
        # initialize the figure
        if figsize is None:
            figsize = [12, numplots*2]
        fig, axes = plt.subplots(numplots, 1, sharex=True, sharey=False, 
                                 figsize=figsize)
        plt.subplots_adjust(hspace=0.06, wspace=0.15)
        
        try:
            axes[0]
        except:
            axes = [axes]
        
        # force plotting range to the time range, turn on ticks, and set labels
        [ax.set_xlim(time_range) for ax in axes]
        [ax.minorticks_on() for ax in axes]
        [ax.tick_params(axis='both', which='both', direction='in') for ax in axes]
        axes[-1].set_xlabel('Time (s) - {}'.format(self._t0), fontsize=self._fontsize)
        fig.text(0.06, 0.5, 'Count Rate (count/s)', fontsize=self._fontsize, 
                 va='center', rotation='vertical')
        self._fig = fig
        self._axes = axes
        
    def _time_bounds(self, event_start, duration, time_range):
        """Internal method for setting the time bounds of the lightcurve x-axis

        Args:
            event_start (float): start time of a potential astrophysical transient
            duration (float): duration of the transient in seconds
            time_range (tuple(2)): start and stop time used force a specific time range boundary

        Returns:
            (float, float): tuple with the start and stop times of the x-axis
        """
        if time_range is not None:
            return time_range
        if event_start is None:
            event_start = 0

        return (event_start - duration * 60.0, event_start + duration * 60.0)

    def _rebin_lc(self, btte, time_res, event_time, time_range, **kwargs):
        """ Method to rebin the BTTE data so that it is synced to the resolution and
        phase of the candidate.

        Args:
            btte (PHAII): binned TTE data for a detector
            time_res (float): duration of the candidate in seconds
            event_time (float): event time of the candidate in seconds
            time_range (tuple(2)): start and stop time used force a specific time range boundary

        Returns:
            (lightcurve): a rebinned lightcurve
        """
        if event_time is None:
            event_time = 0

        # BTTE resolution
        btte_res = btte.data.time_widths[1]

        # event duration is a multiple of the BTTE resolution
        bin_factor = int(round(time_res/btte_res))

        # padding around requested time range
        pad = 2 * time_res

        # bin indices
        bins = np.concatenate([btte.data.tstart, [btte.data.tstop[-1]]])
        ifirst = np.digitize(event_time + 0.5 * btte_res, bins) - 1 # first BTTE bin inside event duration
        istart = ifirst - int((event_time - time_range[0] + pad)/time_res) * bin_factor
        istop = ifirst + int((time_range[1] - event_time + pad)/time_res) * bin_factor

        # correct for boundary errors
        while istart < 0:
            istart += bin_factor
        while istop >= btte.data.tstop.size:
            istop -= bin_factor

        # define a rebin window that covers the time range but is within the data range
        tstart = btte.data.tstart[istart]
        tstop = btte.data.tstop[istop]

        # slice the BTTE in time and integrate over energy
        lc = btte.to_lightcurve(**kwargs, time_range=(tstart, tstop))

        # Do the rebin. Need padding to account for float rounding
        lc = lc.rebin(combine_by_factor, bin_factor, tstart=tstart, tstop=tstop)

        return lc
        
    def _integrate_bkgd(self, btte, bkgd, **kwargs):
        """ Internal method to integrate the background over energy channels (matching the BTTE
        channels that are plotted)

        Args:
            btte (PHAII): binned TTE data for a detector
            bkgd (BackgroundRates): fitted background rates for a detector
        """
        spec = btte.to_spectrum(**kwargs)
        b = bkgd.integrate_energy(*spec.range)
        return b

    def _annotate(self, ax, det, energy_range, summed_plot=False):
        """ Internal method for annotations of the detector name(s) and energy range shown

        Args:
            ax (matplotlib.axes): axes where the plot is shown
            det (str): detector name
            energy_range (tuple(2)): start and stop energies for the range shown
            summed_plot (bool): True when showing the sum from multiple detectors
        """
        if summed_plot == True:
            x = 0.015
            y = 0.95
        else:
            x = 0.01
            y = 0.87

        ax.annotate(det, xycoords='axes fraction', xy=(x,y), zorder=1000)
        ax.annotate('{0:3.0f}-{1:3.0f} keV'.format(*energy_range), xy=(1-x,y),
                    xycoords='axes fraction', horizontalalignment='right',
                    zorder=1000)

    def sum_bkgds(self, bkgds):
        """Sum multiple BackgroundRates together if they have the same time 
        range.  Example use would be summing two backgrounds from two detectors.
        
        Args:
            bkgds (list of :class:`BackgroundRates`):
                A list containing the BackgroundRates to be summed
        
        Returns:
            (:class:`BackgroundRates`)
        """
        rates = np.zeros_like(bkgds[0].rates)
        rates_var = np.zeros_like(bkgds[0].rates)
        for bkgd in bkgds:
            assert bkgd.num_times == bkgds[0].num_times, \
                "The backgrounds must all have the same support"
            rates += bkgd.rates
            rates_var += bkgd.rate_uncertainty ** 2
            
        ebounds = Ebounds.from_bounds(bkgds[0].emin, bkgds[0].emax)
        for bkgd in bkgds[1:]:
            # eb = Ebounds.from_bounds(bkgd.emin, bkgd.emax)
            # ebounds = Ebounds.merge(ebounds, eb)
            ebounds = Ebounds.from_bounds(bkgd.emin, bkgd.emax)  # <-- Need to be double checked

        # averaged exposure, sampling times
        exposure = np.mean([bkgd.exposure for bkgd in bkgds], axis=0)
        tstart = np.mean([bkgd.tstart for bkgd in bkgds], axis=0)
        tstop = np.mean([bkgd.tstop for bkgd in bkgds], axis=0)
        emin = ebounds.low_edges()
        emax = ebounds.high_edges()

        sum_bkgd = BackgroundRates(rates, np.sqrt(rates_var), tstart, tstop, emin, emax,
                       exposure=exposure)

        return sum_bkgd    

    def plot_summed(self, time_res, out_file, event_time,
                    time_range=None, **kwargs):
        """Single-panel plot, summed over channels and detectors

        Args:
            time_res (float):
                Time resolution of the lightcurve.
                Must be a multiple of the resolution of the data
            out_file (str): The filename to be written to
            event_time (float, optional): The time of an event of interest
            time_range (tuple(2), optional):
                The time range of the data to be plotted.  If set, this overrides
                the automatically-determined time range.
            **kwargs:
                channel_range (tuple(2), optional):
                    The channel range of the data to be plotted
                energy_range (tuple(2), optional):
                    The energy range of the data to be plotted
        """
        time_range = self._time_bounds(event_time, time_res, time_range)

        btte = self._btte
        bkgd = self._bkgd
        
        # initialize figure
        spec = btte[0].to_spectrum(**kwargs)
        numchans = spec.size
        numdets = len(self._btte)
        chans = spec.centroids

        self._init_fig(1, time_range, figsize=(12,6))
        ax = self._axes[0]
        
        # rebin the BTTE data, sum, then plot the lightcurve and errorbars
        lcs = [self._rebin_lc(one_btte, time_res, event_time, time_range, 
               **kwargs) for one_btte in btte]
        lc = lcs[0].sum(lcs)
        lcplot = Histo(lc, ax, color=self._lc_color)
        ebars = HistoErrorbars(lc, ax, color=self._lc_color, alpha=0.5)
    
        # Sum the background rates
        bkgd_summed = self.sum_bkgds(bkgd)

        # Integrate the background over the energy range, sum, and plot
        b = self._integrate_bkgd(btte[0], bkgd_summed, **kwargs)

        # # Integrate the background over the energy range, sum, and plot
        bplot = LightcurveBackground(b, ax, zorder=1000, cent_alpha=0.85, 
                                     err_alpha=0.5, color=self._bkgd_color)                    

        # if there is an event time, plot the highlight
        if event_time is not None:
            event = (event_time, event_time + time_res)
            select = ax.axvspan(*event, color=self._sel_color, alpha=0.2)

        # Set the y-axis limit
        lc = lc.slice(*time_range)
        ax.set_ylim(0.9*np.min(lc.rates), 1.1*np.max(lc.rates))

        # Get the detector name or create one if none exists
        if len(btte[0].detector) == 0:
            detector_range = 'Detector %s - %s' % (0,numdets-1)
        else:
            detector_range = '%s - %s' % (btte[0].detector, btte[-1].detector)

        # Annotate the plot
        self._annotate(ax, detector_range, (b.emin[0], b.emax[0]), summed_plot=True)

        if out_file is None:
            plt.show()
            return

        # save the figure
        try:
            plt.savefig(out_file, dpi=self.dpi, bbox_inches='tight')
        except ValueError as err:
            print(err)

        plt.close()
