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
import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import utils
from likelihood import Likelihood

from scipy.interpolate import griddata

from astropy import units as u
from astropy.coordinates import get_sun, SkyCoord

from astropy.coordinates.representation import CartesianRepresentation
from gdt.core.data_primitives import Gti
from gdt.core.coords.quaternion import Quaternion
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.binning.binned import rebin_by_edge_index
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.plot.lightcurve import Lightcurve

from results import Results, UpperLimits
from plots import plot_orbit, Waterfall, TargetedLightcurves

def print_setting(description, value):
    """ Method to maintain a specific width when printing to screen

    Args:
       description (str): text to print
       value (str, float, or int): value to display
    """
    value = str(value)
    total_width = 40  # This is an example width and can be adjusted
    num_dots = total_width - len(description)
    description = description + '.' * num_dots
    
    formatted_str = "{:<40}{:>10}"
    print(formatted_str.format(description, value))

def loadResponse(rsp_file, templates=None, skyGrid=None, channels=None, detectors=None):
    """ Method to a load a given response file

    Args:
        rsp_file (str): path to response file
        templates (list): list of template indices to load. None will load all.
        skyGrid (list): list of indices for response points on the sky. None will load all.
        channels (list): list of energy channel indices to load. None will load all.
        detectors (list): list of detector indices to load. None will load all.

    Returns:
        np.ndarray: response matrix
    """
    response = np.load(rsp_file)

    n_templates = response.shape[0]
    n_skyGrid = response.shape[1]
    n_channels = response.shape[2]
    n_detectors = response.shape[3]

    if templates == None:
        templates = np.arange(n_templates)

    if skyGrid == None:
        skyGrid = np.arange(n_skyGrid)

    if detectors == None:
        detectors = np.arange(n_detectors)

    if channels == None:
        channels = np.arange(n_channels)

    # Fill the matrix
    response = response[templates, :, :, :]
    response = response[:, skyGrid, :, :]
    response = response[:, :, channels, :]
    response = response[:, :, :, detectors]

    return response

def preparePha2Data(tte_data, channel_edges, time_range=[-30, 30], t0=None, resolution=0.064):
    """ Function for preparing binned phaii data from time tagged events

    Args:
        tte_data (list): list of opened Tte data objects from a mission
        channel_edges (list): list of energy channel edges to use when binning data by energy index
        time_range (list): start and stop time used to select data around t0
        t0 (float or Time class): trigger time to use. Use trigtime of the Tte file when None.
        resolution (float): time resolution used when binning the Tte data in time.
                            This will set the minimum searchable duration of the search.

    Returns:
        list: list of PHAII data objects which represent instrument counts binned in energy as a function of time
    """
    # Create a list to contain the phaii product for each detector
    pha2_data = []

    # Make sure the channel edges is a numpy array
    channel_edges = np.array(channel_edges)

    # Loop through each TTE file and create a phaii file with the supplied channel edges
    for tte in tte_data:

        trigtime = tte.trigtime

        if trigtime is None and t0 is None:
            raise ValueError("t0 time is required when using continuous TTE files")
        if t0 is not None:
            # calculate offset to new trigger time
            offset = t0 if trigtime is None else t0 - trigtime
            # apply offset to event times
            tte.data._events['TIME'] -= offset
            # apply offset to good time interval bounds
            gti_start, gti_stop = np.transpose(tte.gti.as_list()) - offset
            tte._gti = Gti.from_bounds(gti_start, gti_stop)
            # update trigtime here but set it after rebin_energy to
            # avoid header mismatch in continuous tte files
            trigtime = t0

        # Bin the TTE data by time and energy
        phaii = tte.to_phaii(bin_by_time, resolution, time_ref=0, time_range=time_range)
        phaii = phaii.rebin_energy(rebin_by_edge_index, channel_edges)
        phaii._trigtime = trigtime

        pha2_data.append(phaii)

    return pha2_data

def fitBackgrounds(pha2_data, time_range=(-30, 30), verbose=True, plot=False):
    """ Method for performing a first order polynomial background fit

    Args:
        pha2_data (list): PHAII data objects for each detector
        time_range (tuple): tuple with start and stop time of the fit region
        verbose (bool): show fit statistic when True
        plot (bool): display a plot of the fit when True

    Returns:
        list: list of background rates objects returned from the fit
    """
    if verbose == True:
        print('\nFitting backgrounds...')
        print('Background fit selection: %s sec to %s sec' % (time_range[0], time_range[1]))
        print('\nStat/DOF:')

        print('--------------------------- Channels ---------------------------')

    # Create a list to contain the background rates for each detector
    background_rates = []

    for phaii in pha2_data:

        # Fit the data
        fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=[time_range])
        fitter.fit(order=1)

        if verbose == True:
            # Round the elements of the array
            goodness_of_fit = np.round(fitter.statistic/fitter.dof, 2)

            # Print the elements in a table format with consistent column spacing
            col_width = 7  # Adjust as needed for wider numbers
            formatted_strings = [f"{value:>{col_width}.2f}" for value in goodness_of_fit]
            print(" ".join(formatted_strings))

        # Get the closest time edge to the search window
        tstart_closest = phaii.data.closest_time_edge(time_range[0], which='low')
        tstop_closest = phaii.data.closest_time_edge(time_range[1], which='high')

        # Get the index of the closest values and pad that index by an additional bin
        index_start = np.abs(phaii.data.tstart - tstart_closest).argmin()
        index_stop = np.abs(phaii.data.tstop - tstop_closest).argmin()

        # Interpolate the fit over the search range
        tstarts = phaii.data.tstart[index_start:index_stop]
        tstops = phaii.data.tstop[index_start:index_stop]
        back_rates = fitter.interpolate_bins(tstarts, tstops)        

        # Save the background object
        background_rates.append(back_rates)

        # Plot the fit
        if plot == True:
            lightcurve = phaii.to_lightcurve()
            lcplot = Lightcurve(data=lightcurve)
            lcplot.set_background(back_rates)
            plt.xlim(*time_range)
            plt.show()

    background_rates = np.array(background_rates)

    return background_rates

def getBackgrounds(background_rates, timebin, channels=None):
    """ Retrieve background counts computed over a specific time bin

    Args:
        background_rates (list): list of background rates objects for each detector
        timebin (tuple): tuple with (bin start time, bin duration)
        channels (list): list of channel indices to use when selecting a subset of detectors

    Returns:
        (np.ndarray, np.ndarray): tuple with arrays of background counts and their uncertainties for each detector
    """
    tstart = timebin[0]
    duration = timebin[1]
    tstop = tstart + duration
    time_range = np.array([tstart, tstart + duration])

    # Determine the number of detectors
    n_detectors = len(background_rates)

    # Determine the number of channels 
    if channels is None:
        n_channels = len(background_rates[0].chan_widths)

    # Create an array to contain the background data
    backgrounds = np.zeros((n_channels, n_detectors))
    background_uncertainties = np.zeros((n_channels, n_detectors))

    for index in range(len(background_rates)):

        # Produce an background object that is integrated over the entire time slice
        background_rate = background_rates[index]
        background_rate_integrated = background_rate.integrate_time(tstart=tstart, tstop=tstop)

        # Eztract arrays of background counts and background counts uncertainty per channel
        background = background_rate_integrated.counts
        background_uncertainty = background_rate_integrated.count_uncertainty

        if isinstance(background_rate.count_uncertainty[0], np.float64):
            background_uncertainty = background_uncertainty.reshape(-1,1)

        # Fill the background and background uncertainy arrays
        backgrounds[:,index] = background[channels]
        background_uncertainties[:,index] = background_uncertainty[channels]

    return backgrounds, background_uncertainties

def getCounts(pha2_data, timebin, channels=None):
    """ Retrieve observed counts computed over a specific time bin

    Args:
        pha2_data (list): list of PHAII data objects for each detector
        timebin (tuple): tuple with (bin start time, bin duration)
        channels (list): list of channel indices to use when selecting a subset of detectors

    Returns:
        np.ndarray: array of counts for each detector
    """

    # Get the time bin information
    tstart = timebin[0]
    duration = timebin[1]
    tstop = tstart + duration
    time_range = np.array([tstart, tstop])

    # Determine the number of detectors and channels
    n_detectors = len(pha2_data)

    if channels is None:
        n_channels = len(pha2_data[0].data.chan_widths)

    # Create an array to contain the count data
    counts = np.zeros((n_channels, n_detectors))

    # Loop through each pha2 file and extract and record the number of counts in the time bin
    for index in range(len(pha2_data)):

        # Integrate the phaii data over time to produce a count spectrum
        phaii = pha2_data[index]
        channel_counts = phaii.to_spectrum(time_range=time_range).counts

        # Fill the counts array
        counts[:,index] = channel_counts[channels]

    return counts
    
def getTimeBins(pha2_data, settings):
    """ Calculate the time bins used in the search. These represent the different
    emission durations of the search shifted across the full search range using
    a given step size.

    Args:
        pha2_data (list): PHAII data objects for each detector
        settings (dict): search settings for duration range, search range, and step size

    Returns:
        list: list with values for the start times and durations of each search bin
    """
    win_width = settings['win_width']
    min_dur = settings['min_dur']
    max_dur = settings['max_dur']
    min_step = settings['min_step']
    num_steps = settings['num_steps']

    search_range = (-win_width/2.0, win_width/2.0)

    # Durations to search in powers of two
    log2maxdur = np.round(np.log2(max_dur))
    log2mindur = np.round(np.log2(min_dur))
    durations = 1.024 * 2. ** np.arange(log2mindur, log2maxdur + 1, 1)

    # Get one of the phaii files to determine the proper data binning
    phaii = pha2_data[0]

    # The search bins before 0
    tstart1 = phaii.data.slice_time(search_range[0] - durations.max()/2.0, 0).tstart
    timebins1 = []
    if len(tstart1):
        timebins1 = ((t, dur) for dur in durations \
                     for t in np.arange(0, tstart1[0], \
                                        -max(min_step, dur/num_steps)) \
                     if t >= search_range[0]-dur/2.0)     

    # The search bins after 0, inclusive
    tstart2 = phaii.data.slice_time(0, search_range[1]).tstart
    timebins2 = []
    if len(tstart2):
        timebins2 = ((t, dur) for dur in durations 
                     for t in np.arange(0, tstart2[-1], \
                                        max(min_step, dur/num_steps)) \
                     if t+dur/2.0 <= search_range[-1])        

    # Combine the search windows. Format: (tstart, duration)
    timebins = sorted(timebins1)
    timebins.extend(sorted(timebins2))

    return timebins

def smallSkyMapCorrection(skymap, skyResolution):
    """ Correction technique used for cases where the resolution of the
    detector response is larger than a provided external localization probability map

    Args:
        skymap (HealPix derived class): the localization probability map
        skyResolution (float): resolution of the detector response grid on the sky

    Returns
        (np.ndarray, np.ndarray): tuple with pixel indices and values for non-zero pixels
                                  in the provided localization probability map
    """
    # Calculate the total area of the pixels with nonzero probability
    nonzero = skymap.prob > 0.
    nonzero_area = (nonzero * skymap.pixel_area).sum()

    # Use the small map correction if the nonzero area is smaller than a single pixel area
    if nonzero_area <= (2 * skyResolution)**2:
        small_map_idx = np.arange(skymap.prob.size)[nonzero]
        small_map_prob = skymap.prob[nonzero]
        return small_map_idx, small_map_prob
    return None, None

def calculateSnr(counts, background, min_channel=0, max_channel=-1):
    """ Method for calculated the top two signal-to-noise ratios from all detectors.
    Computed using a Gaussian approximation.

    Note: this should move to the results or filter class. Not used in search.

    Args:
        counts (np.ndarray): counts in each detector
        background (np.ndarray): background in each detector
        min_channel (int): minimum energy bin index to sum
        max_channel (int): maximum energy bin index to sum

    Returns:
        (float, float): tuple with the highest & second highest signal-to-noise ratios from all detectors
    """

    # Calculate the signal to noise ratio (SNR)
    snr = (counts[min_channel:max_channel,:] - background[min_channel:max_channel,:]).sum(axis=0) / \
              np.sqrt(background[min_channel:max_channel,:].sum(axis=0))

    # Get the inndividual detector SNR and top 2 SNR measurements
    snr1, snr0 = np.sort(snr)[-2:]

    return (snr1, snr0)

def phosphorescenceVeto(counts, background, background_error):
    """Get statistics for cosmic-ray post-veto

    Phosphorescence events should be:
        1) isolated to one detector,
        2) soft primarily channel 0

    Therefore we calculate the signal-to-noise ratio (SNR) for each channel in each detector
    and compare the SNR of channel 0 and 1 in the detector that yeilds the max signal

    Note: this should move to the results or filter class. Not used in search.

    Args:
        counts (np.ndarray): counts in each detector
        background (np.ndarray): background in each detector
        min_channel (int): minimum energy bin index to sum
        max_channel (int): maximum energy bin index to sum

    Returns:
        (float, float, float): tuple with (highest channel 0 SNR,
                               second highest channel 0 SNR,
                               channel 1 SNR for detector with highest channel 0 SNR)
    """

    # Calculate the signal to noise ratio for each channel in each detectors
    snr = (counts-background)/np.sqrt(background+background_error)

    # Top 2 detectors for low channel signal to noise ratio
    (i, j) = np.argsort(snr[0, :])[-2:]

    # SNR of max detector channel 0, 
    pe_veto1 = snr[0, j]
    # 
    # ratio of max chan0 to next-max, 
    pe_veto2 = snr[0, i]

    # ratio of chan0 to chan1
    pe_veto3 = snr[1, j]

    # Package it all up
    pe_veto = (pe_veto1, pe_veto2, pe_veto3)

    return pe_veto
    
def skyPrior(grid, spacecraft_frame, small_map_prob=None, skymap=None):
    """ Calculate the sky prior given a map, or do uniform prior, in the spacecraft frame.
    The prior is in equatorial, so we need to rotate it to spacecraft.

    Args:
        grid (np.ndarray): grid of sky locations used in the instrument response
        spacecraft_frame (Frame): frame object with information about spacecraft position
        small_map_prob (np.ndarray): use existing small skymap projection when not None
        skymap (HealPix class): localization probability to use as the prior. Use uniform prior when None.

    Returns:
        np.ndarray: the sky prior in the spacecraft frame
    """
    if small_map_prob is not None:

        # Small map case is already projected into spacecraft coordinates
        skyprior = small_map_prob

    elif skymap is not None:
        
        # Get the azimuth and zenith of each unmasked sky grid position
        azimuth, zenith = grid

        # Get the equivelent RA and Dec of each unmasked sky grid position
        coords = SkyCoord(azimuth, 0.5 * np.pi - zenith, frame=spacecraft_frame, unit='rad')
        ra = coords.icrs.ra
        dec = coords.icrs.dec

        # Calculate the probability of each sky position
        # For now, do explicit lookup with ang2pix to avoid GDT interpolation of values.
        # We need to use exact values to ensure consistency between multiorder vs single resolution map formats.
        ph, th = ra.rad, 0.5 * np.pi - dec.rad
        pix = hp.ang2pix(skymap.nside, th, ph)
        skyprior = (skymap.prob / skymap.pixel_area)[pix]

    else:
        skyprior = np.ones(len(grid[0]), np.float64)

    # Ensure we're normalized to 1
    skyprior /= skyprior.sum() 
    logskyprior = np.log(np.maximum(1e-100, skyprior))

    return logskyprior
    
def getSpacecraftFrame(spacecraft_frames, t0, tcenter):
    """ Method for returning spacecraft frame at the center of a bin.

    Note: we currently apply compute the bin center relative to t0.
    We can remove this in favor of tcenter after updating GDT to add
    more consistent time handling between TTE files and other classes.

    Args:
        spacecraft_frames (list?): objext with spacecraft frames for interpolation
        t0 (Time): trigger time of the search, given as a Time object
        tcenter: central time of a search bin relative to t0

    Returns:
        Frame: spacecraft frame at t0 + tcenter
    """
    # Find the frame closest to the specified time
    index_frame= np.abs(spacecraft_frames.obstime.value - (t0 + tcenter)).argmin()
    spacecraft_frame = spacecraft_frames[index_frame]

    return spacecraft_frame

def getSunAngle(coordinate_max, t0):
    """ Calculates the sun angle relative to a location.

    Note: this could probably move to the results class.

    Args:
        coordinate_max (SkyCoord): location of maximum likelihood
        t0 (Time): time used to retrieve sun location

    Returns:
        float: angular separation to the sun in degrees
    """
    if t0 is not None:
        sun_coord = get_sun(t0)
        sun_angle = sun_coord.separation(coordinate_max)[0]
    else:
        sun_angle = None

    return sun_angle

def findLocationOfMaxLikelihood(skyGrid, like, spacecraft_frame):
    """ Calculates the location on the sky that maximizes the likelihood.

    Args:
        skyGrid (SkyGrid): object defining the detector response coordinates on the sky
        like (Likelihood): the likelihood method class
        spacecraft_frame (Frame): frame with spacecraft position information

    Returns:
        SkyCoord: spacecraft frame coordinates for the location that maximizes the likelihood
    """
    # Get the azimuth and zenith of the position that yeilds the maximum marginal likelihood
    azimuth_max, zenith_max = skyGrid._points[:,like.max_location]

    # Get the RA and Dec of the position that yeilds the maximum marginal likelihood
    coordinate_max = SkyCoord(azimuth_max, 0.5 * np.pi - zenith_max, frame=spacecraft_frame, unit='rad')
    
    # return ra_max, dec_max
    return coordinate_max

def formatDataForSearch(counts, background, background_error):
    """ Formats the data as 1D vectors for the likelihood method

    Args:
        counts (np.ndarray): observed counts for all detectors and energy bins
        background (np.ndarray): background counts for all detectors and energy bins
        background_error (np.ndarray): error on background counts for all detectors and energy bins

    Returns:
        (counts, background, background_error): the input arrays formatted as 1D vectors
    """
    # Flatten the arrays
    counts = counts.ravel()
    background = background.ravel()
    background_error = background_error.ravel()

    return counts, background, background_error

def saveUpperLimits(pflux_weighted, results, upperlimit_map_pflux, upperlimit_sigma, upperlimit_durations, template_names, resultsDirectory="./"):
    """ Method for saving upper limit values to a file. Currently unused. Need to separate prior-averaged from per-location upper limits.

    Args:
        pflux_weighted (np.ndarray): array with prior-averaged flux and flux error for all upper limit durations
        results (Results): object with search results
        upper_limit_map_pflux (np.ndarray): array with per-location upper limits for all upper limit durations
        upper_limit_sigma (float): confidence level of the upper limit in standard deviations
        upper_limit_durations (list): list of durations over which the upper limits were computed
        template_names (list): list of spectral templates used to compute upper limits
        resultsDirectory (str): path to results directory for file output
    """
    # Create the the results object
    resultsObj = Results.create(results, templates=template_names)

    # Create the upper limits object
    uppers = UpperLimits(pflux_weighted[:,:,0], pflux_weighted[:,:,1], 
                            resultsObj.times, resultsObj.durations, resultsObj.templates, 
                            ul_map=upperlimit_map_pflux,
                            ul_map_sigma=upperlimit_sigma,
                            ul_map_durations=upperlimit_durations)

    uppers.save(resultsDirectory, filename='pflux_upper_limits.npz')

def filterResults(resultsObj, settings):
    """ Ranks results and selects the top candidates

    Note: this should be moved outside the search to its own class

    Args:
        resultsObj (Results): the raw search results
        settings (dict): search settings

    Returns:
        Results: results object ranked by log_lr above a minimum value with overlapping candidates removed.
    """
    filtered_results = resultsObj.downselect(threshold=settings['min_loglr'], no_empty=True)
    filtered_results = filtered_results.downselect(combine_spec=False, fixedwin=settings['win_width'])

    return filtered_results

def createLocalization(tcenter, duration, template, search, cls, systematic, remove_earth, nside_proj=64, nside_out=128):
    """ Calculates the localization probility on the sky for a search result

    Args:
        tcenter (float): central of the search candidate relative to t0 of the search
        duration (float): duration of the search candidate
        template (int): index of best-fit spectral template for the search candidate
        search (dict): dictionary returned by the runSearch() method
        cls (HealPix derived): a healpix localization class for storing the localization
        systematic (tuple): tuple with (systematic_method, args) for use in the convolve() step
        remove_earth (bool): remove the area of the sky blocked by the Earth from the probability space
        nside_proj (int): nside value for projecting between response grid and healpix format
        nside_out (int): nside value of the final localization probability map

    Returns:
        HealPix: a HealPix-derived class with the localization probability
    """
    # Reconstruct the search timebin 
    tstart = tcenter - 0.5 * duration
    timebin = (tstart, duration)

    # Get the closest spacecraft frame
    spacecraft_frame = search['spacecraft_frames'].at(search['t0'] + tcenter * u.second)

    # Extract the counts and background data from the phaii data for this specific timebin
    counts = getCounts(search['data'], timebin)
    background, background_error = getBackgrounds(search['background'], timebin)
        
    # Re-calculate the likelihood object for this timebin
    like = Likelihood(search['response'].shape[0], search['skygrid'].size)
    like.calculate(counts, background, background_error, search['response'])

    # probability assuming Wilks' theorem (likelihood approximates -2x chi-square distribution)
    prob = np.exp(like.llr - np.max(like.llr))

    # project to a healpix grid
    proj_prob, _ = utils.grid2healpix(
        prob[template,:], search['skygrid']._points,
        spacecraft_frame, nside_out=nside_proj)

    # upscale to desired resolution with interpolation
    hires_npix = hp.nside2npix(nside_out)
    theta, phi = hp.pix2ang(nside_out, np.arange(hires_npix))
    upscaled_prob = hp.get_interp_val(proj_prob, theta, phi)

    # create localization object
    loc = cls.from_data(upscaled_prob, trigtime=search['data'][0].trigtime + tcenter,
                        quaternion=spacecraft_frame.quaternion, scpos=spacecraft_frame.obsgeoloc)
    if systematic is not None:
        loc = loc.convolve(*systematic, quaternion=spacecraft_frame.quaternion, scpos=spacecraft_frame.obsgeoloc)
    if remove_earth:
        loc = loc.remove_earth()

    return loc

def runSearch(pha2_data, response, spacecraft_frames, t0, background_range, skyResolution=5, skymap=None, \
              settings=None, templates=None, templates_names=None, results_dir=None):
    """ Runs the targeted search near a trigger time of t0

    Args:
        pha2_data (list): PHAII data objects for each detector
        reponse (np.ndarray): instrument response matrix for all detectors
        spacecraft_frames (list?): objext with spacecraft frames for interpolation
        t0 (Time): trigger time of the search
        background_range (list): start and stop time of the background fit
        skyResolution (float): resolution of the sky locations used in the response matrix
        skymap (HealPix): an external localization probabilty for use as a sky prior weight
        settings (dict): dictionary with values of adjustable search settings
        templates (list): spectral template indices to use in the search
        results_dir (str): path to directory for results output

    Returns:
        dict: Dictionary with results information needed to create localizations
    """
    # Define the results directory
    if results_dir is None:
        results_dir = "."

    if templates is None:
        templates = np.arange(response.shape[0])

    if templates_names is None:
        template_names = templates.astype(str)

    if settings is None:

        print('\nUsing default search parameters:\n')

        # Initialize the default settings dictionary if one is not supplied
        settings = {}   
        settings['win_width'] = 5           # Window around T0 to search
        settings['min_dur'] = 0.064         # Minimum search duration
        settings['max_dur'] = 8.192         # Maximum search duration
        settings['min_step'] = 0.064        # Minimum phase shift
        settings['num_steps'] = 8           # Number of phase shifts
        settings['resolution'] = 0.512      # Unused. Need to implement sliding window background first.
        settings['min_loglr'] = 5         # Minimum loglr to produce plots

    else:

        print('Using custom search parameters:\n')

    print('Window around T0 to search:\t %s sec' % settings['win_width'])
    print('Minimum search duration:\t %s sec' % settings['min_dur'])
    print('Maximum search duration:\t %s sec' % settings['max_dur'])
    print('Minimum phase step:\t\t %s sec' % settings['min_step'])
    print('Number of phase steps:\t\t %s' % settings['num_steps'])
    print('Minimum loglr to produce plots:\t %s' % settings['min_loglr'])

    # Calculate the log of the min and max durations
    log2maxdur = np.round(np.log2(settings['min_dur']))
    log2mindur = np.round(np.log2(settings['max_dur']))
    durations = 1.024 * 2. ** np.arange(log2mindur, log2maxdur + 1, 1)

    # Upper limit settings
    upperlimit_durations = []
    upperlimit_templates = template_names
    upperlimit_sigma = 3.0
    upperlimit_nside = 16

    # Define the time range over which the background is fit and the plots are made
    search_range = np.array([-0.5, 0.5]) * settings['win_width']

    # Generate the timebins to search
    timebins = getTimeBins(pha2_data, settings)   # Format: (tstart, duration
    n_timebins = len(timebins)

    # Get the number of spectral templates
    n_templates = len(templates)

    # Generate the sky grid
    skyGrid = utils.SkyGrid(skyResolution)
    n_skygrid = skyGrid.size

    # Reshape the response matrix
    rsp = response.reshape(n_templates, n_skygrid, -1)

    # Initilize a results array
    results = np.zeros((n_timebins, 23))

    # Fit the backgrounds and return the background objects for each detector
    background_rates = fitBackgrounds(pha2_data, time_range=background_range)

    # Determine if the supplied skymap covers a very small region of the sky
    if skymap is not None:

        # Determine if a small sky map correction needs to be applied
        small_map_idx, small_map_prob = smallSkyMapCorrection(skymap, skyResolution)

    else:
        small_map_prob = None
        skymap = None

    # Create a map to store  upper limit information
    pflux_weighted = np.zeros((n_timebins, n_templates, 2))
    if len(upperlimit_durations):
        shape = (len(upperlimit_durations), n_templates, hp.nside2npix(upperlimit_nside))
        upperlimit_map_pflux = np.full(shape, hp.UNSEEN)

    print('\nRunning search...')
    import time
    t2 = time.time()

    # Loop through each timebin
    for index in range(n_timebins):

        # Get the center of the timebin and the bin duration
        timebin = timebins[index]
        tstart = timebin[0]
        duration = timebin[1]
        tcenter = tstart + duration / 2.0

        # Extract the counts and background data from the phaii data for this specific timebin
        counts = getCounts(pha2_data, timebin)
        background, background_error = getBackgrounds(background_rates, timebin)

        # Calculate the signal to noise ratio (SNR)
        snr = calculateSnr(counts, background)

        # Calculate the phosphorescence veto
        pe_veto = phosphorescenceVeto(counts, background, background_error)

        # Get the spacecraft frame that is closest to this timebin
        spacecraft_frame = spacecraft_frames.at(t0 + tcenter * u.second)

        # Mask out the Earth from the response
        geo_azimuth, geo_zenith, geo_radius = utils.getGeoCoordinates(spacecraft_frame)
        earthmask = utils.createEarthMask(skyGrid._points, geo_azimuth, geo_zenith, geo_radius)
        masked_rsp = rsp[:,earthmask,:]

        # Format the data to optimize the search
        counts, background, background_error = formatDataForSearch(counts, background, background_error)

        # Initilize the likelihood object and perform the calculation
        like = Likelihood(n_templates, skyGrid.size)
        like.calculate(counts, background, background_error, masked_rsp)

        # Find the sky position that yielded the highest signal significance
        coords_max = findLocationOfMaxLikelihood(skyGrid, like, spacecraft_frame)

        # Get the angle between the max position and the Earth and Sun
        geo_angle = spacecraft_frame.geocenter.separation(coords_max)[0]
        sun_angle = getSunAngle(coords_max, t0)

        # Projections needed for the small skymaps
        if small_map_prob is not None:

            # Calculate map pixels in spacecraft coord
            theta, phi = hp.pix2ang(skymap.nside, small_map_idx)
            grid = np.array((phi, 0.5 * np.pi - theta))

            # Create an Earth mask
            earthmask_small = utils.createEarthMask(grid, geo_azimuth, geo_zenith, geo_radius)
            n_visible_pixels = earthmask_small.sum()

            # Likelihood results that need to be projected
            llratio = np.zeros((n_timebins, n_visible_pixels), np.float64)
            pflux = np.zeros((n_timebins, n_visible_pixels), np.float64)
            pflux_sig = np.zeros((n_timebins, n_visible_pixels), np.float64)

            for j in range(n_templates):
                llratio[j] = griddata(tuple(skyGrid._points[:,earthmask]), like.llr[j], tuple(grid[:,earthmask_small]), 'nearest')
                pflux[j] = griddata(tuple(skyGrid._points[:,earthmask]), like._pflux[j], tuple(grid[:,earthmask_small]), 'nearest')
                pflux_sig[j] = griddata(tuple(skyGrid._points[:,earthmask]), like._pflux_sig[j], tuple(grid[:,earthmask_small]), 'nearest')

            # Masked version of small_map_prob
            small_map_earthmask = small_map_prob[earthmask_small]

        else: 

            # Use unprojected values
            llratio = like.llr
            pflux = like._pflux
            pflux_sig = like._pflux_sig
            small_map_earthmask = None

        # Apply sky prior
        log_sky_prior = skyPrior(skyGrid._points[:,earthmask], spacecraft_frame, small_map_earthmask, skymap)
        coinclr = like.coinclr(log_sky_prior, llratio=llratio)

        # Placeholder for now
        in_rock = 0

        # convert to degrees for results storage
        ra_max = coords_max.icrs.ra[0].deg
        dec_max = coords_max.icrs.dec[0].deg
        azimuth_max = coords_max.az.deg
        zenith_max = 90.0 - coords_max.el.deg

        # Collect the search results
        results[index,:] = [tcenter, duration, 1, in_rock, like.status, azimuth_max, zenith_max, ra_max, dec_max, like.max_template, 
                like.photon_fluence/duration, like.optimal_snr, *snr, *like.chisq, sun_angle.deg, geo_angle.deg, like.marginal_llr,
                coinclr, *pe_veto]

        # Calculate upper limits
        if duration in upperlimit_durations:
            values = (like._pflux + upperlimit_sigma * like._pflux_sig) / duration
            coords = (skyGrid._points[0,earthmask], skyGrid._points[1,earthmask])

            # Get the flux upper limits
            pflux_ul, pix = utils.grid2healpix(values, coords, spacecraft_frame, nside_out=upperlimit_nside)

            # Keep only the largest upper limits found during the search for each spectral template
            imap = upperlimit_durations.index(duration)
            for ispec in range(n_templates):
                mask = pflux_ul[ispec] > upperlimit_map_pflux[imap,ispec,pix]
                upperlimit_map_pflux[imap,ispec,pix[mask]] = pflux_ul[ispec][mask]
            
        # Photon flux weighted over the sky prior
        pflux_weighted[index,:,:] = like.prior_weighted_fluence(log_sky_prior, pflux=pflux, pflux_sig=pflux_sig).T / duration
        
        # Write out the search progress
        sys.stdout.write("Progress: %d%%   \r" % ((index/len(timebins)) * 100) )
        sys.stdout.flush()

    # Signal the completion of the search
    print('\nDone.')
    print("\nSearch completed in %.1f seconds." % (time.time() - t2))

    # Save the photon flux upper limits
    #saveUpperLimits(pflux_weighted, results, upperlimit_map_pflux, upperlimit_sigma, upperlimit_durations, template_names)

    # Return the results object as well as other information from the search
    return {'results': Results.create(results, templates=templates), 't0': t0, 'skymap': skymap,
            'skygrid': skyGrid, 'response': rsp, 'data': pha2_data, 'background': background_rates,
            'spacecraft_frames': spacecraft_frames}
