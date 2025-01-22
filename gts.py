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

def runSearch(data, response, spacecraft_frames, t0, background_range, skyResolution=5, skymap=None, \
              settings=None, templates=None, templates_names=None, results_dir=None):
    """ Runs the targeted search near a trigger time of t0

    Args:
        data (Data): data object for all detectors
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
    timebins = data.getTimeBins(settings)
    
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
        tcenter = data.tcenters[index]
        durations = data.durations[index]
        
        # Get the spacecraft frame that is closest to this timebin
        spacecraft_frame = spacecraft_frames.at(t0 + tcenter * u.second)

        # Mask out the Earth from the response
        geo_azimuth, geo_zenith, geo_radius = utils.getGeoCoordinates(spacecraft_frame)
        earthmask = utils.createEarthMask(skyGrid._points, geo_azimuth, geo_zenith, geo_radius)
        masked_rsp = rsp[:,earthmask,:]

        # Format the data to optimize the search
        counts, background, background_error = data.formatDataForSearch(index)

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
            'skygrid': skyGrid, 'response': rsp, 'pha2_data': pha2_data, 'background': background_rates,
            'spacecraft_frames': spacecraft_frames}
