# Copyright 2017-2022 by Universities Space Research Association (USRA). All rights reserved.
#
# Developed by: William Cleveland, Adam Goldstein, and Alex Goberna
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
from likelihood import Likelihood

from gdt.core.phaii import Phaii
from gdt.missions.fermi.time import Time

import numpy as np

from astropy.coordinates import get_sun, SkyCoord


# TODO Remove from search
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
    azimuth_max, zenith_max = skyGrid._points[:, like.max_location]

    # Get the RA and Dec of the position that yeilds the maximum marginal likelihood
    coordinate_max = SkyCoord(azimuth_max, 0.5 * np.pi - zenith_max, frame=spacecraft_frame, unit='rad')

    # return ra_max, dec_max
    return coordinate_max


# TODO Remove from search
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

    counts = counts.reshape(-1, 14)
    background = background.reshape(-1, 14)

    # Calculate the signal to noise ratio (SNR)
    snr = (counts[min_channel:max_channel,:] - background[min_channel:max_channel,:]).sum(axis=0) / \
              np.sqrt(background[min_channel:max_channel,:].sum(axis=0))

    # Get the inndividual detector SNR and top 2 SNR measurements
    snr1, snr0 = np.sort(snr)[-2:]

    return (snr1, snr0)

# TODO Remove from search
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


# TODO Remove from search
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

    counts = counts.reshape(-1, 14)
    background = background.reshape(-1, 14)
    background_error = background_error.reshape(-1, 14)

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


class TargetedScanner():
    """Class for the npy results files

        Attributes:
        -----------

        Public Methods:
        ---------------

        Class Methods:
        ---------------
        """
    def __init__(self, instrument_data, search_configuration, skygrid):
        self.instrument_data = instrument_data
        self.search_configuration = search_configuration
        self.skygrid = skygrid

    def get_bin_starts(self, search_range, durations):
        reference_instrument = self.search_configuration.reference_instrument
        reference_data = self.instrument_data[reference_instrument].counter.data

        tstart = None
        tend = None

        # TODO Irrelevant to PHAII or TTE, align using tstart from data (for search range, not bin alignment)
        # NOTE: Above concern may already be addressed

        for data in reference_data:
            if (isinstance(data, Phaii)):
                tstart1 = data.data.slice_time(search_range[0] - durations.max() / 2.0, 0).tstart[0]
                tend1 = data.data.slice_time(0, search_range[1]).tstart[-1]
                if tstart and tstart1 != tstart:
                    print('Warning, PHAII time bins across reference instrument detectors do not match')
                else:
                    tstart = tstart1
                if tend and tend1 != tend:
                    print('Warning, PHAII time bins across reference instrument detectors do not match')
                else:
                    tend = tend1
            else:
                print('Ignoring unbinned detector data for time bin generation')

        if not tstart:
            tstart = search_range[0]
        if not tend:
            tend = search_range[1]

        return tstart, tend


    def get_timebins(self, t0=None):
        """ Calculate the time bins used in the search. These represent the different
        emission durations of the search shifted across the full search range using
        a given step size.

        Returns:
            list: list with values for the start times and durations of each search bin
        """
        search_configuration = self.search_configuration
        win_width = search_configuration.win_width
        min_dur = search_configuration.min_dur
        max_dur = search_configuration.max_dur
        min_step = search_configuration.min_step
        num_steps = search_configuration.num_steps

        search_range = (-win_width / 2.0, win_width / 2.0)

        # Durations to search in powers of two
        log2maxdur = np.round(np.log2(max_dur))
        log2mindur = np.round(np.log2(min_dur))
        durations = 1.024 * 2. ** np.arange(log2mindur, log2maxdur + 1, 1)

        tstart, tend = self.get_bin_starts(search_range, durations)

        # The search bins before t0
        timebins1 = [(t, dur) for dur in durations for t in np.arange(0, tstart, -max(min_step, dur / num_steps)) if t >= search_range[0] - dur / 2.0]

        # The search bins after t0, inclusive
        timebins2 = [(t, dur) for dur in durations for t in np.arange(0, tend, max(min_step, dur / num_steps)) if t + dur / 2.0 <= search_range[-1]]

        # Combine the search windows. Format: (tstart, duration)
        timebins = sorted(timebins1)
        timebins.extend(sorted(timebins2))

        timebins = self._align_timebins(timebins)

        return timebins


    # Combine counts, backgrounds, responses into one matrix each for input to Likelihood
    def stack_instrument_outputs(self, instrument_outputs):
        if len(instrument_outputs.keys()) > 1:
            print('Multi-instrument search not currently supported')
        else:
            vals = list(instrument_outputs.values())[0]
            return vals['counts'], vals['background_rates'], vals['background_variance'], vals['response']


    def calculate_timebin_likelihood(self, tstart, tstop, t0):
        # Note: Where to store n_templates? energybins?
        n_templates = 3
        n_energybins = 8

        duration = tstop - tstart

        reference_instrument = self.search_configuration.reference_instrument
        reference_frame = self.instrument_data[reference_instrument].get_spacecraft_frame((tstart + tstop) / 2)
        outputs = {}
        num_sky_positions = self.skygrid.size
        for instrument in self.instrument_data.keys():
            instrument_data = self.instrument_data[instrument]
            instrument_config = self.search_configuration.get_instrument_config(instrument)
            if instrument == reference_instrument:
                channel_mask = instrument_config.channel_mask

                counts, exposure = instrument_data.counts(tstart, tstop)

                bkgd_rates, bkgd_variance, good = instrument_data.background_rates(tstart, tstop, exposure)
                # TODO Stack counts, backgrounds across all Skygrid positions

                # TODO Remove from search
                snr = calculateSnr(counts, bkgd_rates)
                pe_veto = phosphorescenceVeto(counts, bkgd_rates, bkgd_variance)

                # Get full skygrid, templates response
                response, earthmask = instrument_data.load_response(tstart, tstop, self.skygrid)
                # What if instrument response skygrid != search skygrid?
                rsp_templates, n_skygrid, _, _ = response.shape

                rsp = response.reshape(n_templates, n_skygrid, -1)
                rsp = rsp[:, earthmask, :]

                mask = channel_mask & good

                outputs[instrument] = {
                    'counts': counts[mask],
                    'background_rates': bkgd_rates[mask],
                    'background_variance': bkgd_variance[mask],
                    'response': rsp[:, :, mask]
                }
            else:
                n_detectors = len(instrument_config.detectors)
                skygrid_counts = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
                skygrid_background = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
                skygrid_background_variance = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
                response = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)

                for i, skypos in enumerate(self.skygrid._points.T):
                    offset = instrument_data.get_timebin_offset(reference_frame, skypos)
                    counts, exposure = instrument_data.counts(tstart + offset, tstop + offset)
                    bkgd_rates, bkgd_variance = instrument_data.background_rates(tstart + offset, tstop + offset, exposure)
                    # This should return a matrix for each template, energy bin, and detector given a specific skypos
                    skypos_response = instrument_data.load_skypos_response(tstart, tstop, skypos, reference_frame)
                    # TODO reproject outputs to match reference

                    # TODO Assign all values to the skygrid matrix representation

                outputs[instrument] = {
                    'counts': skygrid_counts,
                    'background_rates': skygrid_background,
                    'background_variance': skygrid_background_variance,
                    'response': response
                }

        counts, bkgd_counts, bkgd_variance, response = self.stack_instrument_outputs(outputs)

        like = Likelihood(n_templates, self.skygrid.size)
        like.calculate(counts, bkgd_counts, bkgd_variance, response)

        # TODO Remove all following definitions from search
        tcenter = tstart + duration / 2.0

        coords_max = findLocationOfMaxLikelihood(self.skygrid, like, reference_frame)
        # convert to degrees for results storage
        ra_max = coords_max.icrs.ra[0].deg
        dec_max = coords_max.icrs.dec[0].deg
        azimuth_max = coords_max.az.deg
        zenith_max = 90.0 - coords_max.el.deg

        sun_angle = getSunAngle(coords_max, Time(t0, format='fermi'))
        geo_angle = reference_frame.geocenter.separation(coords_max)[0]

        log_sky_prior = skyPrior(self.skygrid._points[:,earthmask], reference_frame, None, None)
        coinclr = like.coinclr(log_sky_prior, llratio=like.llr)

        result = [tcenter, duration, 1, 0, like.status, azimuth_max, zenith_max, ra_max, dec_max, like.max_template,
                  like.photon_fluence/duration, like.optimal_snr, *snr, *like.chisq, sun_angle.deg, geo_angle.deg,
                  like.marginal_llr, coinclr, *pe_veto]

        return result


    def run_search(self, t0):
        timebins = self.get_timebins(t0)
        results = []
        for (tstart, dur) in timebins:
            like = self.calculate_timebin_likelihood(tstart, tstart + dur, t0)
            results.append(like)

        return np.array(results)


    def _align_timebins(self, timebins):
        reference_instrument = self.search_configuration.reference_instrument
        reference_data = self.instrument_data[reference_instrument].counter.data

        for i, (bin, dur) in enumerate(timebins):
            new_bin = None
            for data in reference_data:
                if (isinstance(data, Phaii)):
                    data_bin = data.data.closest_time_edge(bin)
                    if new_bin and new_bin != data_bin:
                        print('Warning, PHAII time bins across reference instrument detectors do not match')
                    else:
                        new_bin = data_bin
                else:
                    print('Ignoring unbinned detector data for time bin alignment')
            timebins[i] = (new_bin, dur)

        return timebins
