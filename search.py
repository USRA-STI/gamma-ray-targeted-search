# Copyright 2017-2025 by Universities Space Research Association (USRA). All rights reserved.
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
import numpy as np

from gdt.core.data_primitives import TimeEnergyBins
from gdt.missions.fermi.time import Time
from astropy.coordinates import get_sun, SkyCoord

from likelihood import Likelihood
from data import InstrumentData
import utils


class TargetedSearch():
    """Class that can perform a single or multi-instrument search for GRBs across a specified skygrid

    Attributes:
        search_configuration: SearchConfiguration object
            Instance of SearchConfiguration class with relevant settings and attributes necessary to conduct search
        skygrid: Skygrid object
            Instance of Skygrid class with expected sky positions and other relevant structures
        instrument_data: Dictionary
            Dictionary containing InstrumentData objects, keyed by instrument name, that allow scanner to access
            counts, background, response, and other necessary data related to a particular instrument

    Public Methods:
        add_instrument:
            Create and add a new InstrumentData instance to scanner's instrument_data attribute
        get_timebins:
            Return a list with values for the start times and durations of each search bin
        stack_instrument_outputs:
            In progress. Allows multi-instrument searches to structure the inputs for Likelihood calculation
        calculate_timebin_likelihood:
            Extract the result of a likelihood calculation on a specific timebin across all instruments in search
        run:
            Performs a scan according to the parameters set in the search_configuration attribute and returns all
            relevant information necessary to construct a Result object
    """
    def __init__(self, search_configuration, skygrid):
        self.search_configuration = search_configuration
        self.skygrid = skygrid
        self.instrument_data = {}

    def add_instrument(self, name, data, fitters, response_generator, frames, fit_checker):
        """Create and add a new InstrumentData instance to scanner's instrument_data attribute

        Args:
            name (str): Instrument name
            data (DataCollection[TTE|Phaii]): Data Collection to extract counts and exposure for this instrument
            fitters (DataCollection[BackgroundFitter]): Data Collection to extract background counts and variance
            response_generator (BaseResponseGenerator): Subclass of BaseResponseGenerator that can represent this
                instrument's expected response at a particular timebin
            frames (SpacecraftFrame): Object with position history to extract spacecraft frames at a particular time
            fit_checker (Callable[[ndarray, ndarray], ndarray]): TODO function that takes counts and background rates
                as input and outputs a ndarray of booleans identifying goodness of fit
            backup_fitters (list[DataCollection[BackgroundFitter]]): A list of replacement background fitters that would
                override parameter fitters in the case of a bad fit of the data
        """
        self.instrument_data[name] = InstrumentData(data, fitters, response_generator, frames, fit_checker)

    def get_timebins(self, t0=None):
        """Calculate the time bins used in the search. These represent the different emission durations of the search
        shifted across the full search range using a given step size.

        Args:
            t0: Currently unused, could possibly be removed

        Returns:
            timebins: list of bins with tuples representing the start times and durations of each search bin
        """
        win_width = self.search_configuration['win_width']
        min_dur = self.search_configuration['min_dur']
        max_dur = self.search_configuration['max_dur']
        min_step = self.search_configuration['min_step']
        num_steps = self.search_configuration['num_steps']

        search_range = (-win_width / 2.0, win_width / 2.0)

        # Durations to search in powers of two
        log2maxdur = np.round(np.log2(max_dur))
        log2mindur = np.round(np.log2(min_dur))
        durations = 1.024 * 2. ** np.arange(log2mindur, log2maxdur + 1, 1)

        # Limits of the data interval using the reference instrument
        reference_instrument = self.search_configuration['reference_instrument']
        reference_data = self.instrument_data[reference_instrument].data
        data_start = max([data.slice_time((search_range[0] - 0.5 * max_dur, 0)).time_range[0] for data in reference_data])
        data_end = min([data.slice_time((0, search_range[1])).time_range[1] for data in reference_data])

        # The search bins before t0
        timebins1 = [(t, dur) for dur in durations for t in np.arange(0, data_start, -max(min_step, dur / num_steps)) if t >= search_range[0] - dur / 2.0]

        # The search bins after t0, inclusive
        timebins2 = [(t, dur) for dur in durations for t in np.arange(max(min_step, dur / num_steps), data_end, max(min_step, dur / num_steps)) if t + dur / 2.0 <= search_range[-1]]

        # Combine the search windows. Format: (tstart, duration)
        timebins = sorted(timebins1)
        timebins.extend(sorted(timebins2))

        timebins = self._align_timebins(timebins)

        return timebins

    # Combine counts, backgrounds, responses into one matrix each for input to Likelihood
    def stack_instrument_outputs(self, instrument_outputs):
        """Calculate the time bins used in the search. These represent the different emission durations of the search
        shifted across the full search range using a given step size.

        Args:
            instrument_outputs (dict): A dictionary with keys representing instruments and values being nested
                dictionaries with the counts, background rates, background variance, and response matrices extracted
                from a particular timebin

        Returns:
            (tuple): 4 value tuple representing counts, background rates, background variance, and response for all
                instruments to be used in the search
        """
        if len(instrument_outputs.keys()) > 1:
            raise NotImplemented('Multi-instrument search not currently supported')
        else:
            vals = list(instrument_outputs.values())[0]
            return vals

    def calculate_likelihood(self, tstart, tstop):
        """Generate the necessary result data for a specific timebin by iterating over the scanner's instruments,
        extracting necessary values, and computing the Likelihood

        Args:
            tstart (float): Float representing the start of the timebin
            tstop (float): Float representing the end of the timebin

        Returns:
            (tuple): Contains necessary parameters to generate a Result object for this timebin
        """
        # always start with the first instrument in the list
        instrument = self.search_configuration['instruments'][0]
        instrument_data = self.instrument_data[instrument['name']]
        reference_frame = instrument_data.get_spacecraft_frame((tstart + tstop) / 2)

        # gather counts, background, response, and response mask for first instrument
        response, sky_mask = instrument_data.response.load_response(tstart, tstop, mask=True)
        counts, background_counts, background_var, good = instrument_data.format_data(tstart, tstop)

        # remove channels excluded from the likelihood
        if sum(instrument.channel_mask) < counts.shape[-1]:
            counts = counts[instrument.channel_mask]
            background_counts = background_counts[instrument.channel_mask]
            background_var = background_var[instrument.channel_mask]
            good = good[instrument.channel_mask]
            response = response[:, :, instrument.channel_mask]

        # append remaining instruments
        for i in range(1, len(self.search_configuration['instruments'])):

                if i == 1:
                    # update instrument shape to match shape needed for np.hstack
                    counts = np.full(response.shape, counts)
                    background_counts = np.full(response.shape, background_counts)

                instrument = self.search_configuration['instruments'][i]
                instrument_data = self.instrument_data[instrument['name']]

                # gather counts, background, response, and response mask for this instrument
                counts_i, background_counts_i, background_var_i, good_i = instrument_data.format_data_by_reference(tstart, tstop, reference_frame, skygrid) # define skygrid
                response_i, sky_mask_i = instrument_data.response.load_response(tstart, tstop, mask=True)

                # remove channels excluded from the likelihood
                if sum(instrument.channel_mask) < counts.shape[-1]:
                    counts = counts[:, instrument.channel_mask]
                    background_counts = background_counts[:, instrument.channel_mask]
                    background_var = background_var[:, instrument.channel_mask]
                    good = good[:, instrument.channel_mask]
                    response_i = response_i[:, :, instrument.channel_mask]

                # combine this instrument with the others
                counts = np.hstack([counts, counts_i])
                background_counts = np.hstack([background_counts, background_counts_i])
                background_var = np.hstack([background_var, background_var_i])
                good = np.hstack([good, good_i])
                response = np.hstack([response, response_i])
                sky_mask = sky_mask | sky_mask_i

        good = good[np.newaxis, np.newaxis, :] if len(good.shape) == 1 else good[np.newaxis, :, :]
        response = response[:, sky_mask, :]

        like = Likelihood(response.shape[0], self.skygrid.size)
        like.calculate(counts, background_counts, background_var, good * response)

        # TODO Move to Likelihood class?
        coords_max = utils.find_location_of_max_likelihood(self.skygrid, like, reference_frame)
        # convert to degrees for results storage
        ra_max = coords_max.icrs.ra[0].deg
        dec_max = coords_max.icrs.dec[0].deg
        # azimuth_max = coords_max.az.deg
        # zenith_max = 90.0 - coords_max.el.deg

        # sun_angle = utils.get_sun_angle(coords_max, Time(t0, format='fermi'))
        # geo_angle = reference_frame.geocenter.separation(coords_max)[0]

        # TODO This function relies on single-instrument context; earthmask for multi-instrument search would need to be
        #      generated or composed.
        log_sky_prior = utils.sky_prior(self.skygrid._points[:, sky_mask], reference_frame, None, None)
        coinclr = like.coinclr(log_sky_prior, llratio=like.llr)

        duration = tstop - tstart
        result = (tstart, duration, ra_max, dec_max, like.max_template, like.photon_fluence/duration, *like.chisq,
                  like.marginal_llr, coinclr)

        return result

    def run(self, t0):
        """Run the search for a given central time

        Args:
            t0 (float): Float representing the target time for the search

        Returns:
            (list[tuple]): A list of tuples from which a Result object can be generated for each timebin
        """
        timebins = self.get_timebins(t0)
        results = []
        for (tstart, dur) in timebins:
            result = self.calculate_likelihood(tstart, tstart + dur)
            results.append(result)

        return results

    def _align_timebins(self, timebins):
        """Ensure that the timebins that have been generated match the reference instrument's bins in the case that the
        reference instrument contains binned Phaii data

        Args:
            timebins (list[tuple]): List of timebins with each tuple representing the start and duration of a given bin

        Returns:
            (list[tuple]): List of timebins with each tuple aligned to the reference instrument's binned data,
                or the original input in the case that alignment was not needed or the reference data was unbinned
        """
        reference_instrument = self.search_configuration['reference_instrument']
        reference_data = self.instrument_data[reference_instrument].data

        for i, (start, dur) in enumerate(timebins):
            new_start = None
            for data in reference_data:
                if isinstance(data.data, TimeEnergyBins):
                    closest = data.data.closest_time_edge(start)
                    if new_start and new_start != closest:
                        raise ValueError('Warning, PHAII time bins across reference instrument detectors do not match')
                    else:
                        new_start = closest
            if new_start:
                timebins[i] = (new_start, dur)

        return timebins
