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

from gdt.core.phaii import Phaii
from gdt.missions.fermi.time import Time
from astropy.coordinates import get_sun, SkyCoord

from likelihood import Likelihood
from data import InstrumentData
import utils


class TargetedScanner():
    """Class that can perform a single or multi-instrument scan for GRBs across a specified skygrid

        Attributes:
        -----------
            search_configuration: SearchConfiguration object
                Instance of SearchConfiguration class with relevant settings and attributes necessary to conduct search
            skygrid: Skygrid object
                Instance of Skygrid class with expected sky positions and other relevant structures
            instrument_data: Dictionary
                Dictionary containing InstrumentData objects, keyed by instrument name, that allow scanner to access
                counts, background, response, and other necessary data related to a particular instrument
        Public Methods:
        ---------------
            add_instrument:
                Create and add a new InstrumentData instance to scanner's instrument_data attribute
            get_bin_starts:
                Return the start and end times for the timebins needed to perform the scan
            get_timebins:
                Return a list with values for the start times and durations of each search bin
            stack_instrument_outputs:
                In progress. Allows multi-instrument searches to structure the inputs for Likelihood calculation
            calculate_timebin_likelihood:
                Extract the result of a likelihood calculation on a specific timebin across all instruments in search
            run_search:
                Performs a scan according to the parameters set in the search_configuration attribute and returns all
                relevant information necessary to construct a Result object

        Class Methods:
        ---------------
    """
    def __init__(self, search_configuration, skygrid):
        self.search_configuration = search_configuration
        self.skygrid = skygrid
        self.instrument_data = {}


    def add_instrument(self, name, data, fitters, response_generator, frames, fit_checker, backup_fitters):
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

        Returns:
            None
        """
        self.instrument_data[name] = InstrumentData(data, fitters, response_generator, frames, fit_checker, backup_fitters)


    def get_bin_starts(self, search_range, durations):
        """Extract the value of the search bins' start and end based on the reference instrument's data type

        Args:
            search_range (tuple): 2-tuple that includes the range start and end from which bin starts are anchored
            durations (ndarray): Array of float values representing the different durations for the targeted search
        Returns:
            tstart (float): Float representing the start bin
            tend (float): Float representing the end bin
        """
        reference_instrument = self.search_configuration['reference_instrument']
        reference_data = self.instrument_data[reference_instrument].data

        tstart = None
        tend = None

        # TODO Align using tstart from data (for search range, not bin alignment) for both TTE and Phaii data
        # NOTE: Above concern may already be addressed

        # TODO: Should these print statements be warnings or exceptions?
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
        """Calculate the time bins used in the search. These represent the different emission durations of the search
        shifted across the full search range using a given step size.

        Args:
            instrument_outputs (Dictionary): A dictionary with keys representing instruments and values being nested
                dictionaries with the counts, background rates, background variance, and response matrices extracted
                from a particular timebin

        Returns:
            tuple: 4 value tuple representing counts, background rates, background variance, and response for all
                instruments to be used in the search
        """
        if len(instrument_outputs.keys()) > 1:
            # TODO Implement multi instrument search stacking of outputs
            print('Multi-instrument search not currently supported')
        else:
            vals = list(instrument_outputs.values())[0]
            return vals['counts'], vals['background_rates'], vals['background_variance'], vals['response']


    def calculate_timebin_likelihood(self, tstart, tstop, t0):
        """Generate the necessary result data for a specific timebin by iterating over the scanner's instruments,
        extracting necessary values, and computing the Likelihood

        Args:
            tstart (float): Float representing the start of the timebin
            tstop (float): Float representing the end of the timebin
            t0 (float): Unused, time representing the central time for the search

        Returns:
            tuple: Contains necessary parameters to generate a Result object for this timebin
        """

        # TODO Where to store n_templates? energybins? Are these to be hardcoded, or added as parameters?
        shape_data = {
            "n_templates": 3,
            "n_energybins": 8,
            "num_sky_positions": self.skygrid.size
        }

        duration = tstop - tstart

        reference_instrument = self.search_configuration['reference_instrument']
        reference_frame = self.instrument_data[reference_instrument].get_spacecraft_frame((tstart + tstop) / 2)
        outputs = {}

        for instrument in self.instrument_data.keys():
            instrument_data = self.instrument_data[instrument]
            instrument_config = self.search_configuration.get_instrument(instrument)
            if instrument == reference_instrument:
                outputs[instrument] = instrument_data.format_data(instrument_config, tstart, tstop,
                                                                  self.skygrid, shape_data)
            else:
                outputs[instrument] = instrument_data.format_data_by_reference(instrument_config, tstart, tstop,
                                                                               reference_frame, self.skygrid, shape_data)

        counts, bkgd_counts, bkgd_variance, response = self.stack_instrument_outputs(outputs)

        like = Likelihood(shape_data['n_templates'], self.skygrid.size)
        like.calculate(counts, bkgd_counts, bkgd_variance, response)

        # TODO Remove all following definitions from search
        tcenter = tstart + duration / 2.0

        # TODO Move to Likelihood class?
        coords_max = utils.findLocationOfMaxLikelihood(self.skygrid, like, reference_frame)
        # convert to degrees for results storage
        ra_max = coords_max.icrs.ra[0].deg
        dec_max = coords_max.icrs.dec[0].deg
        # azimuth_max = coords_max.az.deg
        # zenith_max = 90.0 - coords_max.el.deg

        # sun_angle = utils.getSunAngle(coords_max, Time(t0, format='fermi'))
        # geo_angle = reference_frame.geocenter.separation(coords_max)[0]

        # TODO This function relies on single-instrument context; earthmask for multi-instrument search would need to be
        #      generated or composed.
        _, earthmask = instrument_data.load_response(tstart, tstop, self.skygrid)
        log_sky_prior = utils.skyPrior(self.skygrid._points[:,earthmask], reference_frame, None, None)
        coinclr = like.coinclr(log_sky_prior, llratio=like.llr)

        result = (tcenter, duration, ra_max, dec_max, like.max_template, like.photon_fluence/duration, *like.chisq,
                  like.marginal_llr, coinclr)

        return result


    def run_search(self, t0):
        """Run the search for a given central time

        Args:
            t0 (float): Float representing the target time for the search

        Returns:
            results (list[tuple]): A list of tuples from which a Result object can be generated for each timebin
        """
        timebins = self.get_timebins(t0)
        results = []
        for (tstart, dur) in timebins:
            result = self.calculate_timebin_likelihood(tstart, tstart + dur, t0)
            results.append(result)

        return results


    def _align_timebins(self, timebins):
        """Ensure that the timebins that have been generated match the reference instrument's bins in the case that the
        reference instrument contains binned Phaii data

        Args:
            timebins (list[tuple]): List of timebins with each tuple representing the start and duration of a given bin

        Returns:
            timebins (list[tuple]): List of timebines with each tuple aligned to the reference instrument's binned data,
                or the original input in the case that alignment was not needed or the reference data was unbinned
        """
        reference_instrument = self.search_configuration['reference_instrument']
        reference_data = self.instrument_data[reference_instrument].data

        for i, (bin, dur) in enumerate(timebins):
            new_bin = None
            for data in reference_data:
                # TODO Convert print statements to warnings or exceptions as necessary
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
