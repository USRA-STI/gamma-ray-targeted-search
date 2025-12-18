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
import time
import numpy as np

from rich.progress import track
from astropy.coordinates import SkyCoord
from gdt.core.data_primitives import TimeEnergyBins

from likelihood import Likelihood
from data import InstrumentData
from results import Results

import utils


class TargetedSearch():
    """Class that can perform a single or multi-instrument search for GRBs across a specified skygrid

    Attributes:
        config (SearchConfiguration):
            Instance of SearchConfiguration class with relevant settings and attributes necessary to conduct search
        skygrid (Skygrid):
            Instance of Skygrid class with expected sky positions and other relevant structures
        instrument_data (dict):
            Dictionary containing InstrumentData objects, keyed by instrument name, that allow search to access
            counts, background, response, and other necessary data related to a particular instrument
        like (Likelihood):
            Instance of likelihood class with result of the likelihood fit
        like_points (np.ndarray):
            Array with the sky position associated with each entry in like.llr.
            This can be smaller than Skygrid when sky_mask=True.
        like_frame (SpacecraftFrame):
            The spacecraft frame where like_points are defined

    Public Methods:
        get_timebins:
            Return a list with values for the start times and durations of each search bin
        add_instrument:
            Create and add a new InstrumentData instance to the instrument_data attribute
        add_calculation:
            Add a calculation to perform during the search
        calculate_likelihood:
            Perform likelihood calculation on a specific timebin across all instruments in the search
        run:
            Run the search over a set of timebins
    """
    def __init__(self, config, skygrid):
        self.config = config
        self.skygrid = skygrid
        self.instrument_data = {}

        self.like = None
        self.like_points = None
        self.like_frame = None

        self._calculations = []

    def get_timebins(self, t0=0):
        """Calculate the time bins used in the search. Each bin is defined by a start time
        and duration of source emission. The durations are defined logarithmically using
        a power of 2 spacing from min_dur to max_dur. Start times allow for overlapping
        search windows when step_size(dur) < dur.

        Args:
            t0 (float): Reference time for the center of the search period

        Returns:
            timebins (list[tuple]): List of tuples representing the start times and durations of each search bin
        """
        search_range = self.config['search_range']

        # Durations to search in powers of two
        log2maxdur = np.round(np.log2(self.config['max_dur']))
        log2mindur = np.round(np.log2(self.config['min_dur']))
        durations = 1.024 * 2. ** np.arange(log2mindur, log2maxdur + 1, 1)

        # Limits of the data interval using the reference instrument
        reference_data = self.instrument_data[self.config['reference_instrument']].data
        data_start = max([data.slice_time((search_range[0] - 0.5 * self.config['max_dur'], 0)).time_range[0] for data in reference_data])
        data_end = min([data.slice_time((0, search_range[1])).time_range[1] for data in reference_data])

        # The search bins at t0 and before
        timebins1 = [(t, dur) for dur in durations for t in np.arange(t0, data_start, -self.config.step_size(dur)) if t >= search_range[0] - dur / 2.0]

        # The search bins after t0
        timebins2 = [(t, dur) for dur in durations for t in np.arange(t0 + self.config.step_size(dur), data_end, self.config.step_size(dur)) if t + dur / 2.0 <= search_range[-1]]

        # Combine the search windows. Format: (tstart, duration)
        timebins = sorted(timebins1)
        timebins.extend(sorted(timebins2))

        timebins = self._align_timebins(timebins)

        return timebins

    def add_instrument(self, name, data, fitters, goodness_of_fit, response):
        """Create and add a new InstrumentData instance to the instrument_data attribute

        Args:
            name (str): Instrument name
            data (DataCollection[TTE|Phaii]): Data Collection to extract counts and exposure for this instrument
            fitters (DataCollection[BackgroundFitter]): Data Collection with background fit
            goodness_of_fit (DataCollection[FitStatus]): Data collection with the goodness-of-fit metric
            response (BaseResponse): Instrument response object
        """
        self.instrument_data[name] = InstrumentData(data, fitters, goodness_of_fit, response)

    def add_calculation(self, dtype, method, *args, **kwargs):
        """Adds a calculation to the search loop where `method`
        is a function defined as
        ```
        def method(search: TargetedSearch, result: np.ndarray, *args, **kwargs):
        ```

        Args:
            dtype (list): List of method return types given as [(name1, type1), (name2...)]
            method (function): A function defined according to the the example shown above.
            args (tuple, optional): Arguments passed to method
            kargs (dict, optional): Keyword arguments passed to method
        """
        self._calculations.append({"method": method, "args": args, "kwargs": kwargs, "results": np.empty(0, dtype)})

    def calculate_likelihood(self, tstart, tstop, sky_mask=True):
        """Calculate the likelihood for a given time interval defined by [tstart, tstop].
        Stores the output in the like, like_points, and like_frame attributes.

        Args:
            tstart (float): Float representing the start of the timebin
            tstop (float): Float representing the end of the timebin
            sky_mask (bool, optional): Mask obstructed sky locations (Earth, Moon, etc) when True
        """
        # always start with the first instrument in the list
        instrument = self.config['instruments'][0]
        instrument_data = self.instrument_data[instrument['name']]

        # gather counts, background, response, and sky mask matrix for first instrument
        counts, background_counts, background_var, good, response_matrix, sky_mask_matrix = \
            instrument_data.integrate(tstart, tstop, sky_mask=sky_mask, channel_mask=instrument.channel_mask)

        # save the first instrument frame as a reference for other instruments
        reference_frame = instrument_data.response.frame

        # append remaining instruments
        for i in range(1, len(self.config['instruments'])):
            # Throw error here because this code is untested. There are probably typos.
            raise NotImplemented("Searching multiple instruments is not implemented yet.")

            instrument = self.config['instruments'][i]
            instrument_data = self.instrument_data[instrument['name']]

            # gather counts, background, response, and sky mask matrix for this instrument
            counts_i, background_counts_i, background_var_i, good_i, response_matrix_i, sky_mask_matrix_i = \
                instrument_data.integrate(tstart, tstop, sky_mask=sky_mask, channel_mask=instrument.channel_mask, reference=(refrence_frame, self.skygrid))

            # update first instrument shape before stacking
            if i == 1:
                counts = np.full(response.shape, counts)
                background_counts = np.full(response.shape, background_counts)
                background_var = np.full(response.shape, background_var)
                good = np.full(response.shape, good)

            # stack this instrument with the others
            counts = np.hstack([counts, counts_i])
            background_counts = np.hstack([background_counts, background_counts_i])
            background_var = np.hstack([background_var, background_var_i])
            good = np.hstack([good, good_i])
            response_matrix = np.hstack([response_matrix, response_matrix_i])
            sky_mask_matrix = sky_mask_matrix | sky_mask_matrix_i

        # apply sky mask matrix
        response_matrix = response_matrix[:, sky_mask_matrix, :]

        # match remaining matrix shapes
        if len(counts.shape) > 1:
            counts = counts[:, sky_mask_matrix, :]
            background_counts = background_counts[:, sky_mask_matrix, :]
            background_var = background_car[:, sky_mask_matrix, :]
            good = good[:, sky_mask_matrix, :]
        else:
            good = good[np.newaxis, np.newaxis, :]

        # TO DO: The Likelihood class currently flattens the response_matrix over
        #        spectral templates x sky position assuming that counts is a 1D vector.
        #        Need to account for 2D counts shape.
        self.like = Likelihood(response_matrix.shape[0], self.skygrid.size)
        self.like.calculate(counts, background_counts, background_var, good * response_matrix)

        self.like_points = self.skygrid._points[:, sky_mask_matrix]
        self.like_frame = reference_frame

    def run(self, timebins, time_ref=0.0, sky_mask=True, progress=None):
        """Run the search over a set of timebins.

        Args:
            timebins (list[tuple]): List of tuples representing the start times and durations of each search bin
            time_ref (float, optional): Reference time for results file
            sky_mask (bool, optional): Mask obstructed sky locations (Earth, Moon, etc) when True

        Returns:
            (Results): A Results object with the likelihood result + user calculated fields for each timebin.
        """
        # prepare results arrays
        results = Results(len(timebins), time_ref=time_ref)
        [calc['results'].resize(len(timebins)) for calc in self._calculations]

        task = None if progress is None else progress.add_task("Searching...", total=len(timebins))

        for i, (tstart, duration) in enumerate(timebins):
            # compute the likelihood for this timebin
            self.calculate_likelihood(tstart, tstart + duration, sky_mask=sky_mask)

            # best-fit location
            az_max, zen_max = self.like_points[:, self.like.max_location]

            # store required result fields
            results.data[i] = (
                tstart, duration, az_max, zen_max, self.like.status, self.like.optimal_snr,
                self.like.max_template, self.like.photon_fluence/duration,
                *self.like.chisq, self.like.marginal_llr)

            # build user calculated fields
            for calc in self._calculations:
                calc['results'][i] = calc['method'](self, results.data[i], *calc['args'], **calc['kwargs'])

            if progress is not None:
                progress.update(task, advance=1)

        # combine required + user calculated results into a single array
        if len(self._calculations):
            results.append_arrays([calc['results'] for calc in self._calculations])

        return results

    def _align_timebins(self, timebins):
        """Ensures timebins match the reference instrument's data when the
        reference instrument contains binned Phaii data.

        Args:
            timebins (list[tuple]): List of tuples representing the start times and durations of each search bin

        Returns:
            (list[tuple]): List of timebins where the start time is aligned exactly with Phaii binning
                           when using binned data, otherwise returns original timebins.
        """
        reference_instrument = self.config['reference_instrument']
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
