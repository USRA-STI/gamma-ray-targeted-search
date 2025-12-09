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
import os
import numpy as np

class FitStatus:
    """Placeholder class for fit status behavior

    Attributes:
        good (ndarray): Array of static goodness-of-fit values (always True for now)

    Public Methods:
        status: Retrieves goodness-of-fit status with True = good, False = bad.
    """
    def __init__(self, shape):
        """Class constructor

        Args:
            shape (tuple): Shape of the goodness-of-fit array
        """
        self.good = np.ones(shape, dtype=bool)

    def status(self, tstart, tstop):
        """(ndarray): Goodness-of-fit array

        Note: this returns a placeholder value which is always good
        """
        return self.good


class InstrumentData:
    """Class for storing data from a single instrument in a format accessible to the TargetedSearch class.

    Attributes:
        data (DataCollection): Collection of data for each detector
        fitters (DataCollection): Collection of background fits for each detector
        response (BaseResponse): Instrument response object
        goodness_of_fit (FitStatus): Collection of background fit statuses for each detector
        good (np.ndarray): Goodness-of-fit array for the current integration interval
        counts (np.ndarray): Counts array for the current integration interval
        background_counts (np.ndarray): Background counts array for the current integration interval
        background_var (np.ndarray): Background variance array for the current integration interval
        response_matrix (np.ndarray): Response array for the current integration interval
        sky_mask (np.ndarray): Array with visible sky positions for the current integration interval

    Public Methods:
        get_timebin_offset: Computes time-of-flight from a reference frame to this instrument given a sky location
        format_data: Retrieve data counts, background counts, background variance, and goodness of fit for a time interval
        format_data_by_reference: Similar to format_data, but the time interval is calculated relative to another instrument
    """
    def __init__(self, data, fitters, goodness_of_fit, response):
        """Class constructor

        Args:
            data (DataCollection[TTE|Phaii]): Data Collection to extract counts and exposure for this instrument
            fitters (DataCollection[BackgroundFitter]): Data Collection to extract background counts and variance
            goodness_of_fit (DataCollection[FitStatus]): Data collection with the goodness-of-fit metric
            response (BaseResponse): Instrument response object
        """
        # Sanity checks
        for i, det in enumerate(data.items):
            match_det = data.items[i] == fitters.items[i] == response.detectors[i]
            match_ebounds = data.ebounds()[i].low_edges() == fitters.get_item(det)._data_obj.ebounds.low_edges() \
                            and data.ebounds()[i].high_edges() == fitters.get_item(det)._data_obj.ebounds.high_edges()

            if not match_det:
                raise ValueError("Detectors are not in order across input data, backfitters, or response generator. "
                                 "Please check to ensure correct ordering.")
            if not match_ebounds:
                raise ValueError("Energy bounds do not match across input data and backfitters. "
                                 "Please check to ensure inputs are valid.")

        self.data = data
        self.fitters = fitters
        self.response = response
        self.goodness_of_fit = goodness_of_fit

        # initialize values for the integrate() method
        self.good = None
        self.frame = None
        self.counts = None
        self.sky_mask = None
        self.background_var = None
        self.response_matrix = None
        self.background_counts = None

    @property
    def detectors(self):
        """list[str] representing the names of the instrument's detectors"""
        return self.data.items

    @property
    def ebounds(self):
        """list[Ebounds] representing the energy bounds of each detector in the instrument"""
        return self.data.ebounds()

    def get_timebin_offset(self, frame, location):
        """Computes time-of-flight from a reference frame to this instrument given a sky location.

        Args:
            frame (SpacecraftFrame): Frame object with the position of a reference instrument
            location (tuple): Sky location

        Returns:
            (float or np.ndarray)
        """
        # TODO Calculate time-of-flight from reference frame to the current instrument frame
        #      based on a plane wave coming from location.
        return 0

    def integrate(self, tstart, tstop, reference=None, sky_mask=True, channel_mask=None):
        """Method to integrate data and responses over time interval [tstart, tstop]

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            reference (tuple): Tuple with a reference frame and skygrid
            sky_mask (bool): A boolean representing whether or not to apply/extract a sky mask
                             used to remove regions blocked by the Earth, Moon, etc.
            channel_mask (np.ndarray): A channel mask to apply on the return values

        Returns:
            tuple: Tuple with arrays for counts, background counts, background variance,
                   good fit status, response matrix, and sky mask matrix
        """
        self.response_matrix = self.response.load_response(tstart, tstop)
        self.sky_mask_matrix = self.response.sky_mask() if sky_mask else None

        if reference is None:
            self.counts, self.background_counts, self.background_var, self.good = self.format_data(tstart, tstop)
        else:
            # TODO: rotate response_matrix to the reference frame
            self.counts, self.background_counts, self.background_var, self.good = self.format_data_by_reference(tstart, tstop, *reference)

        # remove masked channels when requested
        if channel_mask is not None and sum(channel_mask) < self.counts.shape[-1]:
            return self.counts[..., channel_mask], self.background_counts[..., channel_mask], \
                   self.background_var[..., channel_mask], self.good[..., channel_mask], \
                   self.response_matrix[..., channel_mask], self.sky_mask_matrix

        return self.counts, self.background_counts, self.background_var, self.good, self.response_matrix, self.sky_mask_matrix

    def format_data(self, tstart, tstop):
        """Formats the counts, background counts, background variance in the current instrument's frame.

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin

        Returns:
            (tuple[ndarray]): A tuple with counts, background counts, background variance, and background goodness-of-fit
        """
        counts, background_counts, background_var, good = [], [], [], []

        for det in self.detectors:

            # get data counts during the interval (tstart, tstop)
            spec = self.data.get_item(det).to_spectrum(time_range=(tstart, tstop))
            counts.append(spec.counts)
            exposure = spec.exposure[0]

            # estimate background counts during the interval (tstart, tstop)
            rates, rate_uncert = self.fitters.get_item(det)._method.interpolate(
                np.array([tstart]), np.array([tstop]))
            background_counts.append(rates[0] * exposure)
            background_var.append(0.5 * (rate_uncert[0] * exposure) ** 2)

            good.append(self.goodness_of_fit.get_item(det).status(tstart, tstop))

        return np.ravel(counts), np.ravel(background_counts), np.ravel(background_var), np.ravel(good)

    def format_data_by_reference(self, tstart, tstop, frame, skygrid):
        """Formats the counts, background counts, background variance for searches that
        use another instrument as the reference frame.

        TODO:
            1. Implement get_timebin_offset
            2. Update Likelihood._flatten_data to handle case where counts
               has the shape (nsky, ndetector_channels) instead of (ndetector_channels)

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            frame (SpacecraftFrame): The frame of the reference instrument
            skygrid (Skygrid): The skygrid we are searching over, from the scanner

        Returns:
            (tuple[ndarray]): A tuple with counts, background counts, background variance, and background goodness-of-fit
        """
        counts, background_counts, background_var, good = [], [], [], []

        # Iterate over all target sky positions
        for i, skypos in enumerate(skygrid._points.T):

            # get data at the time offset for this position
            offset = self.get_timebin_offset(frame, skypos)
            data_at_offset = self.format_data(tstart + offset, tstop + offset)

            # store for output
            counts.append(data_at_offset[0])
            background_counts.append(data_at_offset[1])
            background_var.append(data_at_offset[2])
            good.append(data_at_offset[3])

        return np.array(counts), np.array(background_counts), np.array(background_var), np.array(good)
