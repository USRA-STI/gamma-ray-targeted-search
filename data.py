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

from astropy.time import Time

class FitStatus:
    """Placeholder class for fit status behavior

    Attributes:
        good (ndarray): Array of static goodness-of-fit values (always True for now)

    Public Methods:
        status: Retrieves goodness-of-fit status with True = good, False = bad.
    """
    def __init__(self, shape):
        self.good = np.ones(shape, dtype=bool)

    def status(self, tstart, tstop):
        """(ndarray): Goodness-of-fit array"""
        return self.good


class InstrumentData:
    """Class for storing necessary data components for targeted search, for a single instrument.

    Attributes:
        data (DataCollection): Collection of data for each detector
        fitters (DataCollection): Collection of background fits for each detector
        response (BaseResponse): Instrument response object
        frames (SpacecraftFrame): Position history object
        goodness_of_fit (FitStatus): Collection of background fit statuses for each detector
        time_format (str): Time format used by data

    Public Methods:
        format_data: Retrieve data counts, background counts, background variance, and goodness of fit for a time interval
        format_data_by_reference: Similar to format_data, but the time interval is calculated relative to another instrument
    """
    def __init__(self, data, fitters, response, spacecraft_frames, goodness_of_fit, time_format):
        """ Class constructor

        Args:
            data (DataCollection[TTE|Phaii]): Data Collection to extract counts and exposure for this instrument
            fitters (DataCollection[BackgroundFitter]): Data Collection to extract background counts and variance
            response (BaseResponse): Instrument response object
            frames (SpacecraftFrame): Position history object
            goodness_of_fit (FitStatus): Collection of background fit statuses for each detector
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
        self.spacecraft_frames = spacecraft_frames
        self.goodness_of_fit = goodness_of_fit
        self.time_format = time_format

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

    def get_spacecraft_frame(self, rel_time):
        """Extracts this instrument's spacecraft frame that is the closest match to where it would be at a given time

        Args:
            time (float): Target time

        Returns:
            spacecraft_frame (SpacecraftFrame): The frame the spacecraft was at nearest to the specified time
        """
        t = Time(rel_time + self.data.get_item(self.data.items[0]).trigtime, format=self.time_format)
        return self.spacecraft_frames.at(t)

    def get_timebin_offset(self, reference_frame, target_skypos):
        # TODO Calculate offset based on target sky pos, reference_frame, finding the frame in this instance's frames
        #      that would correspond to when the energy beam would reach this instrument
        #      Return a float representing the timebin offset, along with the spacecraft frame associated with it.
        return 0

    def integrate(self, tstart, tstop, reference=None, sky_mask=True, channel_mask=None):
        """ Method to integrate data and responses over time interval [tstart, tstop]

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
        # Note: we cache return values within the class so that users can access the same values across
        #       calculations for the likelihood, signal-to-noise ratio, phospherescence veto, etc.
        if reference is None:
            rsp, self.frame = self.format_response(tstart, tstop, sky_mask=sky_mask)
            self.counts, self.background_counts, self.background_var, self.good = self.format_data(tstart, tstop)
        else:
            rsp, self.frame = self.format_response_by_reference(tstart, tstop, *reference, sky_mask=sky_mask)
            self.counts, self.background_counts, self.background_var, self.good = self.format_data_by_reference(tstart, tstop, *reference)

        self.response_matrix, self.sky_mask_matrix = rsp if sky_mask else (rsp, None)

        # remove masked channels when requested
        if channel_mask is not None and sum(channel_mask) < self.counts.shape[-1]:
            return self.counts[..., channel_mask], self.background_counts[..., channel_mask], \
                   self.background_var[..., channel_mask], self.good[..., channel_mask], \
                   self.response_matrix[..., channel_mask], self.sky_mask_matrix

        return self.counts, self.background_counts, self.background_var, self.good, self.response_matrix, self.sky_mask_matrix

    def format_response(self, tstart, tstop, sky_mask=False):
        """Extracts the expected response matrix for this instrument, representing all detectors

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            sky_mask (bool): A boolean representing whether or not to apply/extract a sky mask
                             used to remove regions blocked by the Earth, Moon, etc.

        Returns:
            tuple: ((np.ndarray, np.ndarry), SpacecraftFrame) when mask = True, else (np.ndarray, SpacecraftFrame)
        """
        tcent = 0.5 * (tstart + tstop)
        frame = self.get_spacecraft_frame(tcent)

        return self.response.load_response(frame, sky_mask), frame

    def format_response_by_reference(self, tstart, tstop, reference_frame, skygrid, sky_mask=False):
        """Extracts the expected response matrix for this instrument, representing all detectors

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            reference_frame (SpacecraftFrame): reference spacecraft frame
            skygrid (Skygrid): The skygrid we are searching over, from the scanner
            sky_mask (bool): A boolean representing whether or not to apply/extract a sky mask
                             used to remove regions blocked by the Earth, Moon, etc.

        Returns:
            tuple: ((np.ndarray, np.ndarry), SpacecraftFrame) when mask = True, else (np.ndarray, SpacecraftFrame)
        """
        # need to retrieve reponse and rotate into reference frame
        raise NotImplemented("Loading response for a sky position is not implemented yet.")

    def format_data(self, tstart, tstop):
        """Formats the instrument's counts, background rates, background variance, and response, including masking only
        good bins and the earth mask, for the scanner to use in its search. Used if this is the reference instrument
        used by the scanner

        Args:
            instrument_config (InstrumentConfiguration): The configuration for this instrument
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

    def format_data_by_reference(self, tstart, tstop, reference_frame, skygrid):
        """Formats the instrument's counts, background rates, background variance, and response, including masking only
        good bins and the earth mask, for the scanner to use in its search. Used if this is an additional instrument,
        and not the reference instrument, for the scanner

        TODO Implementation in progress, pseudocode only for now

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            reference_frame (SpacecraftFrame): The frame of the reference craft/instrument
            skygrid (Skygrid): The skygrid we are searching over, from the scanner

        Returns:
            (tuple[ndarray]): A tuple with counts, background counts, background variance, and background goodness-of-fit
        """
        counts, background_counts, background_var, good = [], [], [], []

        # Iterate over all target sky positions
        for i, skypos in enumerate(skygrid._points.T):

            # get data at the time offset for this position
            offset = self.get_timebin_offset(reference_frame, skypos)
            data_at_offset = self.format_data(tstart + offset, tstop + offset)

            # store for output
            counts.append(data_at_offset[0])
            background_counts.append(data_at_offset[1])
            background_var.append(data_at_offset[2])
            good.append(data_at_offset[3])

        return np.array(counts), np.array(background_counts), np.array(background_var), np.array(good)
