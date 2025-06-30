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


class InstrumentData:
    """Class for storing necessary data components for targeted search, for a single instrument."""


    def __init__(self, data, fitters, response_generator, spacecraft_frames, goodness_of_fit, backup_fitters):
        """ Class constructor
        Args:
            data (DataCollection[TTE|Phaii]): Data Collection to extract counts and exposure for this instrument
            fitters (DataCollection[BackgroundFitter]): Data Collection to extract background counts and variance
            response_generator (BaseResponseGenerator): Subclass of BaseResponseGenerator that can represent this
                instrument's expected response at a particular timebin
            frames (SpacecraftFrame): Object with position history to extract spacecraft frames at a particular time
            goodness_of_fit (Callable[[ndarray, ndarray], ndarray]): TODO function that takes counts and background rates
                as input and outputs a ndarray of booleans identifying goodness of fit
            backup_fitters (list[DataCollection[BackgroundFitter]]): A list of replacement background fitters that would
                override parameter fitters in the case of a bad fit of the data

        Returns:
            None
        """
        # TODO Backup fitters should be an array of DataCollections of BackFitters to be used in case we find the
        #      background fit is not suitable

        # Sanity checks
        background_bounds = [fitter._data_obj.ebounds for fitter in fitters]
        for i, det in enumerate(data.items):
            match_det = data.items[i] == fitters.items[i] == response_generator.detectors[i]
            match_ebounds = data.ebounds()[i].low_edges() == background_bounds[i].low_edges() \
                            and data.ebounds()[i].high_edges() == background_bounds[i].high_edges()
            # TODO Convert print statements to warnings or exceptions as necessary
            if not match_det:
                print("Warning: Detectors are not in order across input data, backfitters, or response generator."
                      "Please check to ensure correct ordering.")
            if not match_ebounds:
                print("Warning: Energy bounds do not match across input data and backfitters. Please check to ensure "
                      "inputs are valid.")

        self.data = data
        self.fitters = fitters
        self.response_generator = response_generator
        self.spacecraft_frames = spacecraft_frames
        self.goodness_of_fit = goodness_of_fit
        self.backup_fitters = backup_fitters


    @property
    def detectors(self):
        """list[str] representing the names of the instrument's detectors"""
        return self.data.items


    @property
    def ebounds(self):
        """list[Ebounds] representing the energy bounds of each detector in the instrument"""
        return self.data.ebounds()


    def counts(self, tstart, tstop):
        """Extracts the counts and exposure from this instrument given a timebin across all detectors

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin

        Returns:
            tuple(ndarray, ndarray): Tuple consisting of two arrays, one for counts and one for exposure, extracted
                from this instrument

        """
        counts, exposure = [], []
        for spec in self.data.to_spectrum(time_range=(tstart, tstop)):
            counts.append(spec.counts)
            exposure.append(spec.exposure[0])

        return np.ravel(counts), np.ravel(exposure)


    def background_rates(self, tstart, tstop, exposure):
        """Extracts the background rates and background variance for this instrument across all detectors

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            exposure (list[float]): List of exposures for each detector

        Returns:
            tuple (ndarray, ndarray, ndarray): A tuple consisting of the matrices of background rates and variance, and
                an ndarray of booleans representing the goodness of fit for that detector at the given time bin
        """
        tstart = np.atleast_1d(tstart)
        tstop = np.atleast_1d(tstop)

        counts, counts_var, good = [], [], []
        for i, fitter in enumerate(self.fitters):
            rates, rate_uncert = fitter._method.interpolate(tstart, tstop)
            counts.append(rates[0] * exposure[i])
            counts_var.append(0.5 * (rate_uncert[0] * exposure[i]) ** 2)
            # TODO Replace with correct TTE/PHAII counts for goodness of fit.
            good.append(self.goodness_of_fit(counts, rates))

        return np.ravel(counts), np.ravel(counts_var), np.ravel(good)


    def load_response(self, tstart, tstop, skygrid, earthmask=False):
        """Extracts the expected response matrix for this instrument, representing all detectors

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            skygrid (Skygrid): The skygrid we are searching over. Currently unused.
            earthmask (bool): A boolean representing whether or not to apply/extract the earthmask at this time bin

        Returns:
            ndarray: A matrix representing the expected response at a given timebin, representing all detectors
        """
        return self.response_generator.load_response(tstart, tstop)


    def load_skypos_response(self, tstart, tstop, target_skypos, reference_frame, earthmask=False):
        # TODO Additional function to compute response given a target skypos. Should be used by scanner when this
        #      instrument is not the reference instrument
        # Note: Can this function take just the spacecraft frame itself rather than calculating it
        pass


    def get_spacecraft_frame(self, time):
        """Extracts this instrument's spacecraft frame that is the closest match to where it would be at a given time

        Args:
            time (float): Target time

        Returns:
            spacecraft_frame (SpacecraftFrame): The frame the spacecraft was at nearest to the specified time
        """
        frame_index = np.abs(self.spacecraft_frames.obstime.value - time).argmin()
        spacecraft_frame = self.spacecraft_frames[frame_index]

        return spacecraft_frame


    def get_timebin_offset(self, reference_frame, target_skypos):
        # TODO Calculate offset based on target sky pos, reference_frame, finding the frame in this instance's frames
        #      that would correspond to when the energy beam would reach this instrument
        #      Return a float representing the timebin offset, along with the spacecraft frame associated with it.
        pass


    def format_data(self, instrument_config, tstart, tstop, skygrid, shape_data):
        """Formats the instrument's counts, background rates, background variance, and response, including masking only
        good bins and the earth mask, for the scanner to use in its search. Used if this is the reference instrument
        used by the scanner

        Args:
            instrument_config (InstrumentConfiguration): The configuration for this instrument
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            skygrid (Skygrid): The skygrid we are searching over, from the scanner
            shape_data (dict): TODO Currently stores the shape data for templates, energy bins, and sky positions. There
                should be a better way to integrate these parameters

        Returns:
            dict: A dictionary with keys and values for the counts, background rates, background variance, and response
                extracted from this instrument's data classes
        """
        n_templates = shape_data['n_templates']

        channel_mask = instrument_config.channel_mask
        counts, exposure = self.counts(tstart, tstop)

        bkgd_rates, bkgd_variance, good = self.background_rates(tstart, tstop, exposure)

        # Get full skygrid, templates response
        response, earthmask = self.load_response(tstart, tstop, skygrid)
        # TODO What if instrument response skygrid != search skygrid? Currently, load_response does not use skygrid
        #      or earthmask arguments
        rsp_templates, n_skygrid, _, _ = response.shape

        # NOTE: Why get n_skygrid from response, when n_templates is static, and n_skygrid can be grabbed from
        # scanner's skygrid attribute?
        rsp = response.reshape(n_templates, n_skygrid, -1)
        rsp = rsp[:, earthmask, :]

        mask = channel_mask & good

        return {
            'counts': counts[mask],
            'background_rates': bkgd_rates[mask],
            'background_variance': bkgd_variance[mask],
            'response': rsp[:, :, mask]
        }


    def format_data_by_reference(self, instrument_config, tstart, tstop, reference_frame, skygrid, shape_data):
        """Formats the instrument's counts, background rates, background variance, and response, including masking only
        good bins and the earth mask, for the scanner to use in its search. Used if this is an additional instrument,
        and not the reference instrument, for the scanner

        TODO Implementation in progress, pseudocode only for now

        Args:
            instrument_config (InstrumentConfiguration): The configuration for this instrument
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            reference_frame (SpacecraftFrame): The frame of the reference craft/instrument
            skygrid (Skygrid): The skygrid we are searching over, from the scanner
            shape_data (dict): TODO Currently stores the shape data for templates, energy bins, and sky positions. There
                should be a better way to integrate these parameters

        Returns:
            dict: A dictionary with keys and values for the counts, background rates, background variance, and response
                extracted from this instrument's data classes
        """
        n_templates = shape_data['n_templates']
        num_sky_positions = shape_data['num_sky_positions']
        n_energybins = shape_data['n_energybins']

        n_detectors = len(instrument_config.detectors)
        skygrid_counts = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
        skygrid_background = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
        skygrid_background_variance = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
        response = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)

        # Iterate over all target sky positions
        for i, skypos in enumerate(skygrid._points.T):
            offset = self.get_timebin_offset(reference_frame, skypos)
            counts, exposure = self.counts(tstart + offset, tstop + offset)
            bkgd_rates, bkgd_variance = self.background_rates(tstart + offset, tstop + offset, exposure)
            # This should return a matrix for each template, energy bin, and detector given a specific skypos
            skypos_response = self.load_skypos_response(tstart, tstop, skypos, reference_frame)
            # TODO reproject outputs to match reference

            # TODO Assign all values to the skygrid matrix representation

        return {
            'counts': skygrid_counts,
            'background_rates': skygrid_background,
            'background_variance': skygrid_background_variance,
            'response': response
        }
