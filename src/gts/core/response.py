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
import time as unix_time
import numpy as np

from abc import ABC, abstractmethod
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from .utils import get_geo_coordinates, create_earth_mask
from astropy.coordinates import angular_separation
from astropy.time import Time


class BaseResponse(ABC):
    """Abstract class that can load a response matrix for the TargetedSearch

    Attributes:
        detectors (list[str]): List of detector names
        skygrid (Skygrid): Instance of Skygrid class with expected sky positions and other relevant structures

    Public Methods:
        load_response: Abstract method to compute the response matrix for period of time
        sky_mask: Return sky mask with True for visible skygrid locations, False otherwise
    """
    def __init__(self, detectors, skygrid):
        self.detectors = detectors
        self.skygrid = skygrid

    @abstractmethod
    def load_response(self, tstart, tstop, **kwargs):
        pass

    def sky_mask(self):
        return None


class GbmResponse(BaseResponse):
    """Implementation of the GBM instrument response for the TargetedSearch

    Attributes:
        t0 (float): Reference time used by the TargetedSearch class
        detectors (list[str]): List of detector names
        skygrid (Skygrid): Instance of Skygrid class with expected sky positions and other relevant structures
        templates (list): List of spectral template indices to use
        templates_directory (str): String representing the location where response templates are stored
        delta (float): Angular displacement in radians for rebuilding atmospheric scattering response
        zen_margin (float): Zenith margin in radians for choosing atmospheric scattering response
        rocking_zen (float): Zenith angle in radians for the atmospheric scattering response
        det_index (dict): Mapping between detector name and template file index
        spacecraft_frames (SpacecraftFrame): Object with spacecraft orientation over time
        available_azimuths (list): List of available atmospheric response azimuths
        direct (dict): Dictionary with direct response matrices for each detector
        frame (SpacecraftFrame): Frame with spacecraft orientation for current response period
        in_rock (int): Rocking profile for current response period
        time_range (tuple[float]): Time range in seconds for current response period
        geo_zenith (float): Zenith of the Earth center in radians for current response period
        geo_azimuth (float): Azimuth of the Earth center in radians for current response period
        geo_radius (float): Radius of the Earth in radians for current response period
        response_matrix (np.ndarray): Response matrix for current response period
        _preprocessed (dict): Preprocessed values for generating the response matrix

    Public Methods:
        load_response: Method to compute the response matrix for a given time bin
        load_direct_response: Method to load the direct response matrix for a given time bin
        load_atmospheric_response: Method to load the atmospheric response matrix for a given detector and azimuth
        get_available_azimuths: Method to check which azimuths are available in the templates for a given detector
        get_detector_type: Convert detector name to nai or bgo string
        sky_mask: Return sky mask with True for visible skygrid locations, False otherwise
    """
    zen_margin = np.radians(5.0)
    rocking_zen = np.radians(130.0)
    det_index = {'n0': 0, 'n1': 1, 'n2': 2, 'n3':3, 'n4': 4, 'n5': 5, 'n6': 6, 'n7': 7, 'n8': 8, 'n9': 9, 'na': 10, 'nb': 11, 'b0': 0, 'b1': 1}

    def __init__(self, detectors, skygrid, templates_directory, spacecraft_frames, 
                 t0, delta: float = np.radians(0.1), templates: list = None):
        """Class constructor

        Args:
            detectors (list[str]): List of detector names
            skygrid (Skygrid): The skygrid this response should be generated over
            templates_directory (str): String representing the path where the templates for the GBM response are stored
            spacecraft_frames (SpacecraftFrame): Spacecraft position history object
            t0 (float): Reference time of the search
            delta (float): Angular displacement for rebuilding atmospheric scattering response
            templates (list): list of template IDs to use
        """
        super().__init__(detectors, skygrid)
        self.t0 = t0
        self.delta = delta
        self.templates_directory = templates_directory
        self.available_azimuths = self.get_available_azimuths('n0')
        self.templates = templates
        self.spacecraft_frames = spacecraft_frames

        self.direct = {}
        for detector in self.detectors:
            self.direct[detector] = self.load_direct_response(detector)

        self.frame = None
        self.in_rock = None
        self.time_range = None
        self.geo_azimuth = None
        self.geo_zenith = None
        self.geo_radius = None
        self.response_matrix = None

        self._preprocessed = {'load_points': {}}

    def preprocess(self, timebins):
        """Method for pre-processing expensive calculations used during
        a search over many timebins. Running this before a search makes
        the search run faster.

        Args:
            timebins (list[tuple]): List of tuples representing the start times and durations of each search bin
        """
        tstart, dur = np.transpose(timebins)
        tstop = tstart + dur

        # Apply astropy's broadcasting optimizations
        frames = self.spacecraft_frames.at(Time(0.5 * (tstart + tstop) + self.t0, format="fermi"))
        geo_azimuth, geo_zenith, geo_radius = get_geo_coordinates(frames)

        # Store values
        self._preprocessed = {'frames': frames, 'geo_azimuth': geo_azimuth, 'geo_zenith': geo_zenith, 'geo_radius': geo_radius}

        # Run through timebins to determine response load points.
        # This reduces disk i/o by sharing response matrices across
        # similar spacecraft positions.
        prev_geo = None
        time_range = None
        self._preprocessed['load_points'] = {}
        for i in range(tstart.size):
            if prev_geo is None or angular_separation(geo_azimuth[i], 0.5 * np.pi - geo_zenith[i], *prev_geo) >= self.delta:
                prev_geo = (geo_azimuth[i], 0.5 * np.pi - geo_zenith[i])
                time_range = (tstart[i], tstop[i])

            self._preprocessed['load_points'][(tstart[i], tstop[i])] = (i, time_range)

    def load_response(self, tstart, tstop):
        """Generates the response matrix for a given spacecraft frame

        Note: We currently assume a constant response over the full period (i.e. short burst approximation)

        Args:
            tstart (float): Start of the response period
            tstop (float): End of the response period

        Returns:
            (np.ndarray): Response matrix
        """
        load_geo_pos = None

        if (tstart, tstop) in self._preprocessed['load_points']:
            i, time_range = self._preprocessed['load_points'][(tstart, tstop)]

            self.frame = self._preprocessed['frames'][i]
            self.geo_azimuth = self._preprocessed['geo_azimuth'][i]
            self.geo_zenith = self._preprocessed['geo_zenith'][i]
            self.geo_radius = self._preprocessed['geo_radius'][i]

            if time_range == self.time_range: # response is loaded, return it
                return self.response_matrix

            if time_range != (tstart, tstop): # lookup load point tstart, tstop, geo position
                j, (tstart, tstop) = self._preprocessed['load_points'][time_range]
                load_geo_pos = (self._preprocessed['geo_azimuth'][j],
                                self._preprocessed['geo_zenith'][j])
        else:
            self.frame = self.spacecraft_frames.at(Time(0.5 * (tstart + tstop) + self.t0, format="fermi"))
            self.geo_azimuth, self.geo_zenith, self.geo_radius = get_geo_coordinates(self.frame, single=True)

        # load current geo position if we aren't loading a pre-processed position
        if load_geo_pos is None:
            load_geo_pos = (self.geo_azimuth, self.geo_zenith)

        # build reponse matrix from direct + atmospheric scattering components
        # when load is requested or the cached matrix is None
        responses = []

        for detector in self.detectors:
            direct = self.direct[detector]
            atmo = self.load_atmospheric_response(detector, *load_geo_pos)
            responses.append(direct + atmo)

        response = np.concatenate(responses, axis=2)

        if self.templates:
            response = response[self.templates, :, :]

        self.response_matrix = response
        self.in_rock = int(isinstance(atmo, np.ndarray))
        self.time_range = (tstart, tstop)

        return self.response_matrix

    def sky_mask(self):
        """(np.ndarray): Generates sky mask with visible locations set to True, Earth occulted set to False."""
        if self.geo_azimuth is None:
            raise ValueError("Run load_response() before requesting sky mask")
        return create_earth_mask(self.skygrid._points, self.geo_azimuth, self.geo_zenith, self.geo_radius)

    def load_direct_response(self, detector):
        """Loads the direct response matrix for a specific detector

        Args:
            detector (str): The name of the detector

        Returns:
            (ndarray): The direct response matrix/array for one detector
        """
        path = os.path.join(self.templates_directory, 'direct', f"{self.get_detector_type(detector)}.npy")
        return np.load(path)[:, :, :, self.det_index[detector]]

    def load_atmospheric_response(self, detector, geo_az, geo_zen):
        """Loads the atmospheric response matrix for a specific detector

        Args:
            detector (str): The name of the detector
            geo_az (float): Azimuth of the Earth center in radians
            geo_zen (float): Zenith of the Earth center in radians

        Returns:
            (ndarray): The atmospheric response matrix/array for one detector
        """
        if np.abs(geo_zen - self.rocking_zen) > self.zen_margin:
            return 0.0

        # calculate nearest available azimuths
        idx = np.argsort(np.abs(geo_az - self.available_azimuths))[:2]
        nearest_az = self.available_azimuths[idx]

        # load responses
        responses = []
        for az in nearest_az:
            path = os.path.join(self.templates_directory, 'atmo_' + self.get_detector_type(detector), f'atmrates_az{int(np.degrees(az))}_zen130.npy')
            responses.append(np.load(path)[:, :, :, self.det_index[detector]])

        # calculate weighted response
        width = np.abs(nearest_az[0] - nearest_az[1])
        dtheta = np.abs(geo_az - nearest_az)
        weights = 1.0 - (dtheta / width)
        weighted_response = responses[0] * weights[0] + responses[1] * weights[1]

        return weighted_response

    def get_available_azimuths(self, detector):
        """Finds all available azimuths for a given detector

        Args:
            detector (str): The name of the detector

        Returns:
            (ndarray): An array with the available azimuths for the detector
        """
        paths = os.listdir(os.path.join(self.templates_directory, 'atmo_' + self.get_detector_type(detector)))
        azimuths = [np.radians(float(path.split('_')[1][2:]))
                    for path in paths if path.startswith('atmrates_az')]

        # account for angular rollover
        if 0.0 in azimuths:
            azimuths.append(2 * np.pi)

        return np.array(sorted(azimuths))

    def get_detector_type(self, detector):
        """Gets the type of a specified detector according to GBM standards

        Args:
            detector (str): Detector name

        Returns:
            (str): Detector type
        """
        gbm_det = GbmDetectors.from_str(detector)
        if gbm_det.is_nai():
            return 'nai'
        return 'bgo'
