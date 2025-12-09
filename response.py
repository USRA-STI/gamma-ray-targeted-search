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

from abc import ABC, abstractmethod
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from utils import get_geo_coordinates, create_earth_mask
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


class GBMResponse(BaseResponse):
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
        geo_zenith (float): Zenith of the Earth center in radians for current response period
        geo_azimuth (float): Azimuth of the Earth center in radians for current response period
        geo_radius (float): Radius of the Earth in radians for current response period
        cache (dict): Cached response

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

        self.geo_azimuth = None
        self.geo_zenith = None
        self.geo_radius = None

        self.cache = None

    def load_response(self, tstart, tstop):
        """Generates the response matrix for a given spacecraft frame

        Args:
            tstart (float): Start of the response period
            tstop (float): End of the response period

        Returns:
            (np.ndarray): Response matrix
        """
        # constant response over the full period (i.e. short burst approximation)
        t = Time(0.5 * (tstart + tstop) + self.t0, format="fermi") 
        self.frame = self.spacecraft_frames.at(t) 
        self.geo_azimuth, self.geo_zenith, self.geo_radius = get_geo_coordinates(self.frame)

        if self.cache is None or angular_separation(self.geo_azimuth, 0.5 * np.pi - self.geo_zenith, *self.cache["geo"]) >= self.delta:
            # build reponse matrix from direct + atmospheric scattering components
            # when the cached matrix is None or the spacecraft has moved more than delta
            responses = []

            for detector in self.detectors:
                direct = self.direct[detector]
                atmo = self.load_atmospheric_response(detector, self.geo_azimuth, self.geo_zenith)
                responses.append(direct + atmo)

            response = np.concatenate(responses, axis=2)

            if self.templates:
                response = response[self.templates, :, :]

            self.cache = {"response": response, "geo": (self.geo_azimuth, 0.5 * np.pi - self.geo_zenith)}
        else:
            # otherwise retrieve the cached response matrix
            response = self.cache["response"]

        return response

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
