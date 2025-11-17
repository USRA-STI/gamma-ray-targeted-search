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
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from utils import get_geo_coordinates, create_earth_mask
from astropy.coordinates import angular_separation


class BaseResponse(ABC):
    """Abstract class that can load a response matrix for the TargetedSearch

    Attributes:
        detectors (list[str]): List of detector names
        skygrid (Skygrid): Instance of Skygrid class with expected sky positions and other relevant structures
        spacecraft_frames (SpacecraftFrames): The spacecraft frames for the instrument

    Public Methods:
        load_response: Abstract method to compute the response matrix for a given time bin
    """
    def __init__(self, detectors, skygrid, spacecraft_frames):
        self.detectors = detectors
        self.skygrid = skygrid
        self.spacecraft_frames = spacecraft_frames

    @abstractmethod
    def load_response(self, tstart, tstop, mask=False, **kwargs):
        pass


class GBMResponse(BaseResponse):
    """Implementation of the GBM instrument response for the TargetedSearch

    Attributes:
        detectors (list[str]): List of detector names
        skygrid (Skygrid): Instance of Skygrid class with expected sky positions and other relevant structures
        spacecraft_frames (SpacecraftFrames): The spacecraft frames for the instrument
        t0 (float): Central search time
        templates_directory (str): String representing the location where response templates are stored
        delta (float): Angular displacement for rebuilding atmospheric scattering response

    Public Methods:
        load_response: Method to compute the response matrix for a given time bin
        load_direct_response: Method to load the direct response matrix for a given time bin
        load_atmospheric_response: Method to load the atmospheric response matrix for a given detector and azimuth
        get_available_azimuths: Method to check which azimuths are available in the templates for a given detector
    """
    zen_margin = np.radians(5.0)
    rocking_zen = np.radians(130.0)
    det_index = {'n0': 0, 'n1': 1, 'n2': 2, 'n3':3, 'n4': 4, 'n5': 5, 'n6': 6, 'n7': 7, 'n8': 8, 'n9': 9, 'na': 10, 'nb': 11, 'b0': 0, 'b1': 1}

    def __init__(self, detectors, skygrid, spacecraft_frames, t0, templates_directory, delta: float = np.radians(0.1), templates: list = None):
        """ Class constructor

        Args:
            detectors (list[str]): List of detector names
            skygrid (Skygrid): The skygrid this response should be generated over
            spacecraft_frames (SpacecraftFrame): The spacecraft frames for the instrument this response is being
                generated for
            t0 (float): The central time of the search
            templates_directory (str): String representing the path where the templates for the GBM response are stored
            delta (float): Angular displacement for rebuilding atmospheric scattering response
            templates (list): list of template IDs to use
        """
        super().__init__(detectors, skygrid, spacecraft_frames)
        self.t0 = t0
        self.delta = delta
        self.templates_directory = templates_directory
        self.available_azimuths = self.get_available_azimuths('n0')
        self.templates = templates

        self.direct = {}
        for detector in self.detectors:
            self.direct[detector] = self.load_direct_response(detector)

        self.cached = None
        self.cached_geo = None

    def load_response(self, tstart, tstop, mask=False):
        """Generates the response matrix for a given time bin

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin
            mask (bool): return earth mask with response matrix

        Returns:
            (tuple[ndarray]): tuple with matrices for instrument response and the Earth mask representing
                sky positions that were occulted by the earth at the specified bin
        """
        tcenter = 0.5 * (tstart + tstop) + self.t0
        tcenter = Time(tcenter, format='fermi')
        spacecraft_frame = self.spacecraft_frames.at(tcenter)
        geo_azimuth, geo_zenith, geo_radius = get_geo_coordinates(spacecraft_frame)

        if self.cached is None or angular_separation(geo_azimuth, 0.5 * np.pi - geo_zenith, *self.cached_geo) >= self.delta:
            # build reponse matrix from direct + atmospheric scattering components
            # when the cached matrix is None or the spacecraft has moved more than delta
            responses = []

            for detector in self.detectors:
                direct = self.direct[detector]
                atmo = self.load_atmospheric_response(detector, geo_azimuth, geo_zenith)
                responses.append(direct + atmo)

            response = np.concatenate(responses, axis=2)

            if self.templates:
                response = response[self.templates, :, :]

            self.cached = response
            self.cached_geo = (geo_azimuth, 0.5 * np.pi - geo_zenith)
        else:
            # otherwise retrieve the cached response matrix
            response = self.cached

        if mask:
            return response, create_earth_mask(self.skygrid._points, geo_azimuth, geo_zenith, geo_radius)
        return response

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
            geo_az (float): Azimuth
            geo_zen (float): Zenith

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
