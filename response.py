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
    def load_response(self, tstart, tstop):
        pass


class GBMResponse(BaseResponse):
    """Implementation of the GBM instrument response for the TargetedSearch

    Attributes:
        detectors (list[str]): List of detector names
        skygrid (Skygrid): Instance of Skygrid class with expected sky positions and other relevant structures
        spacecraft_frames (SpacecraftFrames): The spacecraft frames for the instrument
        t0 (float): Central search time
        templates_directory (str): String representing the location where response templates are stored

    Public Methods:
        load_response: Method to compute the response matrix for a given time bin
        load_direct_template: Method to compute the direct response matrix for a given time bin
        load_atmo_template: Method to compute the atmospheric response matrix for a given detector and azimuth
        get_available_azimuths: Method to check which azimuths are available in the templates for a given detector
        get_atmospheric_response: Method to compute the atmospheric response matrix for a given detector at a given
            azimuth and zenith position
    """
    zen_margin = 5.0
    rocking_zen = 130.0
    det_index = {'n0': 0, 'n1': 1, 'n2': 2, 'n3':3, 'n4': 4, 'n5': 5, 'n6': 6, 'n7': 7, 'n8': 8, 'n9': 9, 'na': 10, 'nb': 11, 'b0': 0, 'b1': 1}

    def __init__(self, detectors, skygrid, spacecraft_frames, t0, templates_directory):
        """ Class constructor

        Args:
            detectors (list[str]): List of detector names
            skygrid (Skygrid): The skygrid this response should be generated over
            spacecraft_frames (SpacecraftFrame): The spacecraft frames for the instrument this response is being
                generated for
            t0 (float): The central time of the search
            templates_directory (str): String representing the path where the templates for the GBM response are stored
        """
        super().__init__(detectors, skygrid, spacecraft_frames)
        self.t0 = t0
        self.templates_directory = templates_directory

    def load_response(self, tstart, tstop):
        """Generates the response matrix for a given time bin

        Args:
            tstart (float): Start of the time bin
            tstop (float): End of the time bin

        Returns:
            (tuple[ndarray]): tuple with matrices for instrument response and the Earth mask representing
                sky positions that were occulted by the earth at the specified bin
        """
        tcenter = 0.5 * (tstart + tstop) + self.t0
        tcenter = Time(tcenter, format='fermi')
        spacecraft_frame = self.spacecraft_frames.at(tcenter)
        geo_azimuth, geo_zenith, geo_radius = get_geo_coordinates(spacecraft_frame)

        responses = []

        for detector in self.detectors:
            direct = self.load_direct_template(detector)
            atmo = self.get_atmospheric_response(detector, geo_azimuth, geo_zenith)
            responses.append(direct + atmo)

        response = np.stack(responses, axis=2)

        earthmask = create_earth_mask(self.skygrid._points, geo_azimuth, geo_zenith, geo_radius)

        # TODO Hack. This needs to be changed
        response = response[0:3, :, :, :]

        return response, earthmask

    def load_direct_template(self, detector):
        """Loads the direct response matrix for a specific detector

        Args:
            detector (str): The name of the detector

        Returns:
            (ndarray): The direct response matrix/array for one detector
        """
        template_file = os.path.join(self.templates_directory, 'direct', f"{getGbmDetectorType(detector)}.npy")
        data = self.swap_cols(np.load(template_file))
        return data[:, :, self.det_index[detector]]

    def get_atmospheric_response(self, detector, geo_az, geo_zen):
        """Loads the atmospheric response matrix for a specific detector

        Args:
            detector (str): The name of the detector
            geo_az (float): Azimuth
            geo_zen (float): Zenith

        Returns:
            (ndarray): The atmospheric response matrix/array for one detector
        """
        if np.abs(geo_zen - self.rocking_zen) > self.zen_margin:
            template_file = os.path.join(self.templates_directory, 'atmo_' + getGbmDetectorType(detector), 'atmrates_az0_zen130.npy')
            original_data = self.swap_cols(np.load(template_file))
            zero_shape = original_data[:, :, self.det_index[detector]].shape
            return np.zeros(zero_shape)

        azimuths = self.get_available_azimuths(detector)
        azimuths = np.array([355.0, 360.0]) if geo_az > 355 else azimuths

        idx = np.argsort(np.abs(geo_az - azimuths))[:2]
        nearest_az = azimuths[idx]
        nearest_az[nearest_az == 360.0] = 0.0

        # need to check why we're swapping columns and not swapping back
        responses = self.swap_cols([self.load_atmo_template(detector, a) for a in nearest_az])
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
        path = os.path.join(self.templates_directory, 'atmo_' + getGbmDetectorType(detector))
        files = os.listdir(path)
        azimuths = [float(f.split('_')[1][2:]) for f in files if f.startswith('atmrates_az')]
        return np.array(sorted(azimuths))

    def load_atmo_template(self, detector, azimuth):
        """Loads the atmospheric response matrix for a specific detector at a given azimuth

        Args:
            detector (str): The name of the detector
            azimuth (float): Azimuth

        Returns:
            (ndarray): The atmospheric response matrix/array for one detector at a given azimuth
        """
        file_path = os.path.join(self.templates_directory, 'atmo_' + getGbmDetectorType(detector), f'atmrates_az{int(azimuth)}_zen130.npy')
        data = self.swap_cols(np.load(file_path))
        return data[:, :, self.det_index[detector]]

    def swap_cols(rsp):
        """Swaps old response matrix format for the new column ordering

        Args:
            rsp (ndarray): Response matrix

        Returns:
           (ndarray): Response matrix with swapped columns for detectors and energy bins
        """
        ntemplate, nsky, nene, ndet = rsp.shape
        swapped = np.zeros((ntemplate, nsky, ndet, nene), dtype=rsp.dtype)
        for det in range(ndet):
            for ene in range(nene):
                swapped[:,:,det,ene] = rsp[:,:,ene,det]
        return swapped

    def getGbmDetectorType(detector):
        """Gets the type of a specified detector according to GBM standards

        Args:
            detector (str): Detector name

        Returns:
            (str): Detector type
        """
        if GbmDetectors.from_str(detector).is_nai():
            det_type = 'nai'
        elif GbmDetectors.from_str(detector).is_bgo():
            det_type = 'bgo'
        else:
            raise ValueError(f'Detector {detector} not recognized.')

        return det_type
