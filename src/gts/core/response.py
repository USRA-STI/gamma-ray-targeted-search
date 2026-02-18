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
from abc import ABC, abstractmethod


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
