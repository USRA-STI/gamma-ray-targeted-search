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
import sys
import copy
import unittest
import numpy as np

from gts.core import utils
from gdt.core.coords.spacecraft import SpacecraftFrame
from gdt.core.coords.quaternion import Quaternion
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.constants import c

test_dir = os.path.dirname(os.path.abspath(__file__))


class TestSkyGrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.skygrid = utils.SkyGrid(5.0)

    def test_radians(self):
        ref = [[0.,         0., 1.04719755,  2.0943951, 3.14159265,  4.1887902, 5.23598776,         0., 0.52359878, 1.04719755],
               [0., 0.08726646, 0.08726646, 0.08726646, 0.08726646, 0.08726646, 0.08726646, 0.17453293, 0.17453293, 0.17453293]]
        for i in range(10):
            self.assertAlmostEqual(self.skygrid.radians[0][i], ref[0][i])
            self.assertAlmostEqual(self.skygrid.radians[1][i], ref[1][i])

    def test_degrees(self):
        ref = [[0.,  0., 60., 120., 180., 240., 300.,  0., 30., 60.],
               [0.,  5.,  5.,   5.,   5.,   5.,   5., 10., 10., 10.]]
        for i in range(10):
            self.assertAlmostEqual(self.skygrid.degrees[0][i], ref[0][i])
            self.assertAlmostEqual(self.skygrid.degrees[1][i], ref[1][i])

    def test_size(self):
        self.assertEqual(self.skygrid.size, 1634)


class TestMethods(unittest.TestCase):

    def test_relative_timeoffset(self):

        ref_frame = SpacecraftFrame(
            obsgeoloc=CartesianRepresentation(-3219884.19003963, 6096097.16599464, 476118.15179503, unit='m'),
            quaternion=Quaternion([0.34885525, -0.10563145, -0.89395416, -0.26074501]))

        ref_distance = np.linalg.norm(ref_frame.obsgeoloc.xyz)

        # test source positions at Earth center, 90 deg from Earth center,
        # anti-Earth center, and 90 deg from anti-Earth center
        ref_coord = SkyCoord(
            az=[140.2, 140.2, 320.2, 320.2], el=90 - np.array([130, 40, 50, 140]),
            frame=ref_frame, unit='deg')

        # frame of second spacecraft at Earth center for easy testing
        frame = SpacecraftFrame()

        dt = utils.relative_time_offset(frame, ref_coord)

        self.assertAlmostEqual(dt[0], -(ref_distance / c).value) # plane wave arrives at Earth center first
        self.assertAlmostEqual(dt[1], 0.0, 3)                    # perpendicular to both frames
        self.assertAlmostEqual(dt[2], (ref_distance / c).value)  # plane wave arrives at reference frame first
        self.assertAlmostEqual(dt[3], 0.0, 3)                    # perpendicular to both frames
