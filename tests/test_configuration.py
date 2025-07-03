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
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(test_dir , '..')) # to be removed when gts will be installed as a module
import configuration
import unittest


class TestBaseConfiguration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
       cls.ref_kwargs = {"test_str": "test", "test_int": 1, "test_float": 5.0}
       cls.test_file = os.path.join(test_dir, "test.yml")

    @classmethod
    def tearDownClass(cls) -> None:
       if os.path.exists(cls.test_file):
           os.remove(cls.test_file)

    def test_keys(self):
       config = configuration.BaseConfiguration(**self.ref_kwargs)
       self.assertTrue(config.keys() == list(self.ref_kwargs.keys()))

    def test_validate(self):
       bad_config = configuration.BaseConfiguration(dummy=0)
       bad_config.settings = "not_a_dict"
       with self.assertRaises(ValueError):
           bad_config.validate()

    def test_getitem(self):
       config = configuration.BaseConfiguration(**self.ref_kwargs)
       for ref_key, ref_value in self.ref_kwargs.items():
           self.assertEqual(config[ref_key], ref_value)

    def test_write(self):
       ref_config = configuration.BaseConfiguration(**self.ref_kwargs)
       ref_config.write(self.test_file)

       config = configuration.BaseConfiguration.open(self.test_file)
       self.assertEqual(config.keys(), ref_config.keys())
       for ref_key, ref_value in self.ref_kwargs.items():
           self.assertEqual(config[ref_key], ref_value)
