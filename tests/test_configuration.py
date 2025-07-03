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
import copy
import configuration
import unittest
import numpy as np


class TestBaseConfiguration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
       cls.ref_kwargs = {"test_str": "test", "test_int": 1, "test_float": 5.0}
       cls.test_file = os.path.join(test_dir, "base.yml")

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

       with self.assertRaises(KeyError):
           config["bad_key"]

    def test_write(self):
       ref_config = configuration.BaseConfiguration(**self.ref_kwargs)
       ref_config.write(self.test_file)

       config = configuration.BaseConfiguration.open(self.test_file)
       self.assertEqual(config.keys(), ref_config.keys())
       for ref_key, ref_value in self.ref_kwargs.items():
           self.assertEqual(config[ref_key], ref_value)

       with self.assertRaises(FileNotFoundError):
           config = configuration.BaseConfiguration.open("bad_path")

class TestInstrumentConfiguration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
       cls.ref_kwargs = {
           "instrument_name": "test",
           "detectors": {
               "t0": {"channel_edges": [0, 1, 2], "search_channels": [0, 1]},
               "t1": {"channel_edges": [3, 4, 5], "search_channels": [1]},
           }
       }
       cls.test_file = os.path.join(test_dir, "instrument.yml")

    @classmethod
    def tearDownClass(cls) -> None:
       if os.path.exists(cls.test_file):
           os.remove(cls.test_file)

    def test_add_detector(self):
       config = configuration.InstrumentConfiguration(**copy.deepcopy(self.ref_kwargs))

       # add a new detector
       config.add_detector("t2", copy.deepcopy(config.settings["detectors"]["t0"]))
       self.assertEqual(config["detector_names"], ["t0", "t1", "t2"])

       # replace existing detector
       with self.assertWarns(Warning):
           config.add_detector("t0", copy.deepcopy(config.settings["detectors"]["t0"]))

    def test_derived_keys(self):
       config = configuration.InstrumentConfiguration(**self.ref_kwargs)

       self.assertEqual(config["detector_names"], ["t0", "t1"])
       self.assertEqual(config["channel_edges"], {"t0": [0, 1, 2], "t1": [3, 4, 5]})
       self.assertEqual(config["search_channels"], {"t0": [0, 1], "t1": [1]})
       self.assertTrue(np.all(config["channel_mask"] == np.array([True, True, False, True])))

    def test_write(self):
       ref_config = configuration.InstrumentConfiguration(**self.ref_kwargs)
       ref_config.write(self.test_file)

       config = configuration.InstrumentConfiguration.open(self.test_file)
       self.assertEqual(config.keys(), ref_config.keys())
       for ref_key, ref_value in self.ref_kwargs.items():
           self.assertEqual(config[ref_key], ref_value)

    def test_validate(self):
       config = configuration.InstrumentConfiguration(**copy.deepcopy(self.ref_kwargs))
       good_settings = copy.deepcopy(config.settings)

       # test for incorrect type
       config.settings["instrument_name"] = 0
       with self.assertRaises(ValueError):
           config.validate()
       config.settings = copy.deepcopy(good_settings)

       # test for missing settings key
       config.settings.pop("instrument_name")
       with self.assertRaises(ValueError):
           config.validate()
       config.settings = copy.deepcopy(good_settings)

       # test for missing detector key
       config.settings["detectors"]["t0"].pop("search_channels")
       with self.assertRaises(ValueError):
           config.validate()
       config.settings = copy.deepcopy(good_settings)

       # test that search_channels is a list of ints
       config.settings["detectors"]["t0"]["search_channels"] = 0
       with self.assertRaises(ValueError):
           config.validate()
       config.settings = copy.deepcopy(good_settings)

       config.settings["detectors"]["t0"]["search_channels"] = [0.0, 1.0, 2.0]
       with self.assertRaises(ValueError):
           config.validate()


class TestSearchConfiguration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
       ref_instrument = configuration.InstrumentConfiguration(
           instrument_name= "test", detectors={
               "t0": {"channel_edges": [0, 1, 2], "search_channels": [0, 1]},
               "t1": {"channel_edges": [3, 4, 5], "search_channels": [1]},
           })
       cls.ref_kwargs = {"win_width": 60.0, "instruments": [ref_instrument]}
       cls.test_file = os.path.join(test_dir, "search.yml")

    @classmethod
    def tearDownClass(cls) -> None:
       if os.path.exists(cls.test_file):
           os.remove(cls.test_file)

    def test_add_instrument(self):
       config = configuration.SearchConfiguration(**copy.deepcopy(self.ref_kwargs))

       # add a new instrument
       new = copy.deepcopy(config.settings["instruments"][0])
       new.settings["instrument_name"] = "new"
       config.add_instrument(new)
       self.assertEqual(config["instrument_names"], ["test", "new"])

       # replace existing instrument
       with self.assertWarns(Warning):
           config.add_instrument(copy.deepcopy(config.settings["instruments"][0]))

    def test_get_instrument(self):
       config = configuration.SearchConfiguration(**copy.deepcopy(self.ref_kwargs))
       instrument = config.get_instrument("test")
       self.assertTrue(isinstance(instrument, configuration.InstrumentConfiguration))

       with self.assertRaises(SystemExit):
           instrument = config.get_instrument("bad")

    def test_derived_keys(self):
       config = configuration.SearchConfiguration(**self.ref_kwargs)

       self.assertEqual(config["instrument_names"], ["test"])
       self.assertEqual(config["reference_instrument"], "test")
       self.assertEqual(config["time_range"][0], -30.0)
       self.assertEqual(config["time_range"][1], +30.0)

    def test_write(self):
       ref_config = configuration.SearchConfiguration(**self.ref_kwargs)
       ref_config.write(self.test_file)

       config = configuration.SearchConfiguration.open(self.test_file)
       self.assertEqual(config.keys(), ref_config.keys())
       for ref_key, ref_value in self.ref_kwargs.items():
           if ref_key == "instruments":
               for inst_key, inst_value in ref_value[0].settings.items():
                   self.assertEqual(config["instruments"][0][inst_key], inst_value)
           else:
               self.assertEqual(config[ref_key], ref_value)

    def test_validate(self):
       config = configuration.SearchConfiguration(**copy.deepcopy(self.ref_kwargs))
       good_settings = copy.deepcopy(config.settings)

       config.settings["num_steps"] = 8.0
       with self.assertRaises(ValueError):
           config.validate()
       config.settings = copy.deepcopy(good_settings)

       config.settings["win_width"] = "60"
       with self.assertRaises(ValueError):
           config.validate()
       config.settings = copy.deepcopy(good_settings)

       config.settings["instruments"] = None
       with self.assertRaises(ValueError):
           config.validate()
       config.settings = copy.deepcopy(good_settings)

       config.settings["instruments"][0] = "bad"
       with self.assertRaises(ValueError):
           config.validate()
