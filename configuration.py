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
import yaml
import numpy as np

class DetectorConfiguration():
    """Class for the detector configuration

        Attributes:
        -----------
        channel_edges: list
            Values for the energy bin edges to be searched
        search_channels: list
            Indices of channels to be searched

        Public Methods:
        ---------------
        save:
            Save the Results to a yaml file
        to_dict:
            Convert the DetectorConfiguration to a pure dictionary

        Class Methods:
        ---------------
        create:
            Create a DetectorConfiguration object given a valid set of input parameters
        open:
            Open an existing configuration object in a .yaml file
        """
    def __init__(self, channel_edges, search_channels):
        self.channel_edges = channel_edges
        self.search_channels = search_channels


    def to_dict(self):
        return {
            'channel_edges': self.channel_edges,
            'search_channels': self.search_channels
        }


    def save(self, output_file):
        as_dict = self.to_dict()
        with open(output_file, 'w') as file:
            yaml.dump(as_dict, file)


    @classmethod
    def create(cls, detector_config):
        if not 'channel_edges' in detector_config:
            raise KeyError(f"Missing required configuration key: channel_edges")
        if not 'search_channels' in detector_config:
            raise KeyError(f"Missing required configuration key: search_channels")
        return cls(detector_config['channel_edges'], detector_config['search_channels'])


    @classmethod
    def open(cls, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"No such file: '{config_file}'")
        with open(config_file, 'r') as file:
            detector_config = yaml.safe_load(file)
            if not 'channel_edges' in detector_config:
                raise KeyError(f"Missing required configuration key: channel_edges")
            if not 'search_channels' in detector_config:
                raise KeyError(f"Missing required configuration key: search_channels")
            return cls(detector_config['channel_edges'], detector_config['search_channels'])


    def __getattr__(self, item):
        if item == 'channel_mask':
            return [channel in self.search_channels for channel in range(len(self.channel_edges) - 1)]


class InstrumentConfiguration():
    """Class for the instrument configuration

        Attributes:
        -----------
        detector_configs: dict
            Dictionary with keys being the detector names and values being the detector's corresponding configuration
        detectors: list
            Names of detectors from keys in detector_configs

        Public Methods:
        ---------------
        save:
            Save the configuration to a yaml file
        add_detector:
            Add a new detector and corresponding DetectorConfiguration
        to_dict:
            Convert the InstrumentConfiguration to a pure dictionary


        Class Methods:
        ---------------
        create:
            Create an InstrumentConfiguration object given a valid input dictionary with detectors and corresponding
            configurations
        open:
            Create an InstrumentConfiguration object given a valid YAML file with detectors and corresponding
            configurations
        """
    def __init__(self, detectors=None, detector_configs=None):
        self.detector_configs = {}
        if detectors and detector_configs:
            if(len(detectors) == len(detector_configs)):
                for detector, detector_config in zip(detectors, detector_configs):
                    if not isinstance(detector, str):
                        raise ValueError(f"Detector is not of type string. Please check your inputs.")
                    self.detector_configs[detector] = detector_config
            else:
                raise ValueError(f"Length mismatch: {len(detectors)} != {len(detector_configs)}")


    def add_detector(self, detector_name, detector_config):
        if not isinstance(detector_config, DetectorConfiguration):
            raise ValueError(f"Input detector configuration must be of type DetectorConfiguration")
        self.detector_configs[detector_name] = detector_config


    def to_dict(self):
        detector_config_dict = {}
        for detector in self.detectors:
            detector_config_dict[detector] = self.detector_configs[detector].to_dict()

        return { 'detector_configs': detector_config_dict }


    def save(self, output_file):
        as_dict = self.to_dict()
        with open(output_file, 'w') as file:
            yaml.dump(as_dict, file)


    @classmethod
    def create(cls, instrument_config):
        configured_instrument = cls()
        if not 'detector_configs' in instrument_config:
            raise KeyError(f"Missing required configuration key: detector_configs")
        for detector in instrument_config['detector_configs']:
            detector_config = instrument_config['detector_configs'][detector]
            if isinstance(detector_config, DetectorConfiguration):
                configured_instrument.add_detector(detector, detector_config)
            if isinstance(detector_config, dict):
                configured_instrument.add_detector(detector, DetectorConfiguration.create(detector_config))
        return configured_instrument


    @classmethod
    def open(cls, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"No such file: '{config_file}'")
        else:
            configured_instrument = cls()
            with open(config_file, 'r') as file:
                instrument_config = yaml.safe_load(file)
                detector_configs = instrument_config['detector_configs']
                for det in detector_configs:
                    configured_instrument.add_detector(det, DetectorConfiguration.create(detector_configs[det]))
                return configured_instrument

    def __getattr__(self, item):
        if item == 'detectors':
            return list(self.detector_configs.keys())
        if item == 'channel_mask':
            return np.ravel([self.detector_configs[det].channel_mask for det in self.detectors])
        if item == 'search_channels':
            return { det: self.detector_configs[det].search_channels for det in self.detectors }
        if item == 'channel_edges':
            return { det: self.detector_configs[det].channel_edges for det in self.detectors }


class SearchConfiguration():
    """Class for the search configuration

        Attributes:
        -----------
        search_settings: dict
            Dictionary representing a collection of attributes required to run the targeted search
        instrument_configs: dict
            Dictionary where key names are the instrument and the values are their respective configurations
        instruments: list
            Names of available instruments from keys in instrument_configs
        reference_instrument: String
            Key of the instrument to be used as a reference during the search
        time_range: np.array
            Time range associated with the search

        Public Methods:
        ---------------
        save:
            Save the Results to a yaml file
        to_dict:
            Convert the SearchConfiguration to a pure dictionary
        add_instrument:
            Add a new instrument and corresponding InstrumentConfiguration

        Class Methods:
        ---------------
        create:
            Create a Configuration object given a valid set of input parameters
        open:
            Open an existing configuration object in a .yaml file
        build_search_settings:
            Return a valid search configuration dictionary given a set of input parameters
        validate_search_settings:
            Return if the given search_settings dictionary contains the necessary keys
        """
    def __init__(self, search_settings, reference_instrument='', instruments=None, instrument_configs=None):
        self.instrument_configs = {}
        if instruments and instrument_configs:
            if(len(instruments) == len(instrument_configs)):
                for instrument, instrument_config in zip(instruments, instrument_configs):
                    self.instrument_configs[instrument] = instrument_config
            else:
                raise ValueError(f"Length mismatch: {len(instruments)} != {len(instrument_configs)}")
        if self.validate_search_settings(search_settings):
            self.search_settings = search_settings
        else:
            raise ValueError(f"One of the required search_settings parameters has not been set")
        self.reference_instrument = reference_instrument


    def add_instrument(self, instrument, instrument_config):
        if not isinstance(instrument_config, InstrumentConfiguration):
            raise ValueError(f"Input instrument configuration must be of type InstrumentConfiguration")
        self.instrument_configs[instrument] = instrument_config


    def to_dict(self):
        instrument_config_dict = {}
        for instrument in self.instruments:
            instrument_config_dict[instrument] = self.instrument_configs[instrument].to_dict()

        return {
            'instrument_configs': instrument_config_dict,
            'reference_instrument': self.reference_instrument,
            'search_settings': self.search_settings
        }


    def save(self, output_file):
        as_dict = self.to_dict()
        with open(output_file, 'w') as file:
            yaml.dump(as_dict, file)

    @classmethod
    def create(cls, search_config):
        if not 'instrument_configs' in search_config:
            raise KeyError(f"Missing required configuration key: instrument_configs")
        if not 'reference_instrument' in search_config:
            raise KeyError(f"Missing required configuration key: reference_instrument")
        if not 'search_settings' in search_config:
            raise KeyError(f"Missing required configuration key: search_settings")
        configured_search = cls(search_config['search_settings'], search_config['reference_instrument'])
        for instrument in search_config['instrument_configs']:
            instrument_config = search_config['instrument_configs'][instrument]
            if isinstance(instrument_config, InstrumentConfiguration):
                configured_search.add_instrument(instrument, instrument_config)
            if isinstance(instrument_config, dict):
                configured_instrument.add_instrument(instrument, InstrumentConfiguration.create(instrument_config))
        return configured_search

    @classmethod
    def open(cls, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"No such file: '{config_file}'")
        else:
            with open(config_file, 'r') as file:
                search_config = yaml.safe_load(file)

                configured_search = cls(search_config['search_settings'], search_config['reference_instrument'])
                instrument_configs = search_config['instrument_configs']
                for instrument in instrument_configs:
                    configured_search.add_instrument(instrument, InstrumentConfiguration.create(instrument_configs[instrument]))
                return configured_search


    @classmethod
    def build_search_settings(
        cls,
        win_width=60,
        min_loglr=5,
        min_dur=0.064,
        max_dur=8.192,
        min_step=0.064,
        num_steps=8,
        threshold=5.0,
        skygrid_resolution=5
    ):
        """
        Creates a dictionary of parameters with their corresponding values.

        Args:
            win_width (int): Width of the window. Default: 60
            min_loglr (int): Minimum log likelihood ratio. Default: 5
            min_dur (float): Minimum duration in seconds. Default: 0.064
            max_dur (float): Maximum duration in seconds. Default: 8.192
            min_step (float): Minimum step size in seconds. Default: 0.064
            num_steps (int): Number of steps. Default: 8
            threshold (float): Threshold value. Default: 5.0
            skygrid_resolution (int): Resolution for skygrid. Default: 5

        Returns:
            dict: A dictionary with parameter names as keys and their values
        """

        search_settings = {
            'win_width': win_width,
            'min_loglr': min_loglr,
            'min_dur': min_dur,
            'max_dur': max_dur,
            'min_step': min_step,
            'num_steps': num_steps,
            'threshold': threshold,
            'skygrid_resolution': skygrid_resolution
        }

        return search_settings


    @classmethod
    def validate_search_settings(cls, search_settings):
        return 'win_width' in search_settings and isinstance(search_settings['win_width'], int) and \
               'min_loglr' in search_settings and isinstance(search_settings['min_loglr'], int) and \
               'min_dur' in search_settings and isinstance(search_settings['min_dur'], float) and \
               'max_dur' in search_settings and isinstance(search_settings['max_dur'], float) and \
               'min_step' in search_settings and isinstance(search_settings['min_step'], float) and \
               'num_steps' in search_settings and isinstance(search_settings['num_steps'], int) and \
               'threshold' in search_settings and isinstance(search_settings['threshold'], float) and \
               'skygrid_resolution' in search_settings and isinstance(search_settings['skygrid_resolution'], int)


    def __getattr__(self, item):
        if item == 'instruments':
            return list(self.instrument_configs.keys())
        if item == 'time_range':
            max_dur = self.search_settings['max_dur']
            win_width = self.search_settings['win_width']
            return np.array([-1, 1]) * max([0.5 * win_width + max_dur + 1.024, 30])
        if item in self.search_settings.keys():
            return self.search_settings[item]