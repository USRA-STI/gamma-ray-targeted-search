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
import warnings


class InstrumentConfiguration():
    """Class for the instrument configuration

        Attributes:
        -----------
        detector_configs: dict
            Dictionary with keys being the detector names and values being the detector's corresponding configuration
        detectors: list
            Names of detectors from keys in detector_configs
        name: str
            Instrument name
        channel_mask: ndarray
            Array representing valid channels for each detector according to search criteria
        search_channels: dict
            Dict where keys are detector names and values are the search channels to be used for that detector
        channel_edges: dict
            Dict where keys are detector names and values are the channel edges for each detector


        Public Methods:
        ---------------
        save:
            Save the configuration to a yaml file
        add_detector:
            Add a new detector and corresponding configuration
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
    def __init__(self, instrument_name=None, detector_configs=None):
        """ Class constructor
        Args:
            instrument_name (str): Instrument name
            detector_configs (dict): Dictionary representing the configuration for each detector. Keys are detector names
                and values are detector configurations. Each detector should have set at minimum the channel edges and
                the search channels to be used

        Returns:
            None
        """
        self.detector_configs = {}
        if detector_configs and isinstance(detector_configs, dict):
            for detector, detector_config in detector_configs.items():
                if not isinstance(detector, str):
                    raise ValueError(f"Detector name is not of type string. Please check your inputs.")
                try:
                    self._validate_detector_config(detector_config)
                except:
                    raise ValueError(f"Detector {detector} config not valid. Please ensure necessary keys and"
                                     f"values are properly configured.")
                self.detector_configs[detector] = detector_config
        if instrument_name and isinstance(instrument_name, str):
            self.name = instrument_name
        else:
            raise ValueError(f"Instrument must have an assigned name of type string for reference")


    def add_detector(self, detector_name, detector_config):
        """Add a new detector configuration

        Args:
            detector_name (str): Detector name
            detector_config (dict): Dictionary representing the configuration for this detector. Must contain keys for
                channel_edges and search_channels

        Returns:
            None
        """
        self._validate_detector_config(detector_config)
        self.detector_configs[detector_name] = detector_config


    def to_dict(self):
        """Convert the InstrumentConfiguration to a pure dictionary

        Args:
            None

        Returns:
            (dict): Dictionary with all the individual detectors' configurations stored (nested)
        """
        detector_config_dict = {}
        for detector in self.detectors:
            detector_config_dict[detector] = self.detector_configs[detector].to_dict()

        return { 'detector_configs': detector_config_dict }


    def save(self, output_file):
        """Save the instrument configuration to a file

        Args:
            output_file (str): Path to the output file

        Returns:
            None
        """
        as_dict = self.to_dict()
        with open(output_file, 'w') as file:
            yaml.dump(as_dict, file)


    @classmethod
    def create(cls, instrument_config):
        """Create a new instance of InstrumentConfiguration given a dictionary input

        Args:
            instrument_config (dict): Dictionary in the format output by self.to_dict()

        Returns:
            configured_instrument (InstrumentConfiguration): Instance of self configured as desired
        """
        configured_instrument = cls()
        if not 'detector_configs' in instrument_config:
            raise KeyError(f"Missing required configuration key: detector_configs")
        for detector in instrument_config['detector_configs']:
            detector_config = instrument_config['detector_configs'][detector]
            configured_instrument.add_detector(detector, detector_config)
        return configured_instrument


    @classmethod
    def open(cls, config_file):
        """Create a new instance of InstrumentConfiguration given a input file

        Args:
            config_file (str): Path to configuration file

        Returns:
            configured_instrument (InstrumentConfiguration): Instance of self configured as desired
        """
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
            return np.ravel([self._get_detector_channel_mask(det) for det in self.detectors])
        if item == 'search_channels':
            return { det: self.detector_configs[det]['search_channels'] for det in self.detectors }
        if item == 'channel_edges':
            return { det: self.detector_configs[det]['channel_edges'] for det in self.detectors }
        if item == 'name':
            return self.name


    def _validate_detector_config(self, detector_config):
        """Ensure input detector configuration meets expected structure

        Args:
            detector_config (dict): Dictionary representing a detector configuration

        Returns:
            None
        """
        if not isinstance(detector_config, dict):
            raise ValueError(f"Input detector configuration must be a dictionary")
        if 'channel_edges' not in detector_config or not isinstance(detector_config['channel_edges'], list):
            raise ValueError(f"Input detector configuration must contain a key channel_edges with a value of type list")
        if 'search_channels' not in detector_config or not isinstance(detector_config['search_channels'], list):
            raise ValueError(f"Input detector configuration must contain a key search_channels with a value of type list")


    def _get_detector_channel_mask(self, detector):
        """Extract the channel mask for a specific detector

        Args:
            detector (str): Detector name/key

        Returns:
            (list[int]): List including only desired channels
        """
        if detector not in self.detector_configs:
            raise ValueError(f"Requested detector is not in instrument's detector configurations")
        else:
            config = self.detector_configs[detector]
            search_channels = config['search_channels']
            channel_edges = config['channel_edges']

            return [channel in search_channels for channel in range(len(channel_edges) - 1)]


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
        win_width: float
            Search window width
        min_loglr: float
            Minimum log-likelihood ratio to be valid
        min_dur: float
            Minimum bin duration
        max_dur: float
            Maximum bin duration
        min_step: float
            Minimum step between bin durations
        num_steps: int
            Number of steps
        threshold: float
            Threshold value
        skygrid_resolution: float
            Resolution of the sky grid

        Public Methods:
        ---------------
        save:
            Save the Results to a yaml file
        to_dict:
            Convert the SearchConfiguration to a pure dictionary
        add_instrument:
            Add a new instrument and corresponding InstrumentConfiguration
        get_instrument_config:
            Get the instance of a specified instrument's InstrumentConfiguration

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
    def __init__(self, search_settings, instrument_configs=None):
        """ Class constructor
        Args:
            search_settings (dict): A set of key-value pairs for the search settings. See self.validate_search_settings
                for required keys
            instrument_configs (list): A list of instrument configurations

        Returns:
            None
        """
        self.instrument_configs = {}
        if len(instrument_configs):
            self.instrument_configs = instrument_configs
        else:
            raise ValueError(f"There must be at least one instrument assigned to this search tool.")
        if self.validate_search_settings(search_settings):
            self.search_settings = search_settings
        else:
            raise ValueError(f"One of the required search_settings parameters has not been set")


    def add_instrument(self, instrument_config):
        """Adds an instrument configuration to this search configuration

        Args:
            instrument_config (InstrumentConfiguration): An instrument configuration

        Returns:
            None
        """
        if not isinstance(instrument_config, InstrumentConfiguration):
            raise ValueError(f"Input instrument configuration must be of type InstrumentConfiguration")
        i = self._get_existing_index(instrument_config.name)
        if i:
            self.instrument_configs[i] = instrument_config
            warnings.warn(f"Existing configuration replaced for {instrument_config.name}")
        else:
            self.instrument_configs.append(instrument_config)


    def to_dict(self):
        """Converts the SearchConfiguration to a dictionary

        Args:
            None

        Returns:
            Dictionary containing the instrument configurations, reference instrument, and search_settings from this instance
        """
        instrument_config_dict = {}
        for instrument in self.instruments:
            instrument_config_dict[instrument] = self.instrument_configs[instrument].to_dict()

        return {
            'instrument_configs': instrument_config_dict,
            'reference_instrument': self.reference_instrument,
            'search_settings': self.search_settings
        }


    def save(self, output_file):
        """Saves the SearchConfiguration to a file

        Args:
            output_file (str): Path to the output file

        Returns:
            None
        """
        as_dict = self.to_dict()
        with open(output_file, 'w') as file:
            yaml.dump(as_dict, file)

    @classmethod
    def create(cls, search_config):
        """Creates a new SearchConfiguration instance using an input dictionary

        Args:
            search_config (dict): Dictionary requiring keys for instrument_configs, reference_instrument, and search_settings

        Returns:
            SearchConfiguration object instantiated with input configuration
        """
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
        """Creates an instance of SearchConfiguration using a specified input file

        Args:
            config_file (str): Path to configuration file

        Returns:
            SearchConfiguration object instantiated with input configuration
        """
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
        skygrid_resolution=5):
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
        """Ensures a given search_settings dictionary contains the required keys

        Args:
            search_settings (dict): Input search settings dictionary to validate

        Returns:
            (bool): Is the input a valid search settings dictionary
        """
        return 'win_width' in search_settings and isinstance(search_settings['win_width'], int) and \
               'min_loglr' in search_settings and isinstance(search_settings['min_loglr'], int) and \
               'min_dur' in search_settings and isinstance(search_settings['min_dur'], float) and \
               'max_dur' in search_settings and isinstance(search_settings['max_dur'], float) and \
               'min_step' in search_settings and isinstance(search_settings['min_step'], float) and \
               'num_steps' in search_settings and isinstance(search_settings['num_steps'], int) and \
               'threshold' in search_settings and isinstance(search_settings['threshold'], float) and \
               'skygrid_resolution' in search_settings and isinstance(search_settings['skygrid_resolution'], int)


    def get_instrument_config(self, instrument_name):
        """Extracts a specified instrument's corresponding InstrumentConfiguration

        Args:
            instrument_name (str): The name of the instrument

        Returns:
            (InstrumentConfiguration): The instance of InstrumentConfugration that corresponds to the input instrument
        """
        i = self._get_existing_index(instrument_name)
        if i is not None:
            return self.instrument_configs[i]
        else:
            raise KeyError(f"Key {instrument_name} does not exist in instrument configs")


    def _get_existing_index(self, instrument_name):
        """Returns the index in instrument_configs of a requested instrument

        Args:
            instrument_name (str): The name of the requested instrument

        Returns:
            (int): The index corresponding to this instrument's configuration, or None if the specified instrument isn't found
        """
        for i, config in enumerate(self.instrument_configs):
            if config.name == instrument_name:
                return i
        return None


    def __getattr__(self, item):
        if item == 'instruments':
            return list(self.instrument_configs.keys())
        if item == 'time_range':
            max_dur = self.search_settings['max_dur']
            win_width = self.search_settings['win_width']
            return np.array([-1, 1]) * max([0.5 * win_width + max_dur + 1.024, 30])
        if item in self.search_settings.keys():
            return self.search_settings[item]
        if item == 'reference_instrument':
            return self.instrument_configs[0].name