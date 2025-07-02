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

from abc import ABC, abstractmethod


class BaseConfiguration(ABC):
    """A base class for configuration objects.

    Note:
        This class should not be directly instantiated, but rather inherited.
        The inherited class should define a method called ``validate()``
        to enforce the required format of the config dictionary.
    """
    _derived_keys = []

    def __init__(self, **kwargs):
        """Class constructor

        Args:
            kwargs (dict): Keyword dictionary with configuration parameters.
        """
        self.config = kwargs
        self.validate()

    def keys(self):
        """(list): Method to retrieve available configuration keys"""
        return list(self.config.keys()) + self._derived_keys

    def __getitem__(self, key):
        """Method for high level access to config dict and derived keys"""
        if key not in self.keys():
            raise KeyError(f"{key} is not a valid key.")
        if key in self._derived_keys:
            return getattr(self, key)
        return self.config[key]

    def save(self, path):
        """Save the instrument configuration to a file

        Args:
            path (str): Path to the output file

        Returns:
            None
        """
        with open(path, 'w') as file:
            file.write(f"# {type(self)}\n")
            yaml.dump(self.config, file, default_flow_style=None, sort_keys=False)

    @classmethod
    def open(cls, path):
        """Create a new instance of InstrumentConfiguration given a input file

        Args:
            path (str): Path to configuration file

        Returns:
            configured_instrument (InstrumentConfiguration): Instance of self configured as desired
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such file: '{config_file}'")
        else:
            with open(path, 'r') as file:
                config = yaml.safe_load(file)
                return cls(**config)

    @abstractmethod
    def validate(self):
        """This method needs to be defined by the inheriting class. The method
        should check if the config dictionary has the correct format and
        raise errors when the format checks fail"""
        pass


class InstrumentConfiguration(BaseConfiguration):
    """Class for the instrument configuration

        Attributes:
        -----------
        config: dict
            Instrument configuration dictionary with instrument name + detector configurations.
        detector_names: list
            Names of detectors included in this configuration
        channel_edges: dict
            Dict where keys are detector names and values are the channel edges for each detector
        channel_mask: ndarray
            Array representing valid channels for each detector according to search criteria
        search_channels: dict
            Dict where keys are detector names and values are the search channels to be used for that detector


        Public Methods:
        ---------------
        save:
            Save the configuration to a yaml file
        add_detector:
            Add a new detector and corresponding configuration


        Class Methods:
        ---------------
        open:
            Create an InstrumentConfiguration object given a valid YAML file with detectors and corresponding
            configurations
    """
    _derived_keys = ['detector_names', 'channel_edges', 'channel_mask', 'search_channels']

    def __init__(self, instrument_name=None, detectors=None):
        """ Class constructor

        Args:
            instrument_name (str): Instrument name
            detectors (dict): Dictionary representing the configuration for each detector. Keys are detector names
                and values are detector configurations. Each detector should have set at minimum the channel edges and
                the search channels to be used
        """
        super().__init__(instrument_name=instrument_name, detectors=detectors)

    def add_detector(self, detector_name, detector_config):
        """Add a new detector configuration

        Args:
            detector_name (str): Detector name
            detector_config (dict): Dictionary representing the configuration for this detector. Must contain keys for
                channel_edges and search_channels

        Returns:
            None
        """
        if detector_name in self['detectors']:
            warnings.warn(f"Replacing detector {detector_name}")
        self['detectors'][detector_name] = detector_config
        self.validate()

    @property
    def detector_names(self):
        return list(self['detectors'].keys())

    @property
    def channel_mask(self):
        """Construct the mask of allowed detector channels for a search"""
        mask = []
        for det_config in self['detectors'].values():
            mask.append([channel in det_config['search_channels']
                         for channel in range(len(det_config['channel_edges']) - 1)])
        return np.ravel(mask)

    @property
    def search_channels(self):
        return {det: det_config['search_channels'] for det, det_config in self['detectors'].items()}

    @property
    def channel_edges(self):
        return {det: det_config['channel_edges'] for det, det_config in self['detectors'].items()}

    def validate(self):
        """Ensure configuration meets expected structure"""
        if not isinstance(self.config, dict):
            raise ValueError(f"Underlying configuration must be a dictionary")

        for key in ['instrument_name', 'detectors']:
            if key not in self.config:
                raise ValueError(f"Configuration missing '{key}'")

        if not isinstance(self['instrument_name'], str):
            raise ValueError(f"Instrument name is not a string. Please check your inputs.")

        for detector, detector_config in self['detectors'].items():
            for key in ['channel_edges', 'search_channels']:
                if key not in detector_config:
                    raise ValueError(f"Configuration['detectors']['{detector}'] missing '{key}'")

                value = detector_config[key]
                if not isinstance(value, list) or not isinstance(value[0], int):
                    raise ValueError(f"Detector {detector} configuration must contain a key {key} with a value of type list(int)")


class SearchConfiguration(BaseConfiguration):
    """Class for the search configuration

        Attributes:
        -----------
        config: dict
            Dictionary with the search_settings and instruments keys
        instrument_names: list
            Names of available instruments from keys in instrument_configs
        reference_instrument: str
            Key of the instrument to be used as a reference during the search
        time_range: np.array
            Time range associated with the search
            Minimum bin duration

        Public Methods:
        ---------------
        save:
            Save the Results to a yaml file
        add_instrument:
            Add a new instrument and corresponding InstrumentConfiguration
        get_instrument:
            Get the instance of a specified instrument's InstrumentConfiguration
        validate:
            Validate the configuration dictionary

        Class Methods:
        ---------------
        open:
            Open an existing configuration object in a .yaml file
    """
    _derived_keys = ['instrument_names', 'reference_instrument', 'time_range']

    def __init__(self, win_width=60, min_loglr=5.0, min_dur=0.064, max_dur=8.192,
                 min_step=0.064, num_steps=8, skygrid_resolution=5.0,
                 instruments=None, **kwargs):
        """ Class constructor

        Args:
            win_width (float): Width of the window. Default: 60
            min_loglr (float): Minimum log likelihood ratio. Default: 5
            min_dur (float): Minimum duration in seconds. Default: 0.064
            max_dur (float): Maximum duration in seconds. Default: 8.192
            min_step (float): Minimum step size in seconds. Default: 0.064
            num_steps (int): Number of steps. Default: 8
            skygrid_resolution (float): Resolution for skygrid. Default: 5.0
            instruments (list): A list of instrument configurations
            **kwargs (optional): Optional keyword arguments
        """
        super().__init__(win_width=win_width, min_loglr=min_loglr, min_dur=min_dur,
                         max_dur=max_dur, min_step=min_step, num_steps=num_steps,
                         skygrid_resolution=skygrid_resolution, instruments=instruments, **kwargs)

    def add_instrument(self, instrument_config):
        """Adds an instrument configuration to this search configuration

        Args:
            instrument_config (InstrumentConfiguration): An instrument configuration
        """
        if instrument_config['instrument_name'] in self.instrument_names:
            warnings.warn(f"Replacing instrument {instrument_config['instrument_name']}")
            i = self.instrument_names.index(instrument_name)
            self['instruments'][i] = instrument_config
        else:
            self['instruments'].append(instrument_config)
        self.validate()

    def get_instrument(self, instrument_name):
        """Extracts a specified instrument's corresponding InstrumentConfiguration

        Args:
            instrument_name (str): The name of the instrument

        Returns:
            (InstrumentConfiguration): The instance of InstrumentConfugration that corresponds to the input instrument
        """
        try:
            i = self.instrument_names.index(instrument_name)
            return self['instruments'][i]
        except ValueError:
            print(f"{instrument_name} does not exist in instruments list")
            exit(0)

    @property
    def instrument_names(self):
        """(list): list of instrument names"""
        return [instrument['instrument_name'] for instrument in self['instruments']]

    @property
    def reference_instrument(self):
        """(str): Name of the reference instrument (always the first item in the instruments list)"""
        return self['instruments'][0]['instrument_name']

    @property
    def time_range(self):
        """(np.ndarray): search time range (tstart, tstop)"""
        return np.array([-0.5 * self['win_width'], 0.5 * self['win_width']])

    def validate(self):
        """Ensure configuration meets expected structure"""
        # enforce integer types
        for key in ['num_steps']:
            if not isinstance(self.config[key], int):
                raise ValueError(f"{key} must be of type int")

        # enforce number types (int or float)
        for key in ['win_width', 'min_loglr', 'min_dur', 'max_dur', 'min_step', 'skygrid_resolution']:
            if not isinstance(self.config[key], int) and not isinstance(self.config[key], float):
                raise ValueError(f"{key} must be of type int or float")

        # check instrument configs
        if len(self['instruments']) == 0:
            raise ValueError(f"There must be at least one instrument included with search settings")

        for instrument_config in self['instruments']:
            if not isinstance(instrument_config, InstrumentConfiguration):
                raise ValueError(f"Instrument configuration must be of type InstrumentConfiguration")
