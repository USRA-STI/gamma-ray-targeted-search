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


class BaseConfiguration(yaml.YAMLObject):
    """A base class for configuration objects.

    This class stores keyword settings in a dictionary. The class
    can be written to and retrieved from YAML files.

    Users looking for more complex behavior can inherit
    this class and define a set of derived keys
    that are constructed from the base settings.

    Attributes:
        settings (dict): dictionary with the configuration settings
        yaml_tag (str): tag used to serialize the class within YAML files
        _derived_keys (list): list of derived keys to be constructed
                              from base settings.

    Public Methods:
        keys: List the available keys
        write: Write the configuration to a yaml file
        validate: Validate the settings dictionary

    Class Methods:
        open: Create a BaseConfiguration object given a valid YAML file
    """
    yaml_tag = "!configuration.BaseConfiguration"

    _derived_keys = []

    def __init__(self, **kwargs):
        """Class constructor

        Args:
            kwargs (dict): Keyword dictionary with configuration settings.
        """
        self.settings = kwargs
        self.validate()

    def keys(self):
        """(list): Method to retrieve available configuration keys"""
        return list(self.settings.keys()) + self._derived_keys

    def __getitem__(self, key):
        """Method for high level access to settings dict and derived keys

        Returns:
            (multiple types)
        """
        if key not in self.keys():
            raise KeyError(f"{key} is not a valid key.")
        if key in self._derived_keys:
            return getattr(self, key)
        return self.settings[key]

    def write(self, path):
        """Save the instrument configuration to a file

        Args:
            path (str): Path to the output file
        """
        with open(path, 'w') as file:
            yaml.dump(self, file, default_flow_style=None, sort_keys=False)

    @classmethod
    def open(cls, path):
        """Create a new instance of InstrumentConfiguration given a input file

        Args:
            path (str): Path to configuration file

        Returns:
            configured_instrument (InstrumentConfiguration): Instance of self configured as desired
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such file: '{path}'")
        else:
            with open(path, 'r') as file:
                return yaml.full_load(file)

    def validate(self):
        """Basic validation ensuring we have a settings dictionary"""
        if not isinstance(self.settings, dict):
            raise ValueError(f"Underlying settings must be a dictionary")


class InstrumentConfiguration(BaseConfiguration):
    """Class for the instrument configuration

    Attributes:
        settings (dict): dictionary with the configuration settings
        yaml_tag (str): tag used to serialize the class within YAML files
        _derived_keys (list): list of derived keys to be constructed
                              from base settings.

    Public Methods:
        keys: List the available keys
        write: Write the configuration to a yaml file
        validate: Validate the settings dictionary
        add_detector: Add a new detector and corresponding configuration

    Class Methods:
        open: Create an InstrumentConfiguration object given a valid YAML file
    """
    yaml_tag = "!configuration.InstrumentConfiguration"

    _derived_keys = ['detector_names', 'channel_edges', 'channel_mask', 'search_channels']

    def __init__(self, instrument_name=None, detectors=None):
        """Class constructor

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
            detector_config (dict): Dictionary representing the configuration for this detector.
                Must contain keys for channel_edges and search_channels.
        """
        if detector_name in self['detectors']:
            warnings.warn(f"Replacing detector {detector_name}")
        self['detectors'][detector_name] = detector_config
        self.validate()

    @property
    def detector_names(self):
        """(list): List of detector names"""
        return list(self['detectors'].keys())

    @property
    def channel_mask(self):
        """(numpy.ndarray): Construct the mask of allowed detector channels for a search"""
        mask = []
        for det_config in self['detectors'].values():
            mask.append([channel in det_config['search_channels']
                         for channel in range(len(det_config['channel_edges']) - 1)])
        return np.ravel(mask)

    @property
    def search_channels(self):
        """(dict): Dictionary with search_channels keyed accord to detector names"""
        return {det: det_config['search_channels'] for det, det_config in self['detectors'].items()}

    @property
    def channel_edges(self):
        """(dict): Dictionary with channel_edges keyed accord to detector names"""
        return {det: det_config['channel_edges'] for det, det_config in self['detectors'].items()}

    def validate(self):
        """Ensure configuration meets expected structure"""
        super().validate()

        for key in ['instrument_name', 'detectors']:
            if key not in self.keys():
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
        settings (dict): dictionary with the configuration settings
        yaml_tag (str): tag used to serialize the class within YAML files
        _derived_keys (list): list of derived keys to be constructed
                              from base settings.

    Public Methods:
        keys: List the available keys
        write: Write the configuration to a yaml file
        validate: Validate the settings dictionary
        add_instrument: Add a new instrument and corresponding InstrumentConfiguration
        get_instrument: Get the instance of a specified instrument's InstrumentConfiguration

    Class Methods:
        open: Create a SearchConfiguration object given a valid YAML file
    """
    yaml_tag = "!configuration.SearchConfiguration"

    _derived_keys = ['instrument_names', 'reference_instrument', 'time_range']

    def __init__(self, win_width=60, min_loglr=5.0, min_dur=0.064, max_dur=8.192,
                 min_step=0.064, num_steps=8, skygrid_resolution=5.0,
                 instruments=None, **kwargs):
        """Class constructor

        Args:
            win_width (float): Width of the window. Default: 60
            min_loglr (float): Minimum log likelihood ratio. Default: 5
            min_dur (float): Minimum duration in seconds. Default: 0.064
            max_dur (float): Maximum duration in seconds. Default: 8.192
            min_step (float): Minimum step size in seconds. Default: 0.064
            num_steps (int): Number of steps. Default: 8
            skygrid_resolution (float): Resolution for skygrid. Default: 5.0
            instruments (list): A list of instrument configurations
            **kwargs (optional): Optional keyword arguments for user-defined fields
        """
        super().__init__(win_width=win_width, min_loglr=min_loglr, min_dur=min_dur,
                         max_dur=max_dur, min_step=min_step, num_steps=num_steps,
                         skygrid_resolution=skygrid_resolution, instruments=instruments, **kwargs)

    def add_instrument(self, instrument_config):
        """Adds an instrument configuration to this search configuration

        Args:
            instrument_config (InstrumentConfiguration): An instrument configuration
        """
        name = instrument_config['instrument_name']
        if name in self.instrument_names:
            warnings.warn(f"Replacing instrument {name}")
            i = self.instrument_names.index(name)
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
        """(list): List of instrument names"""
        return [instrument['instrument_name'] for instrument in self['instruments']]

    @property
    def reference_instrument(self):
        """(str): Name of the reference instrument (always the first item in the instruments list)"""
        return self['instruments'][0]['instrument_name']

    @property
    def time_range(self):
        """(numpy.ndarray): Search time range (tstart, tstop)"""
        return np.array([-0.5 * self['win_width'], 0.5 * self['win_width']])

    def validate(self):
        """Ensure configuration meets expected structure"""
        super().validate()

        # enforce integer types
        for key in ['num_steps']:
            if not isinstance(self[key], int):
                raise ValueError(f"{key} must be of type int")

        # enforce number types (int or float)
        for key in ['win_width', 'min_loglr', 'min_dur', 'max_dur', 'min_step', 'skygrid_resolution']:
            if not isinstance(self[key], int) and not isinstance(self[key], float):
                raise ValueError(f"{key} must be of type int or float")

        # check instrument configs
        if not isinstance(self['instruments'], list) or len(self['instruments']) == 0:
            raise ValueError(f"There must be at least one instrument included with search settings")

        for instrument_config in self['instruments']:
            if not isinstance(instrument_config, InstrumentConfiguration):
                raise ValueError(f"Instrument configuration must be of type InstrumentConfiguration")
