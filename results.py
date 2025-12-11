# Copyright 2017-2022 by Universities Space Research Association (USRA). All rights reserved.
#
# Developed by: William Cleveland, Adam Goldstein, and Suman Bala
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
# Very closely based on the gamma-ray burst targeted search (gbuts).
# Written by:
#               Lindy Blackburn
#               Center for Astrophysics (CfA) | Harvard & Smithsonian
#               https://github.com/lindyblackburn/gbuts
#
# Included in the generalized targeted search (gts) with permission from Lindy.
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
import numpy as np
import healpy as hp
import os
import numpy.lib.recfunctions

from scipy.integrate import trapezoid
from scipy.optimize import fmin
from astropy.coordinates import SkyCoord

from priors import sky_prior, log_prior

def calculate_top_snr(search, result, instrument, channels, n=1):
    """Calculate top `n` signal-to-noise ratios (SNR) for each result.

    Args:
        search (TargetedSearch): The search class with instrument data
        result (np.ndarray): The current search result
        instrument (str): The instrument name to use
        n (int): The number of SNR values to return
        channels (list): The channels to include given as [(det0_min, det0_max), (det1_min, ... ]

    Returns:
        (tuple): Top "n" SNR measurements
    """
    data = search.instrument_data[instrument]

    counts = data.counts[channels].sum(axis=-1)
    background = data.background_counts[channels].sum(axis=-1)
    snr = (counts - background) / np.sqrt(background)

    # Return the top "n" SNR measurements
    return tuple(np.sort(snr)[-n:])

def calculate_pe_variables(search, result, instrument, channels):
    """Calculate variables used for a phosphorescence event veto.
    These typically involve a comparison between signal-to-noise
    ratios in the lowest two energy channels.

    Args:
        search (TargetedSearch): The search class with instrument data
        result (np.ndarray): The current search result
        instrument (str): The instrument name to use
        channels (list): The channels to include given as [(det0_min, det0_max), (det1_min, ... ]

    Returns:
        (tuple): Top 2 SNR (i, j) in lowest energy channel, SNR[j] in next highest channel
    """
    data = search.instrument_data[instrument]

    counts = data.counts[channels]
    background = data.background_counts[channels]
    var = data.background_var[channels]
    snr = (counts - background) / np.sqrt(background + var)

    (i, j) = np.argsort(snr[:,0])[-2:]

    # NOTE: phosphorescence events should
    #
    #  (1) be isolated to a single detector
    #  (2) predominantly appear in the lowest energy channel
    #
    # Therefore, the brightest brightest detector in the lowest energy
    # channel, indexed by j, should be significantly brighter than the
    # next brightest detector, indexed by i. We also return snr from
    # the next heighest energy channel in detector j since it should be
    # much less than snr[j, 0] for real phosphorescence events.
    return (snr[j, 0], snr[i, 0], snr[j, 1])

def calculate_coinclr(search, result, skymap=None):
    """Marginalizes the likelihood ratio using spatial probability provided by in skymap.

    Args:
        search (TargetedSearch): The search class with instrument data
        result (np.ndarray): The current search result
        skymap (HealPix): A HealPix derived skymap class. Default of
                          None marginalizes over a uniform prior with
                          equal weight at every sky location.

    Returns:
        (float): The likelihood ratio marginalized over skymap
    """
    log_p = log_prior(
        sky_prior(search.like_points, search.like_frame, skymap))
    return search.like.coinclr(log_p, llratio=search.like.llr)

def calculate_marginal_flux(search, result, skymap=None, durations=None):
    """Marginalizes the fitted photon flux using spatial probability provided by in skymap.

    Args:
        search (TargetedSearch): The search class with instrument data
        result (np.ndarray): The current search result
        skymap (HealPix): A HealPix derived skymap class. Default of
                          None marginalizes over a uniform prior with
                          equal weight at every sky location.
        durations (list): Restrict the calculation to the provided durations when not None

    Returns:
        (tuple): The marginalized photon flux followed by the fit error on the flux.
                 Size is equal to 2x the number of spectral templates.
    """
    if durations is not None and result['duration'] not in durations:
        return (0,) * 2 * search.like._pflux.shape[0]

    prior = sky_prior(search.like_points, search.like_frame, skymap)
    pflux = search.like._pflux
    pflux_sig = search.like._pflux_sig

    # marginalized flux using the spatial prior
    marginal_pflux = np.sum(prior[np.newaxis,:] * pflux, axis=1) / result['duration']
    marginal_pflux_sig = np.sqrt(np.sum((prior[np.newaxis,:] * pflux_sig)**2, axis=1)) / result['duration']

    return tuple(marginal_pflux) + tuple(marginal_pflux_sig)

class Results:
    required_dtype = [
        ('tstart', 'f8'),
        ('duration', 'f8'),
        ('az', 'f8'),
        ('zen', 'f8'),
        ('template', 'i4'),
        ('flux_amplitude', 'f8'),
        ('reduced_chisq', 'f8'),
        ('chiplusdof', 'f8'),
        ('loglr', 'f8'),
        #('coinclr', 'f8'),
        #('in_gti', 'bool'), # optional
        #('atmoscat', 'bool'), #optional
        #('flags', 'i4'), # make i8 and optional
        #('snr_0', 'f8'), #optional
        #('snr_1', 'f8'), #optional
        #('snr_2', 'f8'), #optional
        #('sun_angle', 'f8'), # Optional
        #('geo_angle', 'f8'), # Optional
        #('pe_0', 'f8'),#optional
        #('pe_1', 'f8'),#optional
        #('pe_2', 'f8'),#optional
    ]

    def __init__(self):
        """Class constructor"""
        self.data = np.empty(0, dtype=self.required_dtype)
        self.time_ref = 0.0
        self.template_names = np.array([])

    @property
    def size(self):
        """(int): total number of results"""
        return self.data.shape[0]

    @property
    def search_window(self):
        """(np.ndarray): the duration in seconds of the search window"""
        return np.max(self['tstart'] + 0.5 * self['duration']) - np.min(self['tstart'] + 0.5 * self['duration'])

    def __getitem__(self, key):
        return self.data[key]

    def save(self, directory, filename=None):
        np.savez(os.path.join(directory, filename), time_ref=self.time_ref,
                 template_names=self.template_names, **{key: self.data[key] for key in self.data.dtype.names}) 

    @classmethod
    def open(cls, filename):
        file = np.load(filename)

        names = [name for name in file.keys() if name not in ['time_ref', 'template_names']]
        n = len(file[names[0]])

        obj = cls.create(n,  time_ref=file['time_ref'], template_names=file["template_names"])

        # fill required fields
        for name, t in obj.required_dtype:
            if name not in names:
                raise KeyError(f"File is missing required key '{name}'")
            obj.data[name] = file[name]
            names.pop(names.index(name))

        # fill any remaining user-defined fields
        if len(names):
            obj.append_fields(names, [file[name] for name in names])

        return obj        

    @classmethod
    def create(cls, size, time_ref=0.0, template_names=None):
        obj = cls()
        obj.data = np.empty(size, dtype=obj.required_dtype)
        obj.time_ref = time_ref
        obj.template_names = np.array([]) if template_names is None else np.array(template_names)
        return obj

    def append_fields(self, names, data):
        self.data = numpy.lib.recfunctions.append_fields(self.data, names, data)

    def append_arrays(self, arrays):
        if not isinstance(arrays, list):
            arrays = [arrays]
        self.data = numpy.lib.recfunctions.merge_arrays([self.data] + arrays, flatten=True)


# TODO: Review FalseAlarmRate to check for API changes
class FalseAlarmRate():
    """Class for False Alarm Rate distributions
    
    Public Methods:
    ---------------
    candidate:
        Calculate the FAR given a candidate value
    distribution:
        Return the cumulative FAR distribution
    write:
        Write the FAR disribution to a npy file

    Class Methods:
    ---------------
    from_npy:
        Create from a FAR distribution saved in a npy file
    from_array:
        Create from an event array and livetime
    """
    def __init__(self):
        """Class constructor"""
        self._events = None
        self._livetime = None
    
    @property
    def livetime(self):
        """(float): The livetime of the distribution in seconds"""
        return self._livetime
        
    @property
    def size(self):
        """(int): The number of events in the distribution"""
        return len(self._events)
    
    @property
    def domain(self):
        """(float, float): The domain (range of event values)"""
        return (self._events[0], self._events[-1])
    
    @property
    def range(self):
        """(float, float): The range of the FAR distribution"""
        return (self.size/self.livetime, 1.0/self.livetime)
    
    def candidate(self, val):
        """Calculate the FAR given a candidate value

        Args:
            val (float): The candidate value
        
        Returns:
            (float): The False Alarm Rate in Hz
        """
        return np.sum(self._events >= val)/self._livetime
    
    def distribution(self, fraction=False):
        """Return the cumulative FAR distribution

        Args:
            fraction (bool, optional):
                If True, return the cumulative fraction, otherwise return the
                cumulative rate. Default is False.
        
        Returns:
            (np.ndarray, np.ndarray):
                Array of event values and cumulative fraction or rate
        """
        y = (np.arange(self.size)+1.0)
        if fraction:
            y /= float(self.size)
        else:
            y /= self.livetime
        y = y[::-1]
        return (self._events, y)
    
    def write(self, filename):
        """Write the FAR disribution to a npy file

        Args:
            filename (str): The filename
        """
        np.save(filename, (self._events, self._livetime))
    
    @classmethod
    def from_npy(cls, npy_file):
        """Create from a FAR distribution saved in a npy file

        Args:
            npy_file (str): The filename of the file to load
        
        Returns:
            (:class:`FalseAlarmRate`): The new object
        """
        events, livetime = np.load(npy_file, allow_pickle=True)
        obj = cls.from_array(events, livetime)
        return obj
    
    @classmethod
    def from_array(cls, array, livetime):
        """Create from an event array and livetime

        Args:
            array (np.ndarray): The event array
            livetime (float): The associated livetime for the event array
        
        Returns:
            (:class:`FalseAlarmRate`): The new object
        """
        obj = cls()
        obj._events = np.sort(array)
        obj._livetime = livetime
        return obj

# TODO: Move to GBMResponse since these are the spectral templates used by GBM
def soft():
    """ Soft Spectral Template describing lower 1/3rd of GBM GRBs

    Returns:
        (func, dict): functional shape and dictionary containing function parameter values
    """
    return (band, {'epeak': 70.0, 'alpha': -1.9, 'beta': -3.70})

def norm():
    """ Normal Spectral Template describing middle 1/3rd of GBM GRBs

    Returns:
        (func, dict): functional shape and dictionary containing function parameter values
    """
    return (band, {'epeak': 230.0, 'alpha': -1.0, 'beta': -2.30})

def hard():
    """ Hard Spectral Template describing upper 1/3rd of GBM GRBs

    Returns:
        (func, dict): functional shape and dictionary containing function parameter values
    """
    return (comp, {'epeak': 1500.0, 'index': -0.5})
            
def band(params, energies):
    """Band GRB function
    This is evaluated in log space and then exponentiated at the end to
    increase robustness.
    
    Args:
        params (dict):
            Dictionary containing band function parameters
        energies (np.ndarray):
            The energies at which to evaluate the function
    
    Returns:
        (np.array): The evaluated function
    """
    e0 = params['epeak']/(2.0+params['alpha'])
    ebreak = (params['alpha']-params['beta'])*e0
    idx = (energies < ebreak)
    logfxn = np.zeros(len(energies), dtype=float)
    logfxn[idx] = np.log(params['amp']) + params['alpha']*np.log(energies[idx]/100.0) \
                  - energies[idx]/e0
    dindex = params['alpha']-params['beta']
    idx = ~idx
    logfxn[idx] = np.log(params['amp']) + dindex*np.log(dindex*e0/100.0) - \
                dindex + params['beta']*np.log(energies[idx]/100.0)
    return np.exp(logfxn)

def comp(params, energies):
    """Comptonized GRB function (Exponentially cut-off power law)
    
    Args:
        params (dict):
            Dictionary containing comptonized function parameters
        energies (np.array):
            The energies at which to evaluate the function
    
    Returns:
        (np.ndarray): The evaluated function
    """
    return params['amp']*(energies/100.0)**params['index'] * \
           np.exp(-energies*(2.0+params['index'])/params['epeak'])


# TODO: Review UpperLimits to check for API changes
class UpperLimits():
    """Class for photon flux/energy flux upper limits
    
    Parameters:
    -----------
    pflux: np.array
        The array of photon flux estimates
    pflux_std: np.array 
        The standard deviation of the photon flux estimates
    times: np.array
        The times of the photon flux estimates
    durations: np.array
        The bin durations corresponding flux estimates
    spectra: np.array
        The corresponding spectral template for each photon flux estimate
    template_names: list, optional
        The names of the templates. Default is ['hard', 'norm', 'soft']
    template_functions: list, optional
        The template functions. Default is [hard, norm, soft]
    ul_map: np.array, optional
        Array with pre-computed upper limit maps in healpix format.
        Dimensions should be (ndur, nspectra, npix) where ndur
        is the number of durations for which upper limit maps are
        computed, nspectra matches the length of template_names,
        and npix represents the number of healpix pixels in the map.
    ul_map_sigma: float, optional
        Significance level of the upper limit maps
    ul_map_durations: list, optional
        List durations for the corresponding ul_map array
                
    Attributes:
    -----------
    templates: list
        The templates available
    timescales: list
        The timescales available

    Public Methods:
    ---------------
    energy_flux_range:
        Calculate the non-zero upper limit range (low, high) for a given
        template and timescale
    report:
        Produce an upper limit report for given timescales and templates
    save:
        Save the upper limits to a npz file
    to_energy_flux:
        Calculate the energy flux for every bin in a given timescale for a 
        given template
    

    Class Methods:
    ---------------
    open:
        Open a saved upper limits npz file
    """
    def __init__(self, pflux, pflux_std, times, durations, spectra,
                 template_names=['hard', 'norm', 'soft'], 
                 template_functions=None,
                 ul_map=None, ul_map_sigma=0., ul_map_durations=None):
        """ Class constructor

        Args:
            pflux (np.array):
                The array of photon flux estimates
            pflux_std (np.array):
                The standard deviation of the photon flux estimates
            times (np.array):
                The times of the photon flux estimates
            durations (np.array):
                The bin durations corresponding flux estimates
            spectra (np.array):
                The corresponding spectral template for each photon flux estimate
            template_names (list, optional):
               The names of the templates. Default is ['hard', 'norm', 'soft']
            template_functions (list, optional):
                The template functions. Default is [hard, norm, soft]
            ul_map (np.array, optional):
                Array with pre-computed upper limit maps in healpix format.
                Dimensions should be (ndur, nspectra, npix) where ndur
                is the number of durations for which upper limit maps are
                computed, nspectra matches the length of template_names,
                and npix represents the number of healpix pixels in the map.
            ul_map_sigma (float, optional):
                Significance level of the upper limit maps
            ul_map_durations (list, optional):
                List durations for the corresponding ul_map array
        """
        known_functions = {'hard': hard, 'norm': norm, 'soft': soft}
        if template_functions is None:
            template_functions = []
            # lookup template functions from known functions
            for name in template_names:
                if name in list(known_functions.keys()):
                    template_functions.append(known_functions[name])
                else:
                    raise ValueError("unknown function '%s'" % name)

        self._pflux = pflux
        self._pflux_std = pflux_std
        self._times = times
        self._durations = durations
        self._spectra = spectra
        self._templates = np.asarray(template_names)
        self._functions = np.asarray(template_functions)
        self._ul_map = ul_map
        self._ul_map_sigma = ul_map_sigma  
        self._ul_map_durations = np.asarray(ul_map_durations)
 
    @property
    def templates(self):
        """(list): The names of the templates."""
        return self._templates.tolist()
    
    @property
    def timescales(self):
        """(np.ndarry): The emission timescales of the flux upper limit estimates"""
        return np.unique(self._durations)

    @property
    def ul_map_durations(self):
        """(list): durations for the corresponding ul_map array"""
        return self._ul_map_durations.tolist()
    
    def save(self, directory, filename=None):
        """Save the upper limits to a npz file

        Args:
            directory (str):
                The directory to write to
            filename (str):
                The filename
        """
        filename = os.path.join(directory, filename)
        np.savez(filename, times=self._times, pflux=self._pflux, 
                 pflux_std=self._pflux_std, durations=self._durations,
                 spectra=self._spectra, templates=self._templates,
                 functions=self._functions, ul_map=self._ul_map,
                 ul_map_sigma=self._ul_map_sigma,
                 ul_map_durations=self._ul_map_durations)
    
    @classmethod
    def open(cls, filename, **kwargs):
        """Open a saved upper limits npz file and return an UpperLimits object

        Args:
            filename (str):
                The filename to open
            **kwargs (optional):
                Keywords to pass to the initializer

        Returns:
            (:class:`UpperLimits`): The loaded object
        """
        file = np.load(filename, allow_pickle=True)
        obj = cls(file['pflux'], file['pflux_std'], file['times'], 
                  file['durations'], file['spectra'], file['templates'],
                  file['functions'], file['ul_map'], file['ul_map_sigma'],
                  file['ul_map_durations'], **kwargs)
        return obj
    
    def report(self, templates=['soft', 'norm', 'hard'], 
               timescales=[0.128, 1.024, 8.192], **kwargs):
        """Produce an upper limit report for given timescales and templates

        Args:
            templates (list, optional):
                The template(s). Default is ['soft', 'norm', 'hard']
            timescales (list, optional):
                The timescale(s). Default is [0.128, 1.024, 8.192]
            **kwargs (optional):
                Keyword arguments to pass to to_energy_flux()
        
        Returns:
            (str): The report
        """
        nspectra = len(templates)
        ndurs = len(timescales)
        table = np.zeros((nspectra, ndurs))
        for i in range(nspectra):
            for j in range(ndurs):
                try:
                    _, eflux = self.energy_flux_range(templates[i], timescales[j], 
                                                      **kwargs)
                    table[i,j] = eflux
                except ValueError as err: print(err)
        
        try:
            sigma = kwargs['sigma']
        except:
            sigma = 3.0
        try:
            erange = kwargs['energy_range']
        except:
            erange = (10.0, 1000.0)
        title = '\n{:2.1f} sigma Energy Flux Upper Limits '.format(sigma)
        title+= ' ({0:2.0f}-{1:2.0f} keV):\n'.format(*erange)
        hdr = 'Timescale  '
        hdr += ''.join(['{:<9}'.format(x) for x in templates])
        div = '-'*len(hdr)
        lines = [title, hdr, div]
        for i in range(ndurs):
            vals = ['{:2.1e}'.format(table[spec,i]) for spec in range(nspectra)]
            vals = ''.join(['{:<9}'.format(val) for val in vals])
            lines.append('{0} s:   {1}'.format(timescales[i], vals))
        
        return '\n'.join(lines)
        
    def photon_flux(self, template, timescale, sigma=3.0):
        """Return the photon flux UL in 50-300 keV for a given template and 
        timescale

        Args:
            template (str): The template
            timescale (float): The timescale
            sigma (float, optional): The Gaussian-equivalent sigma
            
        Returns:
            (np.array, np.array): Arrays for the times of each bin and photon flux upper limits
        """
        if template not in self.templates:
            raise ValueError('{} is not a valid template'.format(template))
        if timescale not in self.timescales:
            raise ValueError('{} is not a valid timescale'.format(timescale))
        if sigma <= 0.0:
            raise ValueError('sigma must be positive')

        # masks for duration and spectrum, get the template function definition
        dur_mask = (self._durations == timescale)
        spec_mask = (self._templates == template)

        pflux_ul = self._pflux + sigma*self._pflux_std
        
        # mask the data for the selected timescale and spectrum
        times = self._times[dur_mask]
        pflux_ul = pflux_ul[dur_mask,:]
        pflux_ul = pflux_ul[:,spec_mask]
        
        return (times, pflux_ul)
    
    def energy_flux_range(self, template, timescale, sigma=3.0, **kwargs):
        """Calculate the non-zero upper limit range (low, high) for a given
        template and timescale

        Args:
            template (str): The template
            timescale (float): The timescale
            sigma (float, optional): The Gaussian-equivalent sigma
            **kwargs (optional): Keyword arguments to pass to to_energy_flux()
            
        Returns:
            (float, float): The minimum, non-zero energy flux and maximum energy flux
        """
        if template not in self.templates:
            raise ValueError('{} is not a valid template'.format(template))
        if timescale not in self.timescales:
            raise ValueError('{} is not a valid timescale'.format(timescale))
        if sigma <= 0.0:
            raise ValueError('sigma must be positive')

        # masks for duration and spectrum, get the template function definition
        dur_mask = (self._durations == timescale)
        spec_mask = (self._templates == template)
 
        # mask the data for the selected timescale and spectrum
        times = self._times[dur_mask]
        pflux_ul = self._pflux + sigma*self._pflux_std
        pflux_ul = pflux_ul[dur_mask,:]
        pflux_ul = pflux_ul[:,spec_mask]

        eflux = self.to_energy_flux(pflux_ul, template, **kwargs)
        min_eflux = np.min(eflux[eflux > 0.0])
        max_eflux = np.max(eflux)
        return (min_eflux, max_eflux)
    
    def to_energy_flux(self, pflux, template, energy_range=(10.0, 1000.)):
        """Calculate the energy flux from a photon flux

        Args:
            pflux (np.array): Photon flux measured over 50-300 keV
            template (str): The template
            energy_range (tuple(2), optional):
                The energy range over which to calculate the energy flux, in keV.
                Default is (10.0, 1000.0).
            
        Returns:
            (np.array): The energy flux
        """
        if template not in self.templates:
            raise ValueError('{} is not a valid template'.format(template))
        
        # get the template function definition
        spec_mask = (self._templates == template)
        func, params = self._functions[spec_mask][0]()

        # templates are normalized and photon flux calculated over 50-300 keV
        input_energies = np.logspace(np.log10(50.0), np.log10(300.0), 1000)
        output_energies = np.logspace(np.log10(energy_range[0]), 
                                      np.log10(energy_range[1]), 1000)
       
        # need to solve for the photon model amplitude given the model and pflux
        eflux = np.zeros_like(pflux)
        for i in range(pflux.size):
            if pflux[i] <= 0.0:
                continue
            the_args = (pflux[i], func, params, input_energies)
            log_amp = fmin(self._amplitude_solver, [np.log10(0.01)], the_args, disp=False)
            params['amp'] = 10.0**log_amp[0]

            # now calculate energy flux over the desired energy range
            eflux[i] = trapezoid(output_energies*func(params, output_energies),
                                output_energies)*1.6e-9
        
        return eflux
    
    def _amplitude_solver(self, amp, pflux, function, params, energies):
        """ Method to retrieve the photon flux amplitude from a spectral shape integrated over energy

        Note: amplitude is a log-distributed scale parameter, so we should evaluate
        it in log space to increase solution stability

        Args:
            amp (float): input amplitude to ttest
            pflux (float): photon flux intergrated over an energy range. units are photons/cm2/s.
            function (func): functional shape of the spectrum
            params (dict): dictionary with parameter values for the spectrum
            energies (np.ndarray): energies over which the flux integral is computed

        Returns:
            (float): difference between desired photon flux and photon flux computed with test amplitude
        """
        params['amp'] = 10.0**amp[0]
        photon_model = function(params, energies)
        test_pflux = trapezoid(photon_model, energies)
        return np.abs(test_pflux - pflux)        
        
    def remove_earth(self, input_map, duration, poshist, output_nside=512):
        """ Method to remove the earth from an upper limit map

        Note: this should be moved to a map handling class and updated to use spacecraft frames

        Args:
            input_map (np.ndarray): healpix map values
            duration (float): duration used to compute the upper limit map
            poshist (PosHist): deprecated position history class from the old GBM data tools
            output_nside (int): nside value of the returned map

        Returns:
            (np.ndarray, np.ndarray, np.ndarray):
                arrays with the healpix map values after removing the Earth,
                a healpix map with the earth region set to 1.0 and all other values zero,
                values of the earth geocenter positions in right ascension/declination and its angular radius
        """
        input_nside = hp.npix2nside(input_map.size)
        output_npix = hp.nside2npix(output_nside)
        earth_map = np.zeros(output_npix, np.float64)

        times = self._times[self._durations == duration]
        geocenters = np.zeros((times.size, 4), np.float64)
        for i, t in enumerate(times):
            rad = poshist.get_earth_radius(t)
            ra, dec = poshist.get_geocenter_radec(t)
            vec = hp.ang2vec(np.radians(90. - dec), np.radians(ra))
            pix = hp.query_disc(output_nside, vec, np.radians(rad))
            earth_map[pix] = 1.0
            geocenters[i] = (t, ra, dec, rad)

        output_map = hp.ud_grade(input_map, output_nside)
        output_map[earth_map > 0.] = hp.UNSEEN

        return output_map, earth_map, geocenters

    def get_ul_map(self, spectrum, duration, poshist=None,
                   energy_range=[10., 1000.],
                   energy_flux=True, earthmask=False):
        """ Function for returning upper limit maps as an array of
            healpix pixel values.

        Note: this method needs to be updated to use spacecraft frames

        Args:
            spectrum (str):
                Spectral template of the upper limit map
            duration (float64):
                Duration of the upper limit map in seconds
            poshist (PosHist, optional):
                Position history class needed for earthmask option
            energy_range (list, optional):
                Energy range in keV used for reporting energy flux
            energy_flux (bool, optional):
                Return energy flux in erg/s/cm2 when True
            earthmask (bool, optional):
                Return map with sum of earth occultations when True.
                Requires poshist argument.

        Returns:
            (np.ndarray):
                Array with upper limit values for each pixel of a healpix skymap
        OR
            (np.ndarray, np.ndarray, np.ndarray):
                Arrays with upper limit values for each pixel of a healpix skymap,
                marking earth occulted positions. 1 = occulted, 0 = visible, and
                list of geocenters formatted as (met, ra, dec, radius).
        """
        if duration not in self._ul_map_durations:
            raise ValueError("Upper limit map not available for %.3f duration" % duration)
        if spectrum not in self._templates:
            raise ValueError("Upper limit map not available for %s spectrum" % spectrum)

        idur = np.where(self._ul_map_durations == duration)[0][0]
        ispec = np.where(self._templates == spectrum)[0][0]
        ul_map = self._ul_map[idur][ispec].copy()

        if energy_flux:
            unit_flux = np.array([1.0])
            scale = self.to_energy_flux(unit_flux, spectrum, energy_range)[0]
            ul_map *= scale

        if earthmask:
            if poshist is None:
                raise ValueError("Must provide poshist object to calculate earthmask")
            return self.remove_earth(ul_map, duration, poshist)
        return ul_map
