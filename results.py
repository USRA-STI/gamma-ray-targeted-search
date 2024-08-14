# Copyright 2017-2022 by Universities Space Research Association (USRA). All rights reserved.
#
# Developed by: William Cleveland and Adam Goldstein
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
import sys
import os
from scipy.integrate import trapz
from scipy.optimize import fmin

class Results():
    """Class for the npy results files
    
    Attributes:
    -----------
    amplitudes: np.array
        Array of MLE count amplitudes for each bin
    atmoscat: np.array
        Boolean array indicating if the atmospheric scattering was used
    chisq: np.array
        The chisq and chisq+ for each bin
    coinclr: np.array
        The spatially-coincident log-likelihood ratio for each bin
    durations: np.array
        The bin durations
    flags: np.array
        Array of "good" flags: if background fit was good, or pre-threshold used
    geo_angle: np.array
        The angle between the localization centroid and geocenter for each bin
    in_gti: np.array
        Boolean array indicating if each bin is in a GTI
    locs: (np.array, np.array)
        The RA and Dec of the localization centroid for each bin
    locs_sc: (np.array, np.array)
        The spacecraft az/zen of the localization centroid for each bin
    loglr: np.array
        The log-likelihood ratio for each bin
    pe_values: np.array
        The phosphorescent event values for each bin
    size: int
        The number of bins
    snr: np.array
        The S/N ratio of each bin
    sun_angle: np.array
        The angle between the localization centroid and sun for each bin
    t0: float
        The reference time
    templates: np.array
        The spectral template for each bin
    times: np.array
        The array of bin times
    times_relative: np.array
        The array of bin times relative to the search time
    timescales: list
        The timescales contained in the search
    window_width: float
        The duration in seconds of the search window
    
    Public Methods:
    ---------------
    downselect:
        Filter and combine events to keep only the most significant of 
        overlapping bins. Returns new Results
    remove_dur_spec:
        Remove a combination of duration/spectrum
    remove_pe:
        Remove likely phosphorescent events and return new Results
    save:
        Save the Results to a npy file
    sky_cut:
        Remove bins that have less than a threshold difference between coinclr
        and loglr. Returns new Results
    sort: 
        In-place sort on an attribute
    write:
        Pretty-print write of the results to a file or stdout
    
    Class Methods:
    ---------------
    create:
        Create a Results object given a valid data array
    open:
        Open an existing data array in a valid npy file
    """
    def __init__(self):
        """Class constructor"""
        self._data = None
        self._timeref = 0.0
        self._template_names = None

    @property
    def t0(self):
        """(float): The reference time for the results"""
        return self._timeref
    
    @property
    def size(self):
        """(int): total number of results"""
        return self._data.shape[0]
    
    @property
    def timescales(self):
        """(np.ndarray): The photon emission timescales contained in the search"""
        return np.unique(self.durations)
    
    @property
    def window_width(self):
        """(np.ndarray): The duration in seconds of the search window"""
        return np.max(self.times)-np.min(self.times)
        
    @property
    def times(self):
        """(np.ndarray): central times of the search bins"""
        return self._data[:,0]
    
    @property
    def times_relative(self):
        """(np.ndarray): The array of bin times relative to the search time"""
        return self.times-self._timeref
    
    @property
    def tstart(self):
        """(np.ndarray): start times of the search bins"""
        return self.times-self.durations/2.0
    
    @property
    def tstop(self):
        """(np.ndarray): stop times of the search bins"""
        return self.times+self.durations/2.0
        
    @property
    def durations(self):
        """(np.ndarray): durations of the search bins"""
        return self._data[:,1]
    
    @property
    def in_gti(self):
        """(np.ndarray): True when search bin is within a good time interval (GTI) of the underlying data"""
        return self._data[:,2].astype(bool)
    
    @property
    def atmoscat(self):
        """(np.ndarray): True when atmospheric scattering effects are included in the response matrix"""
        return self._data[:,3].astype(bool)
    
    @property
    def flags(self):
        """(np.ndarray): additional instrument-specific flags"""
        return self._data[:,4].astype(int)
        
    @property
    def locs_sc(self):
        """(np.ndarray): spacecraft frame azimuth and zenith in degrees of the best-fit position for a search bin"""
        az = self._data[:,5]
        az[(az < 0.0)] += 360.0
        zen = self._data[:,6]
        return (az, zen)
    
    @property
    def locs(self):
        """(np.ndarray): right ascension and declination in degrees in the spacecraft frame of the best-fit position for a search bin"""
        ra = self._data[:,7]
        ra[(ra < 0.0)] += 360.0
        dec = self._data[:,8]
        return (ra, dec)
    
    @property
    def templates(self):
        """(np.ndarray): best-fit spectral templates for each search bin"""
        if self._template_names is not None: 
            return self._template_names[self._data[:,9].astype(int)]
        else:
            return self._data[:,9].astype(int)
    
    @property
    def amplitudes(self):
        """(np.ndarray): best-fit photon flux marginalized over the sky for each search bin"""
        return self._data[:,10]
    
    @property
    def snr(self):
        """(np.ndarray): signal-to-noise ratios for:
                            1. the best-fit position and spectral template
                            2. the highest single detector snr summed over a user-specified energy range
                            3. the second highest single detector snr summed over a user-specified energy range
        """
        return self._data[:,11:14]
    
    @property
    def chisq(self):
        """(np.ndarray): chi square computed relative to the response for the best-fit position and spectral template"""
        return self._data[:,14:16]
    
    @property
    def sun_angle(self):
        """(np.ndarray): angle between the best-fit location and sun position in degrees"""
        return np.rad2deg(self._data[:,16])

    @property
    def geo_angle(self):
        """(np.ndarray): angle between the best-fit location and Earth center in degrees"""
        return np.rad2deg(self._data[:,17])
    
    @property
    def loglr(self):
        """(np.ndarray): the log-likelihood ratio for each search bin.
        This is marginalized over the full sky and all templates using a uniform prior."""
        return self._data[:,18]
    
    @property
    def coinclr(self):
        """(np.ndarray): the 'coincident' log-likelihood ratio for each search bin.
        This is marginalized over the sky using an external localization prior in addition to a uniform prior over spectral templates."""
        return self._data[:,19]
        
    @property
    def pe_values(self):
        """(np.ndarray): variables used in the instrument-specific phosphorescence event (pe) veto."""
        return self._data[:,20:]
    
    def remove_pe(self, cr1=5, cr2=1, cr2thr=8):
        """Remove likely phosphorescent events (PEs) and return new Results

        Args:
            cr1 (float, optional):
                Threshold value for comparing detectors with highest and
                second highest signal-to-noise ratios in lowest energy channel
            cr2 (float, optional):
                Threshold value for comparing signal-to-noise ratios for the
                lowest two energy channels in detector with the highest signal-to-noise ratio from cr1
            cr2thr (float, optional):
                Absolute threshold on signal-to-noise ratio of second lowest energy channel
                in detector with the highest signal-to-noise ratio from cr1
        
        Returns:
            (Results): A new Results object with the PEs removed
        """
        if self.size == 0:
            return self
        icr1 = self.pe_values[:,0] / np.maximum(0.1, self.pe_values[:,1]) < cr1
        icr2 = (self.pe_values[:,0] / np.maximum(0.1, self.pe_values[:,2]) < cr2) | \
               (self.pe_values[:,0] < cr2thr)
        
        obj = Results.create(self._data[(icr1 & icr2),:], time_ref=self._timeref,
                             templates=self._template_names)
        return obj
    
    def remove_dur_spec(self, dur, spec):
        """Remove a combination of duration/spectrum.  
        This is done in place without creating a new object

        Args:
            dur (float): A timescale to remove
            spec (str): A template to remove
        """
        if self.size > 0:
            mask = (self.durations == dur) & (self.templates == spec)
            self._data = self._data[~mask,:]
      
    def sky_cut(self, sky_diff=2):
        """Remove bins that have less than a threshold difference between 
        coinclr and loglr. Returns a new Results object.

        Args:
        sky_diff (float, optional):
            The threshold such that bins with (coinclr-loglr) < sky_diff
            are removed.  Default is 2.
        
        Returns:
            (Results): A new Results object with the resulting bins removed
        """
        if self.size == 0:
            return self
        isky = (self.coinclr-self.loglr) > sky_diff
        obj = Results.create(self._data[isky,:], time_ref=self._timeref,
                             templates=self._template_names)
        return obj
        
    def downselect(self, overlap_factor=0.2, threshold=None, combine_spec=True, 
                   fixedwin=0, no_empty=False):
        """Filter and combine events to keep only the most significant of 
        overlapping bins. Returns a new Results object.

        Args:
            overlap_factor (float, optional):
                Only remove a bin if a brighter bin has a larger S/N ratio by this
                factor. Default is 0.2.
            threshold (float, optional):
                Filter out bins with loglr below this threshold.  If not set, no
                filtering is performed.
            combine_spec (bool, optional):
                If True, combine spectral templates
            fixedwin (float, optional):
                Fixed coincidence window. Default is 0.
                NOTE: The behavior of this argument is not fully understood. Might create problems.
            no_empty (bool, optional):
                If True, forces the single bin with the most significant loglr to be 
                retained, even if it is below the defined threshold. Default is False.
        
        Returns:
            (Results): A new Results object with the filtered and downselected bins
        """
        if self.size == 0:
            return self
        
        if threshold:
            mask = (self.loglr >= threshold)
            if (mask.sum() == 0) and no_empty:
                mask = (self.loglr == self.loglr.max())
            data = self._data[mask,:]
        else:
            data = self._data        
        
        unique_events = []
        sorted_events = data[(-data[:,18]).argsort(), :]
        
        for e1 in sorted_events:
            keep = True
            for e2 in unique_events:
                toverlap = min(e1[0]+e1[1]/2.0, e2[0]+e2[1]/2.0) \
                           - max(e1[0]-e1[1]/2.0, e2[0]-e2[1]/2.0) + fixedwin
                
                if (combine_spec or (e2[9] == e1[9])) and (toverlap > 0):
                    amplitude = e1[11]/np.sqrt(e1[1])
                    snr_expected = amplitude * toverlap / np.sqrt(e2[1])
                    if e2[11] * overlap_factor < snr_expected:
                        keep = False
                        break
            if (keep):
                unique_events.append(e1)
        
        data = np.array(unique_events)
        obj = Results.create(data, time_ref=self._timeref, templates=self._template_names)
        return obj
    
    def sort(self, loglr=False, coinclr=False, time=False, duration=False, 
             template=False, snr=False, sun_angle=False, geo_angle=False, 
             amplitude=False, reverse=False):
        """ In-place sort on an attribute

        NOTE: We should either use getattr() or a structured numpy array to simplify this
        function. That would let us use a string as the argument for the sorting field.

        Args:
            loglr, coinclr, time, duration, template, snr, sun_angle, geo_angle,
            amplitude (bool):
                Set one of these to True to sort on that attribute.
            reverse (bool, optional):
                If True, then reverse sort. Default is False
        """
        if loglr:
            idx = np.argsort(self.loglr)
        elif coinclr:
            idx = np.argsort(self.coinclr)
        elif time:
            idx = np.argsort(self.times)
        elif duration:
            idx = np.argsort(self.durations)
        elif template:
            idx = np.argsort(self.templates)
        elif snr:
            idx = np.argsort(self.snr[:,0])
        elif amplitude:
            idx = np.argsort(self.amplitudes)            
        elif sun_angle:
            idx = np.argsort(self.sun_angle)
        elif geo_angle:
            idx = np.argsort(self.geo_angle)
        else:
            raise ValueError("Must set a valid value to sort over")
        
        if reverse:
            idx = idx[::-1]
        
        self._data = self._data[idx,:]
    
    def write(self, output=None):
        """Pretty-print write of the results to a file or stdout

        Args:
            output (file handle, optional):
                The file handle to write to.  If not set, will write to stdout.
        """
        if output is None:
            output = sys.stdout
        
        output.write('Total number of bins: {}\n'.format(self.size))
        output.write('In GTI: {}\n'.format(np.sum(self.in_gti)))
        output.write('Used atmoscat: {}\n'.format(np.sum(self.atmoscat)))
        output.write('Pre-filtered: {}\n'.format(np.sum(self.flags == 2)))
        output.write(
            "--------------------------------------------------------------------------------------------------------------------------------------------------\n")
        output.write(
            "    tcent    duration  gti rock good  phi  theta  ra  dec  spec ampli  snr  snr0  snr1 chisq chisq+ sun  earth    logLR   coincLR  PE0   PE1   PE2\n")
        output.write(
            "--------------------------------------------------------------------------------------------------------------------------------------------------\n")
        data = np.copy(self._data)
        data[:,5], data[:,6] = self.locs_sc
        data[:,7], data[:,8] = self.locs
        for row in data:
            row = list(row)
            row[0] -= self._timeref
            output.write(
                "%13.3f %7.3f %3d %4d %4d  %5.1f %5.1f %5.1f %5.1f %1d %5.2f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %8.2f %8.2f %5.1f %5.1f %5.1f\n" % tuple(
                    row))
    
    def save(self, directory, filename=None):
        """Save the Results to a npy file

        Args:
            directory (str):
                The directory to write to
            filename (str, optional):
                The filename to write to        
        """
        np.savez(os.path.join(directory, filename),
                 data=self._data, template_names=self._template_names)
    
    @classmethod
    def open(cls, filename, time_ref=0.0):
        """Open an existing data array in a valid npy file

        Args:
            filename (str):
                The full filename of the file to be opened
            time_ref (float, optional):
                The reference time to apply to the data. Default is 0.
        
        Returns:
            (Results): The Results object containing the data
        """
        file = np.load(filename, allow_pickle=True)
        return cls.create(file["data"], time_ref=time_ref, templates=file["template_names"])
    
    @classmethod
    def create(cls, data, time_ref=0.0, templates=['hard', 'norm', 'soft']):
        """Create a Results object given a valid data array.

        Args:
            data (np.array):
                A valid array of shape (n, 23) for n bins
            time_ref (float, optional):
                The reference time to apply to the data. Default is 0.
            templates (list):
                A list of template names. Default is ['hard', 'norm', 'soft']
        
        Returns:
            (Results): The Results object containing the data
        """
        obj = cls()
        obj._data = data
        if obj.size == 0:
            obj._data = obj._data.reshape(0, 23)
        obj._timeref = time_ref
        obj._template_names = np.asarray(templates)
        return obj


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
            eflux[i] = trapz(output_energies*func(params, output_energies), 
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
        test_pflux = trapz(photon_model, energies)
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
