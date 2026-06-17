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

from scipy.integrate import trapezoid
from scipy.optimize import fmin


def amplitude_solver(amp, pflux, function, params, energies):
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
    params[0] = 10.0**amp[0]
    photon_model = function.eval(params, energies)
    test_pflux = trapezoid(photon_model, energies)

    return np.abs(test_pflux - pflux)        


def photon_to_energy_flux(pflux, func, params, erange_in=(50.0, 300.0), erange_out=(10.0, 1000.)):
    """Calculate the energy flux from a photon flux.

    Args:
        pflux (np.array): Photon flux measured over `erange_in`
        func (gdt.core.spectra.functions.Function): Functional form of the spectral shape
        params (list): List of parameter values
        erange_in (tuple(2), optional):
            The energy range of the input photon flux, in keV.
            Default is (50.0, 300.0).
        erange_out (tuple(2), optional):
            The energy range over which to calculate the energy flux, in keV.
            Default is (10.0, 1000.0).
            
    Returns:
        (np.array): The energy flux
    """
    # templates are normalized and photon flux calculated over 50-300 keV
    input_energies = np.logspace(*np.log10(erange_in), 1000)
    output_energies = np.logspace(*np.log10(erange_out), 1000)
       
    # need to solve for the photon model amplitude given the model and pflux
    eflux = np.zeros_like(pflux)
    for i in range(pflux.size):
        if pflux[i] <= 0.0:
            continue
        the_args = (pflux[i], func, params, input_energies)
        log_amp = fmin(amplitude_solver, [np.log10(0.01)], the_args, disp=False)
        params[0] = 10.0**log_amp[0]

        # now calculate energy flux over the desired energy range
        eflux[i] = trapezoid(output_energies*func.eval(params, output_energies),
                             output_energies)*1.6e-9
        
    return eflux


def upper_limit_table(values, timescales=[0.128, 1.024, 8.192], sigma=3.0, erange_in=(50.0, 300.0), erange_out=(10.0, 1000.0), report=True):
    """Produce an upper limit report for given timescales and templates

    Args:
        templates (list, optional):
            The template(s). Default is ['soft', 'norm', 'hard']
        timescales (list, optional):
            The timescale(s). Default is [0.128, 1.024, 8.192]
        report (bool, optional):
            Return text report
        **kwargs (optional):
            Keyword arguments to pass to to_energy_flux()
        
    Returns:
        (str): The report
    """
    if sigma <= 0.0:
        raise ValueError('sigma must be positive')

    nspectra = len(values)
    ndurs = len(timescales)
    table = np.zeros((nspectra, ndurs))
    for i in range(nspectra):
        name, func, dur, flux, err = values[i]
        for j in range(ndurs):
            try:
                if timescales[j] not in dur:
                    raise ValueError('{} is not a valid timescale'.format(timescales[j]))

                # masks for duration and spectrum, get the template function definition
                dur_mask = (dur == timescales[j])
 
                # mask the data for the selected timescale and spectrum
                pflux_ul = (flux + sigma * err)[dur_mask]
                eflux_ul = photon_to_energy_flux(pflux_ul, func, func.default_values, erange_in, erange_out)
                table[i,j] = np.max(eflux_ul)
            except ValueError as err: print(err)

    if report == False:
        return table

    title = '\n{:2.1f} sigma Energy Flux Upper Limits '.format(sigma)
    title += ' ({0:2.0f}-{1:2.0f} keV):\n'.format(*erange_out)
    hdr = 'Timescale  '
    hdr += ''.join(['{:<9}'.format(v[0]) for v in values])
    div = '-'*len(hdr)
    lines = [title, hdr, div]
    for i in range(ndurs):
        vals = ['{:2.1e}'.format(table[spec,i]) for spec in range(nspectra)]
        vals = ''.join(['{:<9}'.format(val) for val in vals])
        lines.append('{0} s:   {1}'.format(timescales[i], vals))
        
    return '\n'.join(lines)
