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
