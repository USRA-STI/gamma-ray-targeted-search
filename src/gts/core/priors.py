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

from astropy.coordinates import SkyCoord

def sky_prior(grid, frame, skymap=None, pmin=1e-100):
    """Calculate the sky prior given a map, or do uniform prior, in a spacecraft frame.
    Rotate skymap into the spacecraft frame if it's a HealPix array, otherwise treat it as an
    array in the spacecraft frame.

    Args:
        grid (np.ndarray): Grid of sky locations used in the instrument response
        frame (Frame): Frame object with information about spacecraft position
        skymap (HealPix | np.ndarray): Localization probability to use as the prior. Use uniform prior when None.

    Returns:
        (np.ndarray): The sky prior in the spacecraft frame
    """
    if skymap is None:
        prior = np.ones(len(grid[0]), np.float64)
    elif isinstance(skymap, np.ndarray):
        prior = skymap
    else:
        # get the azimuth and zenith of each unmasked sky grid position
        azimuth, zenith = grid

        # get the equivelent RA and Dec of each unmasked sky grid position
        coords = SkyCoord(azimuth, 0.5 * np.pi - zenith, frame=frame, unit='rad')
        ra = coords.icrs.ra
        dec = coords.icrs.dec

        # calculate the probability of each sky position
        # Note: for now, do explicit lookup with ang2pix to avoid GDT interpolation of values.
        # we need to use exact values to ensure consistency between multiorder vs single resolution map formats.
        ph, th = ra.rad, 0.5 * np.pi - dec.rad
        pix = hp.ang2pix(skymap.nside, th, ph)
        prior = (skymap.prob / skymap.pixel_area)[pix]

    # ensure we're normalized to 1
    prior /= prior.sum()

    return prior

def log_prior(prior, pmin=1e-100):
    """Return log of a prior with protection against zero divergence

    Args:
        prior (np.ndarray): Normalized probability at each sky location
        pmin (float): Minimum allowed probability (avoids zero divergence)

    Returns:
        (np.ndarray): Log of the sky prior
    """
    return np.log(np.maximum(prior, pmin))
