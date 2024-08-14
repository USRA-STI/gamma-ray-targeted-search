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

from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation

class SkyGrid():
    """Class to produce an approximate evenly space grid on the sky in
    azimuth and zenith
    """
    def __init__(self, resolution):
        """ Class constructor

        Args:
            resolution (float): The sky grid resolution
        """
        self._resolution = resolution
        self._points = self._calculate(resolution)

    @property
    def size(self):
        """(int): Number of grid points on the sky"""
        return self._points.shape[1]
    
    @property
    def radians(self):
        """(np.ndarray): The azimuth and zenith coordinates in radians"""
        return self._points
    
    @property
    def degrees(self):
        """(np.ndarray): The azimuth and zenith coordinates in degrees"""
        return np.rad2deg(self._points)
    
    def _calculate(self, res):
        """ Method to calculate locations of the response grid on the sky.

        (phi, theta) grid designed to match up with the ones in GBM response 
        tables (in radians) while table values are rounded to the nearest 
        degree, the actual response was calculated at arcmin precision except 
        for the channel-by-channel direct response which is at the rounded 
        resolution

        Args:
            res (float): Angular separation between grid points

        Returns:
            np.ndarray: Array with azimuth and zenith locations of grid points in radians
        """
        theta = np.arange(res, 180, res)
        # angular distance around axis in 2*pi radians
        adist = np.sin(theta * np.pi / 180.)
        # some python roundoff error at 30 degrees
        nphi = np.floor(adist * 360./float(res) + 1e-8).astype(int)
        # initial point at north pole (0, 0) 
        rows = [[0, 0]]
        # go down the rows
        for (t, n) in zip(theta, nphi):  
            # evenly spaced snapped to n deg grid
            phi = np.linspace(0, 360, n, endpoint=False)  
            # phi = np.round(np.linspace(0, 360, n, endpoint=False)) # evenly 
            # spaced snapped to 1deg grid
            rows += [[p, t] for p in phi]
        # final point at south pole (0, 180)
        rows += [[0, 180]]  

        return np.deg2rad(np.array(rows).T)

def getGeoCoordinates(frame, unit='rad'):
    """ Convert the geocenter coordinates from celestial to spacecraft coordinates

    Args:
        frame (Frame): frame object with spacecraft position 
        unit (str): unit to return

    Returns:
        (float, float, float): tuple with geocenter (az, zen) and Earth's angular radius in the specified unit
    """
    geo_coord = SkyCoord(frame.geocenter.ra, frame.geocenter.dec).transform_to(frame)
    geo_azimuth = geo_coord.az
    geo_zenith = 90 * u.deg - geo_coord.el

    return geo_azimuth[0].to_value(unit), geo_zenith[0].to_value(unit), frame.earth_angular_radius.to_value(unit)

def createEarthMask(points, geo_azimuth, geo_zenith, geo_radius):
    """ Creates a mask with visible locations set to True and non-visible
    locations blocked by the Earth set to False

    Args:
        points (np.ndarray): sky grid with azimuth and zenith points
        geo_azimuth (float): azimuth of Earth center in spacecraft frame in radians
        geo_zenith (float): zenith of Earth center in spacecraft frame in radians
        geo_radiius (float): angular radius of the Earth in the spacecraft frame

    Returns:
        np.ndarray: array of booleans where True indicates a visible location
                    from the set of points, False indicates the location is behind the Earth.
    """
    return angular_separation(geo_azimuth, 0.5 * np.pi - geo_zenith,
                              points[0,:], 0.5 * np.pi - points[1,:]) > geo_radius

def grid2healpix(values, coords, spacecraft_frame, nside_out=64,
                 coord_type='instrument', return_proj_coord=False):
    """ Convert grid points to healpix pixel values
            
    Args:
        values (np.array): Original grid values
        coords (tuple(2)): Tuple with (az, zen) or (ra, dec) locations
                           in radians for each grid point
        spacecraft_frame (Frames): Frames object with information about the spacecraft location
        nside_out (int): NSIDE value for the output map
        coord_type (str): Either 'equatorial' if coords are (ra, dec) otherwise assumes (az, zen)
        return_proj_coord (bool): debugging option which returns the projected coordinates for easier plotting

    Returns:
        proj_values (np.ndarray): Array with the projected values
        proj_pix (np.ndarray): Array with the healpix pixel IDs for each projected value
        proj_az (np.ndarray, optional): Array of azimuth values in degrees for each projected pixel
        proj_zen (np.ndarray, optional): Array of zenith values in degrees for each projected pixel
        proj_ra (np.ndarray, optional): Array of right ascension values in degrees for each projected pixel
        proj_dec (np.ndarray, optional): Array of declination values in degrees for each projected pixel
    """
    if coord_type == 'equatorial':
      ra, dec = coords
    else:
      az, zen = coords
      skycoords = SkyCoord(az, 0.5 * np.pi - zen, frame=spacecraft_frame, unit='rad')
      ra = skycoords.icrs.ra.rad
      dec = skycoords.icrs.dec.rad

    proj_pix = np.arange(hp.nside2npix(nside_out))
    proj_th, proj_ph = hp.pix2ang(nside_out, proj_pix)
    proj_ra, proj_dec = proj_ph, 0.5 * np.pi - proj_th

    idx = [angular_separation(proj_ra[i], proj_dec[i], ra, dec).argmin() for i in np.arange(proj_ra.size)]

    if len(values.shape) > 1:
        proj_values = values[:, idx]
    else:
        proj_values = values[idx]

    if return_proj_coord:
        proj_coords = SkyCoord(proj_ra, proj_dec, unit='rad').transform_to(spacecraft_frame)
        proj_az = proj_coords.az
        proj_zen = 0.5 * np.pi - proj_coords.el
        return proj_values, proj_pix, proj_az, proj_zen, proj_ra, proj_dec
    
    return proj_values, proj_pix
