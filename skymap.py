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
from scipy.stats import norm
import healpy as hp
import astropy
import astropy.io.fits as fits
import astropy_healpix as ah
from astropy import units as au
from astropy.table import Table
from gdt.core.healpix import HealPixLocalization
from collections import OrderedDict

def O3_DGAUSS_Model(atm, zen):
    """Systematic model for localizations using a double Gaussian function.

    Args:
        atm (int):
            the type of atmospheric scattering used in the localization.
            0 == None, 1 ==  average rocking angle.
        zen (float):
            zenith angle of the best position relative to Fermi.

    Returns:
        ([float, float], [float]):
            sigma1 and sigma2 Guassian widths for systematic smoothings
            as well as the fractional contribution for sigma1. The
            fraction contribution for sigma2 is 1 - frac1.
    """
    if atm == 0:
        if zen < 40.:
          sigma1 = np.deg2rad(5.99)
          sigma2 = np.deg2rad(60.41)
          frac1 = 0.59
        elif 40. <= zen and zen < 65.:
          sigma1 = np.deg2rad(3.61)
          sigma2 = np.deg2rad(35.68)
          frac1 = 0.77
        elif 65. <= zen and zen < 85.:
          sigma1 = np.deg2rad(3.84)
          sigma2 = np.deg2rad(30.08)
          frac1 = 0.73
        elif 85. <= zen and zen < 110.:
          sigma1 = np.deg2rad(2.34)
          sigma2 = np.deg2rad(22.22)
          frac1 = 0.74
        else:
          sigma1 = np.deg2rad(1.84)
          sigma2 = np.deg2rad(38.47)
          frac1 = 0.42
    elif atm == 1:
        if zen < 40.:
          sigma1 = np.deg2rad(1.58)
          sigma2 = np.deg2rad(6.41)
          frac1 = 0.67
        elif 40. <= zen and zen < 60.:
          sigma1 = np.deg2rad(1.84)
          sigma2 = np.deg2rad(6.81)
          frac1 = 0.69
        elif 60. <= zen and zen < 80.:
          sigma1 = np.deg2rad(2.42)
          sigma2 = np.deg2rad(10.91)
          frac1 = 0.77
        elif 80. <= zen and zen < 110.:
          sigma1 = np.deg2rad(1.87)
          sigma2 = np.deg2rad(7.59)
          frac1 = 0.66
        else:
          sigma1 = np.deg2rad(2.53)
          sigma2 = np.deg2rad(16.27)
          frac1 = 0.72
    else:
        raise ValueError("atm must be either 0 or 1")
    return ([sigma1, sigma2], [frac1])

def find_greedy_credible_levels(p, area=None):
    """Calculate the credible values of a probability array using a greedy
    algorithm. Allow ranking by pixel density to handle multi-order LIGO maps.

    Args:
        p (np.array): probability array
        area (np.ndarray): optional area array to rank pixels by density instead of p-value

    Returns:
        (np.ndarray): The credible values
    """
    p = np.asarray(p)
    pflat = p.ravel()

    if area is None:
        aflat = 1.
    else:
        area = np.asarray(area)
        aflat = area.ravel()

    i = np.argsort(pflat / aflat)[::-1]
    cs = np.cumsum(pflat[i])
    cls = np.empty_like(pflat)
    cls[i] = cs
    return cls.reshape(p.shape)

class LigoHealPix(HealPixLocalization):
    """Class for the LIGO/Virgo HEALPix localization files. Inherits from
    HealPix
    """
    def __init__(self):
        """ Class constructor """
        super().__init__()
        self._uniq = None
        self._dist_mu = None
        self._dist_sig = None
        self._dist_norm = None
        self._filename = None
        self._full_path = None
        self._headers = OrderedDict()
        self._npix = 0

    @property
    def trigtime(self):
        """(str): The ISO formatted time corresponding to the localization"""
        return self.headers[self._data_ext()]['DATE-OBS']
    
    @property
    def dist_mean(self):
        """(float): The mean distance (in Mpc)"""
        return self.headers[self._data_ext()]['DISTMEAN']

    @property
    def dist_std(self):
        """(float): The standard deviation of the distance (in Mpc)"""
        return self.headers[self._data_ext()]['DISTSTD']
    
    @property
    def detector(self):
        """(str): string of instrument names contributing to the localization"""
        try:
            dets = self.headers[self._data_ext()]['INSTRUME']
            return tuple(dets.split(','))
        # currently LALInference does not contain the 'INSTRUME' keyword
        except:
            return ('',)
        
    @property
    def creator(self):
        """(str): The creator method of the HealPix map"""
        return self.headers[self._data_ext()]['CREATOR']
    
    @property
    def nside(self):
        """(int or array-like): nside values for single and multiresolution formats"""
        if self._uniq is None:
            return hp.npix2nside(self.npix)
        level, ipix = ah.uniq_to_level_ipix(self._uniq)
        return ah.level_to_nside(level)
    
    @property
    def pixel_area(self):
        """(float or array-like): pixel areas for single and multiresolution formats"""
        if self._uniq is None:
            return 4.0*180.0**2/(np.pi*self.npix)
        return self._uniq_to_pixel_area(self._uniq, degrees=True)

    @property
    def filename(self):
        """(str): localization file name"""
        return self._filename

    @property
    def headers(self):
        """(headers): FITS headers"""
        return self._headers
            
    @property
    def npix(self):
        """(int): Number of pixels in the HEALPix map"""
        return hp.nside2npix(self._nside)
        # TODO: add multiorder handling
        # return len(self._npix)
                
    #mark TODO: What to we want to return?
    def distance(self, clevel):
        """ Collect distance information over a specific confidence level

        Args:
            clevel (float): confidence level between 0 and 1

        Returns:
            (np.ndarray, np.ndarray): arrays with distance as well as marginalized probability for distance squared?
        """
        mask = (1.0-self._sig <= clevel)
        dist_min = 0.9*np.min(self._dist_mu[mask]-self._dist_sig[mask])
        dist_max = 3.0*np.max(self._dist_mu[mask]+self._dist_sig[mask])
        
        x = np.linspace(dist_min, dist_max, 100)
        y = [np.sum(self.prob[mask] * xx**2 * self._dist_norm[mask] * \
                    norm(self._dist_mu[mask], self._dist_sig[mask]).pdf(xx)) \
                    for xx in x]
        return (x, np.array(y))
    
    def proj_to_multiorder(self, uniq, input_map, nest=False, density=False):
        """ Projects an input map onto a set of multiordered pixels

        Args:
            uniq (np.ndarray): array of unique pixel IDs for the output map
            input_map (np.ndarray): the input map to project
            nest (bool): True when the input map uses NESTED healpix pixel IDs. False for RING format.
            density (bool): Treat the 

        Returns:
            (np.ndarray): values of the multiorder map for each unique pixel ID
        """
        # levels, nside, pixel id, and angle of unique pixels
        level, ipix = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)
        th, ph = hp.pix2ang(nside, ipix, nest=True)
        # NOTE: uniq pixels always used nested ordering

        # nside, pixel id, and angle of input map
        input_nside = hp.npix2nside(input_map.size)
        input_ipix = np.arange(input_map.size)
        input_th, input_ph = hp.pix2ang(input_nside, input_ipix, nest=nest)

        # output map and status of output pixels
        output = np.zeros(uniq.size, np.float64)
        filled = np.zeros(uniq.size, np.int64)

        # index of overlapping unique pixel for input map pixel
        i = self._ang_to_index(uniq, input_th, input_ph)

        # fill any downgraded or equal resolution pixels
        mask = (nside[i] <= input_nside)
        fill = i[mask]
        np.add.at(output, fill, input_map[input_ipix[mask]])
        np.add.at(filled, fill, 1)
        if density:
            output[fill] /= np.float64(filled[fill])

        # fill the remaining pixels which are at
        # higher resolutions in the multiorder map
        unfilled = (filled == 0)
        i = hp.ang2pix(input_nside, th[unfilled], ph[unfilled], nest=nest)
        if density:
            output[unfilled] = input_map[i]
        else:
            scale = 4.**(level[unfilled] - ah.nside_to_level(input_nside))
            output[unfilled] = input_map[i] / scale

        return output

    def apply_minimum_nside(self, uniq, val, min_nside, density=True):
        r""" Ensure a multiorder map has nside >= min_side for all pixels

        Args:
            uniq (np.ndarray):
                Array with UNIQ pixel ids of original map
            val (np.ndarray or astropy.table.table.Table):
                Array with original map values
            min_nside (int):
                Minimum nside to enforce
            density (bool):
                Treat map values as density when True

        Returns:
            (np.ndarray, np.ndarray): Arrays with new UNIQ pixel ids and values of the map
        OR
            (astropy.table.table.Table): Table with values of map(s) with min resolution
        """
        # orignal multi-order map information
        level, ipix = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)
        i = np.arange(uniq.size) # original array indices for uniq/val

        # return original uniq, val when minimum nside is already satisfied
        if np.min(nside) >= min_nside:
            return uniq, val

        # information for healpix map at the specified minimum nside
        min_nside_level = ah.nside_to_level(min_nside)
        min_nside_pixels = np.arange(hp.nside2npix(min_nside))
        th, ph = hp.pix2ang(min_nside, min_nside_pixels, nest=True)

        # match pixels from minimum nside map to there counterparts in i
        min_nside_i = self._ang_to_index(uniq, th, ph)

        # masks of which pixels need to be upgraded to the
        # minimum nside and which need to be kept as-is
        to_upgrade = (nside < min_nside)
        to_keep = ~to_upgrade
        to_keep_min_nside = np.isin(min_nside_i, i[to_upgrade])

        min_nside_i = min_nside_i[to_keep_min_nside]
        min_nside_pixels = min_nside_pixels[to_keep_min_nside]

        # gather new uniq indices and map values for pixels
        # that will be upgraded to the minimum nside
        new_uniq = ah.level_ipix_to_uniq(min_nside_level, min_nside_pixels)
        new_val = val[min_nside_i]
        if density == False:
            new_val /= 4.**(min_nside_level - level[min_nside_i])

        # this return allows us to handle multiple map columns as
        # a single sorted astropy Table
        if isinstance(val, astropy.table.table.Table):
            new_val = astropy.table.vstack([new_val, val[to_keep]])
            new_val['UNIQ'] = np.concatenate([new_uniq,  uniq[to_keep]])
            new_val.sort('UNIQ')
            return new_val

        # this returns handles a single healpix map given as an array
        return np.concatenate([new_uniq,  uniq[to_keep]]), \
               np.concatenate([new_val,  val[to_keep]])

    def probability(self, ra, dec, per_pixel=False, interp=True):
        """Calculate the localization probability at a given point while
        handling fixed resolution and multiorder resolutions

        Args:
            ra (float): The right ascension in degrees
            dec (float): The declination in degrees
            per_pixel (bool, optional): If True, return probability/pixel, otherwise return probability
                                        per square degree. Default is False.
            interp (bool, optional): If True, interpolate between nearest neighbors of fixed resolution maps

        Returns:
            (float): The localization probability
        """
        if interp and self._uniq is None:
            phi = self._ra_to_phi(ra)
            theta = self._dec_to_theta(dec)
            prob = hp.get_interp_val(self.prob, theta, phi)
        else:
            pix = self._ang_to_pix(ra, dec)
            prob = self.prob[pix]

        if not per_pixel:
            prob /= self.pixel_area
        return prob

    def region_probability(self, healpix, prior=0.5):
        """The probability that the HealPix localization is associated with
        the localization region from another HealPix map.  This is calculated 
        against the null hypothesis that the two HealPix maps represent 
        unassociated sources, each having equal probability of origination 
        anywhere in the sky. 
        
        Note: Localization regions overlapping the Earth will be zeroed out.
        The calculation is performed over the unocculted sky.
        
        Args:
            healpix (HealPix):
                The healpix map for which to calculate the spatial association.
            prior (float, optional):
                The prior probability that the localization is associated
                with the source. Default is 0.5
        
        Returns:
            (float): The probability that the two HealPix maps are associated.
        """
        if (prior < 0.0) or (prior > 1.0):
            raise ValueError('Prior probability must be within 0-1, inclusive')
        # convert uniform prob/sr to prob/pixel
        u = 1.0 / (4.0 * np.pi)
        
        # ensure maps are the same resolution and convert uniform prob/sr to 
        # prob/pixel
        probmap1 = self.prob
        probmap2 = healpix.prob
        uniq2 = hasattr(healpix, "_uniq") and healpix._uniq is not None
        if self._uniq is not None and not uniq2:
            probmap2 = self.proj_to_multiorder(self._uniq, probmap2, density=False)
            probmap2 = self._assert_prob(probmap2)
            u *= self._uniq_to_resol(self._uniq)**2
        elif self._uniq is None and uniq2:
            probmap1 = self.proj_to_multiorder(healpix._uniq, probmap1, density=False)
            probmap1 = self._assert_prob(probmap1)
            u *= self._uniq_to_resol(healpix._uniq)**2
        elif self.nside > healpix.nside:
            probmap2 = hp.ud_grade(probmap2, nside_out=self.nside)
            probmap2 = self._assert_prob(probmap2)
            u *= hp.nside2resol(self.nside)**2
        elif self.nside < healpix.nside:
            probmap1 = hp.ud_grade(probmap1, nside_out=healpix.nside)
            probmap1 = self._assert_prob(probmap1)
            u *= hp.nside2resol(healpix.nside)**2
        else:
            u *= hp.nside2resol(self.nside)**2
        # NOTE: need one more option to handle case where two multiresolution
        # maps with different uniq columns. Flatten both to max nside.
        # Also, should create a match_resolutions function to do this step.
        # Have it return matched probmap1, probmap2, resol**2
        
        # alternative hypothesis: they are related
        alt_hyp = np.sum(probmap1*probmap2)
        # null hypothesis: one of the maps is from an unassociated source
        # (uniform spatial probability)
        null_hyp = np.sum(probmap1 * u)

        # since we have an exhaustive and complete list of possibilities, we can
        # easily calculate the probability
        prob = (alt_hyp*prior) / ((alt_hyp*prior) + (null_hyp*(1.0-prior)))
        return prob
 
    @classmethod
    def open(cls, filename, min_nside=128, flatten=False, prob_only=False):
        """Open a LIGO/Virgo healpix FITS file and return the HealPix object

        Args:
            filename (str): The filename of the FITS file
            min_nside (int): minimum NSIDE value to enforce
            flatten (bool): flatten multi-order resolution maps to single resolution when True
            prob_only (bool): only load localization probability when True. Skip source distance information.
        
        Returns:
            (LigoHealPix): The HealPix object
        """
        obj = cls()

        # obj._file_properties(filename)
        obj._filename = filename

        # open FITS file
        with fits.open(filename) as hdulist:
            for hdu in hdulist:
                obj._headers.update({hdu.name: hdu.header})

        # handle multiresolution file
        if obj._headers[obj._data_ext()]['ORDERING'] == 'NUNIQ':
            if prob_only:
                uniq, area_sr, prob_density = \
                    obj._load_multiorder(filename, min_nside, flatten, columns=['PROBDENSITY'])
            else:
                uniq, area_sr, prob_density, dist_mu, dist_sig, dist_norm = \
                    obj._load_multiorder(filename, min_nside, flatten)
                obj._dist_mu = dist_mu
                obj._dist_sig = dist_sig
                obj._dist_norm = dist_norm
            obj._uniq = uniq
            obj._hpx = prob_density * area_sr
        # handle files that include distance info...or just localization
        elif obj._headers[obj._data_ext()]['TFIELDS'] == 4 and not prob_only:
            obj._hpx, obj._dist_mu, obj._dist_sig, obj._dist_norm = \
                           hp.read_map(filename, field=(0,1,2,3))
            obj._nside = hp.get_nside(obj._hpx)
            area_sr = hp.nside2pixarea(obj._nside)
        else:
            obj._hpx = hp.read_map(filename, field=(0,))
            obj._nside = hp.get_nside(obj._hpx)
            area_sr = hp.nside2pixarea(obj._nside)
        obj._hpx /= np.sum(obj._hpx)
        
        # calculate the significance levels
        obj._sig = 1.0-find_greedy_credible_levels(obj._hpx, area_sr)

        return obj
    
    def _data_ext(self):
        """ Method to get key to the data header

        Returns:
            (str): name of the header with data information
        """
        return list(self.headers.keys())[1]

    def _load_multiorder(self, filename, min_nside=128, flatten=False,
                         columns=['PROBDENSITY', 'DISTMU', 'DISTSIGMA', 'DISTNORM']):
        """ Method to load a multiorder map columns

        Args:
            filename (str): The filename of the FITS file
            min_nside (int): minimum NSIDE value to enforce
            flatten (bool): flatten multi-order resolution maps to single resolution when True
            columns (list): column names to load
        
        Returns:
            (list): list with the unique pixel IDs, pixel areas, and requested column values
        """
        table = Table.read(filename)
        table.keep_columns(['UNIQ'] + columns)

        # uniq pixel ids for every map pixel and their area
        uniq = np.array(table['UNIQ'])
        area_sr = self._uniq_to_pixel_area(uniq, degrees=False)

        if flatten:
            level, ipix = ah.uniq_to_level_ipix(uniq)
            min_nside = np.max(ah.level_to_nside(level))

        table = self.apply_minimum_nside(uniq, table, min_nside)

        col_arr = [np.array(table[c]) for c in columns]

        # reset uniq, area arrays to match the maps
        # obtained after applying the minimum nside
        uniq = np.array(table['UNIQ'])
        area_sr = self._uniq_to_pixel_area(uniq, degrees=False)

        if flatten:
            # convert to ring format so it matches other unflattened maps
            for i, c in enumerate(col_arr):
                col_arr[i] = hp.reorder(c, n2r=True)

            # set uniq to None and scalar area because map is now flat
            uniq = None
            area_sr = area_sr[0]
 
        return [uniq, area_sr] + col_arr

    def _ang_to_pix(self, ra, dec):
        """ Method to convert RA/Dec to healpixels while accounting for multiorder behavior

        Args:
            ra (float or np.ndarray): right ascension in degrees
            dec (float or np.ndarray): declination in degrees

        Returns:
            (int or list): pixel(s) for the given RA/Dec
        """
        theta = self._dec_to_theta(dec)
        phi = self._ra_to_phi(ra)
        if self._uniq is None:
            pix = hp.ang2pix(self.nside, theta, phi)
        else:
            pix = self._ang_to_index(self._uniq, theta, phi)
        return pix
 
    def _mesh_grid(self, num_phi, num_theta):
        """ Method to create the mesh grid in phi and theta

        Args:
            num_phi (int): number of phi points to use
            num_theta (int): number of theta points to use

        Returns:
            (np.ndarray, np.ndarry, np.ndarray): arrays with healpix indices as well as phi, theta value of the grid
        """
        theta = np.linspace(np.pi, 0.0, int(num_theta))
        phi = np.linspace(0.0, 2*np.pi, int(num_phi))
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        if self._uniq is None:
            grid_pix = hp.ang2pix(self.nside, theta_grid, phi_grid)
        else:
            grid_pix = self._ang_to_index(self._uniq, theta_grid, phi_grid)
        return (grid_pix, phi, theta)

    @staticmethod
    def _ang_to_index(uniq, theta, phi):
        """ Convert from theta, phi to index of pixel
        in the UNIQ pixel array

        NOTE: This function is based on LIGO's multiorder guide
        https://emfollow.docs.ligo.org/userguide/tutorial/multiorder_skymaps.html

        Args:
            uniq (int, array-like): UNIQ healpix pixel ID(s)
            theta (float, array-like): Theta angle in radians (0 to pi/2)
            phi (float, array-like): Phi angle in radians (-pi to +pi)

        Returns:
            (int, array-like):
                Index/Indices of element in the UNIQ array. This can
                be used to index the map value at a given pixel.
        """
        level, ipix = ah.uniq_to_level_ipix(uniq)

        max_level = 29
        max_nside = ah.level_to_nside(max_level)
        index = ipix * (2**(max_level - level))**2

        sorter = np.argsort(index)
        match_ipix = ah.lonlat_to_healpix(
            (2 * np.pi * (phi < 0) + phi) * au.rad, 
            (np.pi / 2. - theta) * au.rad, 
            max_nside, order='nested')

        return sorter[
            np.searchsorted(index, match_ipix, side='right', sorter=sorter) - 1]

    @staticmethod
    def _vec_to_index(uniq, vectors):
        """ Convert from vector to pixel index

        Args:
            uniq (int, array-like):
                UNIQ healpix pixel ID(s)
            vectors (array-like float):
                The vector(s) to convert, shape is (3,) or (N, 3)

        Returns:
            (int, array-like):
                Index/Indices of element in the UNIQ array. This can
                be used to index the map value at a given pixel.
        """
        th, ph = hp.pixelfunc.vec2ang(vectors)
        return self._ang_to_index(uniq, th, ph)

    def _uniq_to_pixel_area(self, uniq, degrees=False):
        """ Retrieve area of each pixel
 
        Args:
            uniq (int, array-like):
                UNIQ healpix pixel ID(s)
            degrees (bool):
                Return answer in square degrees when true

        Returns:
            (float, array-like):
                Area of pixel(s) in sr or degrees
        """
        nside, ipix = self._uniq_to_nside_ipix(uniq)
        return hp.nside2pixarea(nside, degrees)

    def _uniq_to_resol(self, uniq, arcmin=False):
        """ Retrieve area of each pixel
 
        Args:
            uniq (int, array-like):
                UNIQ healpix pixel ID(s)
            degrees (bool):
                Return answer in square degrees when true

        Returns:
            (float, array-like):
                Area of pixel(s) in sr or degrees
        """
        nside, ipix = self._uniq_to_nside_ipix(uniq)
        return hp.nside2resol(nside, arcmin)

    @staticmethod
    def _uniq_to_nside_ipix(uniq):
        level, ipix = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)
        return nside, ipix

    @staticmethod
    def _uniq_to_nside_ipix(uniq):
        level, ipix = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)
        return nside, ipix

    def _uniq_to_ang(self, uniq):
        nside, ipix = self._uniq_to_nside_ipix(uniq)
        return hp.pix2ang(nside, ipix, nest=True)

class IceCubeHealPix(HealPixLocalization):
    """Class for handling  asymmetric error regions from IceCube"""
    @classmethod
    def from_ellipsoid(cls, center_ra, ra_err_minus, ra_err_plus,
                       center_dec, dec_err_minus, dec_err_plus,
                       clevel=1.0, nside=64):
        """Create a HealPix object of a top-hat ellipsoid with asymmetric
           errors on both RA and DEC

        Args:
            center_ra (float):
                The RA of the center of the ellipsoid
            ra_err_minus (float):
                The lower error bound on RA
            ra_err_plus (float):
                The upper error bound on RA
            center_dec (float):
                The Dec of the center of the ellipsoid
            dec_err_minus (float):
                The lower error bound on Dec in degrees
            dec_err_plus (float):
                The upper error bound on Dec in degrees
            clevel (float):
                Confidence level (0 - 1) contained in the ellipsoid contour
            nside (int, optional):
                The nside of the HealPix to make. Default is 64.
        
        Returns:
            (HealPix): The map as a HealPix object
        """
        # gather center declination and errors in radians
        dec_rad, dec_minus_rad, dec_plus_rad = np.radians([
            center_dec, dec_err_minus, dec_err_plus])
        if dec_minus_rad > 0:
            raise ValueError("Lower DEC error bound must be <= 0")
        if dec_plus_rad < 0:
            raise ValueError("Upper DEC error bound must be >= 0")

        # gather center right ascension and errors in radians
        if center_ra is not None:
            ra_rad, ra_minus_rad, ra_plus_rad = np.radians([
                center_ra, ra_err_minus, ra_err_plus])
            if ra_minus_rad > 0:
                raise ValueError("Lower RA error bound must be <= 0")
            if ra_plus_rad < 0:
                raise ValueError("Upper RA error bound must be >= 0")

        # locations of healpix pixels
        i = np.arange(hp.nside2npix(nside))
        th, ph = hp.pix2ang(nside, i)
        ra, dec = ph, np.pi / 2. - th

        if center_ra is not None: # do error ellipse when ra errors are present

            # avoid wrapping near 2 * pi boundary
            dra = ra - ra_rad
            ra[dra > np.pi] -= 2 * np.pi
            ra[dra < -np.pi] += 2 * np.pi

            r1 = np.where(ra < ra_rad,
                ra_minus_rad * np.cos(dec_rad),
                ra_plus_rad * np.cos(dec_rad))
            r2 = np.where(dec < dec_rad,
                dec_minus_rad,
                dec_plus_rad)

            # map angular differences to polar coordinates
            r = hp.rotator.angdist([np.pi / 2. - dec_rad, ra_rad],
                                   np.array([np.pi / 2. - dec, ra]))
            angle = cls._plane_angle(ra_rad, dec_rad, ra, dec)

            # mask points inside the ellipse defined by r1, r2
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            mask = (x / r1)**2 + (y / r2)**2 <= 1
        else: # otherwise use dec errors only
            mask = (dec_rad + dec_minus_rad <= dec) & \
                   (dec <= dec_rad + dec_plus_rad)

        # probability map for error ellipse
        probmap = np.zeros(i.size, np.float64)
        probmap[mask] = clevel / mask.sum()
        probmap[~mask] = (1.0 - clevel) / (~mask).sum()

        obj = cls.from_data(probmap)
        return obj

    @staticmethod
    def _plane_angle(ra0, dec0, ra1, dec1):
        r""" Compute angular orientation of a plane passing through
        (ra0, dec0) and (ra, dec). Provided angle is relative to the
        +z axis after rotating (ra0, dec0) to (0, 0).

        Args:
            ra0 (float):
                Right ascension of reference point in radians
            dec0 (float):
                Declination of reference point in radians
            ra1 (float, np.ndarray(float)):
                Right ascension in radians of second point(s)
                defining a plane with (ra0, dec0)
            dec1 (float, np.ndarray):
                Declination in radians of second point(s)
                defining a plane with (ra0, dec0)

        Returns:
            (float or np.ndarray): plane angle(s) in radians
        """

        # internally handle arrays vs scalar types
        if isinstance(ra1, np.ndarray) or isinstance(dec1, np.ndarray):
            return_array = True
        else:
            return_array = False

        ra1 = np.atleast_1d(ra1)
        dec1 = np.atleast_1d(dec1)

        # relative angles
        th = np.pi / 2. - (dec1 - dec0)
        ph = ra1 - ra0

        # ensure 0 - pi domain for th angle
        mask = th < 0
        th[mask] = -th[mask]
        ph[mask] += np.pi

        mask = th > np.pi
        th[mask] = 2 * np.pi - th[mask]
        ph[mask] += np.pi

        x = np.array([1., 0., 0.]) # +x axis vector
        z = np.array([0., 0., 1.]) # +z axis vector

        # vector representation of (ra, dec) in coord system
        # where (ra0, dec0) is rotated so dec0 == 0
        vec = hp.ang2vec(th, ph)

        # normal vector for plane of (ra0, dec0) and (ra, dec)
        n = np.cross(vec, x)
        n /= np.linalg.norm(n, axis=1)[:,np.newaxis]

        # compute angle between this normal vector and +z axis
        angle = np.arccos(np.linalg.norm(n * z, axis=1))

        mask = (ra1 < ra0) & (dec1 >= dec0)
        angle[mask] = np.pi - angle[mask]
        mask = (ra1 < ra0) & (dec1 < dec0)
        angle[mask] += np.pi
        mask = (ra1 >= ra0) & (dec1 < dec0)
        angle[mask] = 2 * np.pi - angle[mask]

        if return_array:
            return angle
        return angle[0]
