# CONTAINS TECHNICAL DATA/COMPUTER SOFTWARE DELIVERED TO THE U.S. GOVERNMENT WITH UNLIMITED RIGHTS
#
# Contract No.: CA 80MSFC17M0022
# Contractor Name: Universities Space Research Association
# Contractor Address: 7178 Columbia Gateway Drive, Columbia, MD 21046
#
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
from scipy.special import erf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utils

class Likelihood():
    """The log-likelihood ratio calculation (Blackburn+ 2015)
    
    :
    -----------
    numspec: int
        Number of spectral templates used
    numsky: int
        Number of sky points used
    gamma: int, optional
        Power-law prior parameter for the amplitude. Default is 2.5
    num_iters: int, optional
        The number of iterations performed in the maximization. Default is 3
    prethresh: float, optional
        The initial guess threshold above which the numerical maximization 
        will commence. Default is 5.0
    
    Attributes:
    -----------
    chisq: (float, float)
        The reduced chi-squared including and excluding zero-count bins
        
    Public Methods:
    ---------------
    calculate:
        Calculate the loglr given the data, background, and responses
    coinclr:
        Calculate the likelihood ratio given a sky prior
    """

    # keep scale-free: beta = 1.0
    def __init__(self, numspec, numsky, gamma=2.5, num_iters=3, prethresh=5.0, skyres=5):
        """ Class constructor

        Args:
            numspec (int):
                Number of spectral templates used
            numsky (int):
                Number of sky points used
            gamma (int, optional):
                Power-law prior parameter for the amplitude. Default is 2.5
            num_iters (int, optional):
                The number of iterations performed in the maximization. Default is 3
            prethresh (float, optional):
                The initial guess threshold above which the numerical maximization 
                will commence. Default is 5.0
            skyres (float, option):
                Resolution spacing of skygrid points in degrees
        """
        self._nspec = numspec
        self._nsky = numsky
        self._gamma = gamma
        self._num_iters = num_iters
        self._prethresh = prethresh
        self._lref = -np.log(gamma)
        
        self._nvisible = None
        self._fit_flag = 0
        self._llratio = None
        self._pflux = None
        self._pflux_sig = None
        self._llnorm = None
        self._llmarg = None
        self._max_idx = None
        self._rsp = None
        self._max_rsp = None
        self._counts = None
        self._bkgd_counts = None

        self._skyres = skyres

    @property
    def status(self):
        """(int): Status of the fit.  O=failed; 1=succeeded; 2=prefiltered"""
        return self._fit_flag
    
    @property
    def llr(self):
        """(np.ndarray): The log-likelihood ratio for each template and sky point"""
        return self._llratio
    
    @property
    def max_template(self):
        """(int): The template value which maximizes the likelihood"""
        specmarg = np.exp(self._llnorm).sum(axis=1)
        return np.argmax(specmarg)
    
    @property
    def max_location(self):
        """(int): The index of the sky grid location which maximizes the likelihood"""
        locmarg = np.exp(self._llnorm).sum(axis=0)
        return np.argmax(locmarg)
    
    @property
    def marginal_llr(self):
        """(float): The fully marginalized log-likelihood ratio, loglr"""
        return self._llmarg
    
    @property
    def photon_fluence(self):
        """(float): The photon fluence that maximizes the likelihood"""
        return self._pflux[self._max_idx]
    
    @property
    def photon_fluence_sigma(self):
        """(float): The standard deviation of the photon fluence"""
        return self._pflux_sig[self._max_idx]
            
    @property
    def optimal_snr(self):
        """(float): The signal-to-noise ratio of the model at the max likelihood location"""
        num = np.sum(self._max_rsp*(self._counts-self._bkgd_counts)/self._bkgd_counts)
        denom = np.sqrt(np.sum(self._max_rsp**2/self._bkgd_counts))
        return num/denom
    
    @property
    def chisq(self):
        """(float): The chi-square of the model at the max likelihood location"""
        error = self._counts-self._bkgd_counts-(self._max_rsp*self.photon_fluence)
        chisq = error**2/(self._bkgd_counts+(self._max_rsp*self.photon_fluence))
        reduced_chisq = np.sum(chisq)/len(chisq)
        chiplusdof = np.sum(chisq[error > 0])/np.sum(error > 0)
        return (reduced_chisq, chiplusdof)


    def _flatten_data(self, counts, bkgd_counts, bkgd_var, rsp):
        """ Method to flatten out everything into 2 axes: (variables, measurements)

        Args:
            counts (np.ndarray): observed counts for all energies
            bkgd_counts (np.ndarray): background counts for all energies
            bkgd_var (np.ndarray): uncertainty on background counts for all energies
            rsp (np.ndarray): instrument response matrix

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                flattened counts, background, background uncertainty, excess above background, response
        """
        f = counts.ravel()[np.newaxis, :]  
        b = bkgd_counts.ravel()[np.newaxis, :]
        vb = bkgd_var.reshape(b.shape)
        # do not allow unphysical negative background estimate
        b = np.maximum(0, b)
        d = f - b  # excess above background counts
        # cast to 2D [spectrum*locations, detectors*channels]
        r = rsp.reshape((-1, counts.size))
        
        return f, b, vb, d, r


    def _apply_prethreshold(self, vn, vfinv, d, f, p, s, rsq):
        """ Applies the pre-threshold.
        This takes the initial guess for the amplitude (photon fluence; s)
        and calculates the loglr

        Args:
            vn (np.ndarray): total variance in background counts (poisson + fit error)
            vfinv (np.ndarray): inverse of the total variance of foreground counts
            d (np.ndarray): excess counts above backgrouund
            f (np.ndarray): observed counts
            p (np.ndarray): predict counts from the model (background + source)
            s (np.ndarray): predicted source counts
            rsq (np.ndarray): square of the response matrix

        Returns:
            (bool, np.ndarray, np.ndarray): tuple with fit status, marginalized log-likelihood ratio, and variance of the likelihood
        """
        llratio, sqvinv = self._marginalized_likelihood(vn, vfinv, d, f, p, s, rsq)

        self._normalize_likelihood(llratio)
        if self._llmarg < self._prethresh:
            return True, llratio, sqvinv
        else:
            return False, llratio, sqvinv


    def _max_amplitude(self, f, b, vb, r, vfinv, s, spos, p):
        """ Newtonian extrapolation to find true max likelihood

        Args:
            f (np.ndarray): observed counts
            b (np.ndarray): background counts
            vb (np.ndarray): variance in background counts 
            r (np.ndarray): response matrix
            vfinv (np.ndarray): inverse of the total variance of foreground counts
            s (np.ndarray): predicted source counts
            spos (np.ndarray): mask for positive predicted source counts. True for s > 0
            p (np.ndarray): predict counts from the model (background + source)

        Returns:
            (float, float, float, float): maximized values of vfinv, s, spos, p
        """
        for i in range(self._num_iters):
            # alpha factor for simplification
            a = (f-p)*vfinv
            # s derivative of vf
            dvf = spos[:,np.newaxis] * r
            asqmvfinv = a**2 - vfinv
            # exact first and second derivatives dl/ds
            dl = np.sum(r*a + 0.5*dvf*asqmvfinv, axis=-1)
            ddl = np.sum(-vfinv * (dvf*a + r)**2 + 0.5*(dvf*vfinv)**2, axis=-1)
            # new guess for source ampltiude (dL/da = 0)
            s -= dl/ddl
            # second predicted foreground rate
            p = b + r*s[:,np.newaxis]
            # only include source terms to variance for positive s
            spos = s > 0
            # total foreground variance
            vf = np.maximum(b, p) + vb
            # inverse to save on divides
            vfinv = 1.0/vf
        return vfinv, s, spos, p


    def _marginalized_likelihood(self, vn, vfinv, d, f, p, s, rsq):
        """ The likelihood ratio marginalized over a power law prior for source flux.
        Prior has the form

        P(s) = (1 - exp(-(s / gamma sigma_L)) s^-1

        where sigma_L is the square root of the variance on the likelihood (formulated below as sqvinv = 1/sigma_L).

        Args:
            vn (np.ndarray): total variance in background counts (poisson + fit error)
            vfinv (np.ndarray): inverse of the total variance of foreground counts
            d (np.ndarray): excess counts above backgrouund
            f (np.ndarray): observed counts
            p (np.ndarray): predict counts from the model (background + source)
            s (np.ndarray): predicted source counts
            rsq (np.ndarray): square of the response matrix

        Returns:
            (np.ndarray, np.ndarray): marginalized log likelihood and the inverse square root of the likelihood variance
        """        
        # max likelihood
        ll = np.sum(0.5*(np.log(vn*vfinv) + d**2/vn - (f-p)**2 * vfinv), axis=-1)  
        # positive indices where prior is valid, avoid numerical issues near 0
        spos = s > (1e-6)

        # standard deviation of L distrib peak (for approx maginalization over s)
        sqvinv = np.sqrt(np.sum(rsq * vfinv, axis=-1))
        logsqv = -np.log(sqvinv)
        
        # log prior flattened at s < gamma * sigma
        logppos = np.log(1.0-np.exp(-(s*(1.0/self._gamma)*sqvinv)[spos])) \
                  - np.log(s[spos])
        # log prior at s = 0, we need all logsqv indices for llmarg anyway
        logp = -(np.log(self._gamma) + logsqv)
        # place logprior for positive indices
        np.place(logp, spos, logppos)
        # overlap between gaussian and s>0 region (error function)
        # technically logo should have extra term of -ln(2), but paper formula 
        # skips because lref can cancel it out
        logo = np.log(1.0 + erf(s*sqvinv*(1.0/np.sqrt(2.0))))

        # full marginalized likelihood
        # log likelihood = amplitude prior + approx marginalization over s 
        # + max log likelihood over s
        llmarg = logsqv + logo + logp + ll  
        
        # subtract off the reference likelihood
        llratio = llmarg-self._lref
        
        # reshape into input shape 
        llratio = llratio.reshape(self._nspec, -1)
        sqvinv = sqvinv.reshape(self._nspec, -1)
        
        return llratio, sqvinv

    def _normalize_likelihood(self, ll):
        """ Marginalization of the likelihood over the sky + spectral templates.
        This is performed with a uniform prior so the marginalization term is just -np.log(self._nspec * self._nsky).

        Args:
            ll (np.ndarray): the log likelihood ratio to marginalize
        """
        imax = np.argmax(ll)
        
        # we have to subtract np.log(nspec) to assume flat prior over spectra
        # sky prior is already normalized to sum=1 so we don't need to factor that here
        llmax = ll.ravel()[imax]
        self._llnorm = ll - llmax
        self._llmarg = llmax + np.log(np.exp(self._llnorm).sum()) - \
                       np.log(self._nspec * self._nsky)

        self._max_idx = np.unravel_index(imax, ll.shape)

        self._max_rsp = self._rsp[self._max_idx]

    def calculate(self, counts, bkgd_counts, bkgd_var, rsp):
        """Calculate the loglr given the data, background, and responses

        Args:
            counts (np.ndarray):
                The observed counts data
            bkgd_counts (np.ndarray):
                The modeled background counts
            bkgd_var (np.ndarray):
                The variance of the background counts fit/estimate
            rsp (np.ndarray):
                The response for each sky point and template
        """
        # f := foreground
        # b := background
        # vb := background variance
        self._nvisible = rsp.shape[1]
        self._rsp = rsp
        self._counts = counts
        self._bkgd_counts = bkgd_counts
        f, b, vb, d, r = self._flatten_data(counts, bkgd_counts, bkgd_var, rsp)
        rsq = r**2
    
        # constant vf chi-sq solution initial guess for source amplitude
        # total variance in background counts (poisson + fit error)
        vn = b + vb
        # inverse of the total variance of foreground counts
        vfinv = 1.0 / np.resize((np.maximum(b, f) + vb), r.shape)

        s = np.sum(r * d * vfinv, axis=-1) / np.sum(rsq * vfinv, axis=-1)
        # response uncertainty will only contribute for positive amplitude
        spos = s > 0
        # predicted foreground rate
        p = b + r * s[:, np.newaxis]
        
        # the pre-filter
        # anything below the pre-filter does not advance to the numerical
        # estimate stage
        is_noise, ll, sqvinv = self._apply_prethreshold(vn, vfinv, d, f, p, s, rsq)
        if is_noise:
            self._llratio = ll
            self._pflux = s.reshape(self._nspec, -1)
            self._pflux_sig = 1.0/sqvinv
            self._fit_flag = 2
            return
            
        # find the maximal amplitude (s_best)
        vfinv, s, spos, p = self._max_amplitude(f, b, vb, r, vfinv, s, spos, p)
        
        # produce the marginalized log-likelihood ratio
        llratio, sqvinv = self._marginalized_likelihood(vn, vfinv, d, f, p, s, rsq)
        self._normalize_likelihood(llratio)
        
        # save some stuff
        self._llratio = llratio
        self._pflux = s.reshape(self._nspec, -1)
        self._pflux_sig = 1.0/sqvinv
        self._fit_flag = 1

    def plotLlr(self):
        """ Method to plot the log likelihood ratio """
        # Get the sky grid
        skyGrid = utils.SkyGrid(5)

        theta_grid = skyGrid.radians[1] # 0 to pi
        phi_grid = skyGrid.radians[0] # 0 to 2pi)
          
        x = np.sin(theta_grid)* np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)

        data = self._llratio 
        data = np.nan_to_num(data)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatterplot = ax.scatter(x, y, z, marker='o', s=50, c=data[0], cmap='jet')
        ax.view_init(azim=90)

        # Add a color bar and change the label format and text accordingly
        colorBarObject = plt.colorbar(scatterplot, orientation='vertical', pad=0.1, aspect=25, shrink=0.7, ax=ax)                                
        colorBarObject.set_label(r'log likelihood ratio', size=10)

        # Change the label size
        colorBarObject.ax.tick_params(labelsize=10) 

        plt.show()

    def plotPflux(self):
        """ Method to plot the fitted photon flux """

        # Get the sky grid
        skyGrid = utils.SkyGrid(5)

        theta_grid = skyGrid.radians[1] # 0 to pi
        phi_grid = skyGrid.radians[0] # 0 to 2pi)
          
        x = np.sin(theta_grid)* np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)

        data = self._pflux
        data = np.nan_to_num(data)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatterplot = ax.scatter(x, y, z, marker='o', s=50, c=data[0], cmap='jet')

        # Add a color bar and change the label format and text accordingly
        colorBarObject = plt.colorbar(scatterplot, orientation='vertical', pad=0.1, aspect=25, shrink=0.7, ax=ax)                                
        colorBarObject.set_label(r'Source flux', size=10)

        # Change the label size
        colorBarObject.ax.tick_params(labelsize=10) 

        plt.show()

    def localize(self):
        """ Return the best-fit location

        Returns:
            (float, float): best-fit location as azimuth and zenith in the spacecraft frame
        """
        # Generate the sky grid
        skyGrid = utils.SkyGrid(self._skyres)

        # get the location, sun-, geo-angle of the max likelihood
        max_az, max_zen = skyGrid.degrees[:,self.max_location]

        return max_az, max_zen
    
    def coinclr(self, logskyprior, **kwargs):
        """Calculate the likelihood ratio given a sky prior

        Args:
        logskyprior (np.ndarray):
            The logarithm of the sky prior
        llratio (np.ndarray):
            Likelihoord ratio array for multiplication with logskyprior (optional)        

        Returns:
            (float): The marginalized log-likelihood ratio
        """
        llratio = kwargs.pop("llratio", self._llratio)
        
        # return 0 in cases where the sky prior is 0 over the visible sky
        if llratio.shape[1] == 0:
            return 0.
        
        # (add) multiply the (log) sky prior
        llcoinc = llratio + logskyprior[np.newaxis, :]
        
        # normalize and get the marginalized log-likelihood ratio
        icoincmax = np.argmax(llcoinc)
        llmaxcoinc = llcoinc.ravel()[icoincmax]
        llnormcoinc = llcoinc - llmaxcoinc
        llmargcoinc = llmaxcoinc + np.log(np.exp(llnormcoinc).sum())- \
                       np.log(self._nspec)
        
        return llmargcoinc
    
    def prior_weighted_fluence(self, logskyprior, **kwargs):

        skyprior = np.exp(logskyprior)
        pflux = kwargs.pop("pflux", self._pflux)
        pflux_sig = kwargs.pop("pflux_sig", self._pflux_sig)
        weighted_pflux = np.sum(skyprior[np.newaxis,:]*pflux, axis=1)
        weighted_pflux_sig = np.sqrt(np.sum((skyprior[np.newaxis,:]*pflux_sig)**2,
                                            axis=1))
        return np.vstack((weighted_pflux, weighted_pflux_sig))    
