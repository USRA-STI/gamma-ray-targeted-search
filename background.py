
import numpy as np
from scipy.interpolate import interp1d

class BackgroundRatesMatrix:
    """Class for retrieving the background counts matrix.

    To-Do:
        1. Add method to compute goodness-of-fit flags

    Parameters:
        fitters (DataCollection): collection of background fitters
    """ 
    def __init__(self, fitters):
        """Constructor"""
        self.fitters = fitters

    @property
    def detectors(self):
        """(list): list of detector names"""
        return self.fitters.items

    @property
    def ebounds(self):
        """(list): list of ebounds object for each detector"""
        # convenience method for validating energy binning against data classes
        return [fitter._data_obj.ebounds for fitter in self.fitters]

    def counts(self, tstart, tstop, exposure):
        """Calculate background counts from the interpolated
        background rate over a given window.

        Note: user must provide exposure to ensure background counts
        have the same exposure as the data. This is particularly
        important when working with binned data since failing to
        provide the exposure will result in a mismatch unless the
        background has an indentical time binning.

        Args:
            tstart (float): start time of the counts window
            tstop (float): stop time of the counts window
            exposure (list): list of exposure for each detector

        Returns:
            (np.array, np.array, np.array): arrays with background counts,
                                            counts variance, goodness of fit
        """
        tstart = np.atleast_1d(tstart)
        tstop = np.atleast_1d(tstop)
        counts, counts_var, good = [], [], []
        for i, fitter in enumerate(self.fitters):
            rates, rate_uncert = fitter._method.interpolate(tstart, tstop)
            counts.append(rates[0] * exposure[i])
            counts_var.append(0.5 * (rate_uncert[0] * exposure[i])**2)
            good.append(np.ones_like(counts[-1], dtype=bool)) # set to True for now

        return np.ravel(counts), np.ravel(counts_var), np.ravel(good)
