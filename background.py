
import numpy as np

from gdt.core.binning.binned import rebin_by_edge_index
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.background.unbinned import NaivePoisson

"""
Thoughts:

Original search used a background model that interpolated rate on a 0.256 resolution alone.
Could have a method use an internal interpolation on rates alone to avoid summing 0.064 sec bins.
Should check what the speed difference is for summing versus interpolation.

Need a method to apply goodness-of-fit check. Have an external method that will check padding regions for GBM.

looks like BackgroundRates has a .rates property, .rate_uncertainty, tstart, tstop property which I could
use to create the interpolation

to init background rates I need:

rates, rate_uncertainty, tstart, tstop, emin, emax,
                 exposure=None

which I can store in a .npz format

TTE versus Phaii handling is a little odd. Phaii will come binned in energy but TTE will require additional binning.
What if we had a from_tte and from_phaii?


Ok, I think I'm settling on a few things:
    1. Handle TTE/Phaii preparation outside BackgroundRates. This means
       Rebinning needs to be done on the underlying TTE data before the
       background step. This should be fine. Likely saves time relative
       to the old search.
    2. have class methods for from_tte() and from_phaii() to clarify
       when we're handling each init.
    3. Have a save step that can save the rates, rate_uncertainty, tstart, tstop, emin, emax, exposure
       arrays needed to re-create the backrates objects
"""

class BackgroundFitterMatrix:
    # REPLACE WITH FITTER DataCollection

    def __init__(self):
        self.fitters = []

    def from_tte(cls, ttes, *args, **kwargs):
        f = cls()
        for tte in ttes:
            fitter = BackgroundFitter.from_tte(tte, *args, **kwargs)
            f.fitters.append(fitter)

    def from_phaii(cls, phaiis, *args, **kwargs):
        f = cls()
        for phaii in phaiis:
            fitter = BackgroundFitter.from_phaii(phaii, *args, **kwargs)
            f.fitters.append(fitter)

    def fit(self, *args, **kwargs):
        for fitter in self.fitters:
            fitter.fit(*args, **kwargs)


def naive_poisson(tte, time_range, window_width, fast=True): 
    # REPLACE WITH FITTER DataCollection
    fitter = BackgroundFitter.from_tte(tte, NaivePoisson)
    return fitter.fit(window_width=window_width, fast=fast)

def polynomial(phaii, time_range, order, fit_ranges=None):
    # REPLACE WITH FITTER DataCollection
    if fit_ranges is None:
        fit_ranges = [time_range]

    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=fit_ranges)
    return fitter.fit(order=order)

class BackgroundRatesMatrix:

    def __init__(self, fitters, time_range, resolution=0.256):

        self.time_range = time_range
        self.resolution = resolution

        self.times = 0.5 * resolution + np.arange(
                resolution * (int(time_range[0] / resolution) - 1),
                resolution * (int(time_range[1] / resolution) + 1), resolution)

        self.background_rates = None
        self.good = None

        for fitter in fitters:
            rates = fitter.interpolate_times(self.times)
            good = np.ones(rates.size)
            # append arrays
            self.background_rates.append(rates)
            self.good.append(good)

    def interpolate_rate(self, time):
        pass

    def interpolate_good(self, time):
        pass
