
import numpy as np
from scipy.interpolate import interp1d

class BackgroundRatesMatrix:

    def __init__(self, fitters, time_range, resolution=0.256):

        self.detectors = fitters.items
        self.time_range = time_range
        self.resolution = resolution

        self.times = 0.5 * resolution + np.arange(
                resolution * (int(time_range[0] / resolution) - 1),
                resolution * (int(time_range[1] / resolution) + 1), resolution)

        self.good = []
        self.bkgd = []
        self.interpolations = []

        for fitter in fitters:
            self.bkgd.append(fitter.interpolate_times(self.times))
            self.good.append(np.ones(self.bkgd[-1].rates.shape))
            # interpolation methods
            self.interpolations.append([
                interp1d(self.times, self.bkgd[-1].rates.transpose(), fill_value='extrapolate'),
                interp1d(self.times, self.good[-1].transpose(), fill_value='extrapolate')])

    def rates(self, time):
        rates, good = [], []
        for interp in self.interpolations:
            rates.append(interp[0](time))
            good.append(interp[1](time))
        return np.ravel(rates), np.ravel(good)

    def write(self, filename, detectors=None):
        """Method to write class contents to file(s)"""
        pass

    def open(self, filename, detectors=None):
        """Method to create class from file(s)"""
        pass
