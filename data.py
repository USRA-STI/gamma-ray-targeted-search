import numpy as np

from gdt.core.data_primitives import Gti
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.binning.binned import rebin_by_edge_index

class PhaiiMatrix:
    """
    Class interface for managing a matrix of PHAII data from multiple detectors.

    Args:
        ttes (list): List with TTE data for each detector
        settings (dict): Settings dictionary defining detector bins
        time_range (list | tuple): The time range to select
        t0 (float): An external trigger time
        resolution (float): Temporal resolution of time bins
    """
    def __init__(self, ttes, settings, time_range=[-30, 30], t0=None, resolution=0.064):

        # create PHAII data from TTE data
        self.phaiis = []
        for tte in ttes:

            trigtime = tte.trigtime

            if trigtime is None and t0 is None:
                raise ValueError("t0 time is required when using continuous TTE files")
            if t0 is not None:
                # calculate offset to new trigger time
                offset = t0 if trigtime is None else t0 - trigtime
                # apply offset to event times
                tte.data._events['TIME'] -= offset
                # apply offset to good time interval bounds
                gti_start, gti_stop = np.transpose(tte.gti.as_list()) - offset
                tte._gti = Gti.from_bounds(gti_start, gti_stop)
                # update trigtime here but set it after rebin_energy to
                # avoid header mismatch in continuous tte files
                trigtime = t0

            # bin the TTE data by time
            phaii = tte.to_phaii(bin_by_time, resolution, time_ref=0, time_range=time_range)

            # re-bin PHAII energy
            channel_edges = np.array(settings['detectors'][phaii.detector]['channel_edges'])
            phaii = phaii.rebin_energy(rebin_by_edge_index, channel_edges)

            # set trigtime
            phaii._trigtime = trigtime

            self.phaiis.append(phaii)

    def counts(self, tstart, tstop):
        """Retrieve observed counts and their corresponding exposure
        computed over a specific time bin
    
        Args:
            tstart (float): start time of the counts window
            tstop (float): stop time of the counts window
    
        Returns:
            tuple: arrays of counts and exposure for each detector
        """
        counts, exposure = [], []
        for phaii in self.phaiis:
            spec = phaii.to_spectrum(time_range=(tstart, tstop))
            counts.append(spec.counts)
            exposure.append(spec.exposure[0])
            print(phaii.detector, "counts", counts[-1])
        return np.ravel(counts), np.ravel(exposure)
