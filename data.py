import numpy as np

from gdt.core.tte import PhotonList
from gdt.core.data_primitives import EventList, Gti
from gdt.core.binning.binned import rebin_by_edge_index

def update_tte_trigtime(tte, t0):
    """Updates the trigtime for triggered and continuous TTE files.
    This is needed to ensure all times are relative to the time of
    interest, t0.

    Args:
        tte (PhotonList): time tagged event data derived from PhotonList
        t0 (float): the time of interest for the targeted search

    Returns:
        (PhotonList)
    """
    if tte.trigtime is None:
        # continuous TTE case, offset by t0
        offset = t0
    else:
        # trigger TTE case, shift data from trigtime to t0
        offset = t0 - tte.trigtime

    # event times relative to trigtime
    data = EventList(tte.data.times - offset,
                     tte.data.channels, tte.data.ebounds)

    # good time interval bounds relative to trigtime
    gti_start, gti_stop = np.transpose(tte.gti.as_list()) - offset
    gti = Gti.from_bounds(gti_start, gti_stop)

    return PhotonList.from_data(data, gti=gti, trigger_time=t0,
                                event_deadtime=tte.event_deadtime,
                                overflow_deadtime=tte.overflow_deadtime)

class PhaiiCountMatrix:
    """
    Class interface for retrieving the detector counts
    matrix from a collection of PHAII data.

    Args:
        phaiis (DataCollection): Data collection object with PHAII data for each detector
    """
    def __init__(self, phaiis):

        self.phaiis = phaiis

    @property
    def detectors(self):
        return self.phaiis.items()

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

        return np.ravel(counts), np.ravel(exposure)

    def write(self, filename, detectors=None):
        """Method to write class contents to file(s)"""
        pass

    def open(self, filename, detectors=None):
        """Method to create class from file(s)"""
        pass
