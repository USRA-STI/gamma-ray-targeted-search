import numpy as np

from gdt.core.tte import PhotonList
from gdt.core.collection import DataCollection
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


class CountMatrix:
    """Class for retrieving detector counts matrix.

    Parameters:
        data (DataCollection): a data collection object
    """
    def __init__(self, data: DataCollection):
        """Constructor"""
        # note: need to validate that data are TTE or Phaii
        self.data = data

    @property
    def detectors(self):
        """(list): list of detector names"""
        return self.data.items

    @property
    def ebounds(self):
        """(list): list of ebounds object for each detector"""
        # convenience method for validating energy binning against background fitters
        return self.data.ebounds()

    def counts(self, tstart: float, tstop: float):
        """Retrieve observed counts and corresponding exposure
        within the interval (tstop, tstart)

        Args:
            tstart (float): interval start time in seconds
            tstop (float): interval stop time in seconds

        Returns:
            (np.array, np.array): counts and exposure arrays
        """
        counts, exposure = [], []
        for spec in self.data.to_spectrum(time_range=(tstart, tstop)):
            counts.append(spec.counts)
            # note: take the first exposure element since they're all the same
            exposure.append(spec.exposure[0])

        return np.ravel(counts), np.ravel(exposure)
