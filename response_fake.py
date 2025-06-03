import os

import utils
import gts

import numpy as np

from gdt.missions.fermi.time import Time


def swap_cols(rsp):
    """ swaps old response matrix format for the new column ordering """
    ntemplate, nsky, nene, ndet = rsp.shape
    swapped = np.zeros((ntemplate, nsky, ndet, nene), dtype=rsp.dtype)
    for det in range(ndet):
        for ene in range(nene):
            swapped[:,:,det,ene] = rsp[:,:,ene,det]
    return swapped


class FakeResponseGenerator():
    """Class for the npy results files

        Attributes:
        -----------

        Public Methods:
        ---------------

        Class Methods:
        ---------------
        """
    def __init__(self, frames, t0, skygrid):
        self.frames = frames
        self.t0 = t0
        self.skygrid = skygrid


    def load_response(self, tstart, tstop):
        direct_path = os.path.join(os.getcwd(), 'templates/GBM/direct/nai.npy')

        kwargs = {'templates': [0, 1, 2], 'channels': [0, 1, 2, 3, 4, 5, 6, 7]}

        nai_response = gts.loadResponse(direct_path, **kwargs)
        bgo_response = gts.loadResponse('templates/GBM/direct/bgo.npy', **kwargs)

        spacecraft_time = (tstart + tstop) / 2 + self.t0

        az, zen, rad = utils.getGeoCoordinates(self.frames.at(Time(spacecraft_time, format='fermi')), unit='deg')
        if 125.0 < zen and zen < 135.0:
            # add atmospheric scattering component
            allowed_az = np.arange(0, 361, 5)
            closest = allowed_az[np.fabs(allowed_az - az).argmin()] % 360
            atmo_path = os.path.join(os.getcwd(), f'templates/GBM/atmo_nai/atmrates_az{closest}_zen130.npy')
            nai_response += gts.loadResponse(atmo_path, **kwargs)
            bgo_response += gts.loadResponse(f'templates/GBM/atmo_bgo/atmrates_az{closest}_zen130.npy', **kwargs)

        az, zen, rad = np.radians(az), np.radians(zen), np.radians(rad)
        response = np.concatenate((nai_response, bgo_response), axis=-1)

        earthmask = utils.createEarthMask(self.skygrid._points, az, zen, rad)

        return swap_cols(response), earthmask