import os
import numpy as np
from abc import ABC, abstractmethod


from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from utils import getGeoCoordinates, createEarthMask

det_index = {'n0': 0, 'n1': 1, 'n2': 2, 'n3':3, 'n4': 4, 'n5': 5, 'n6': 6, 'n7': 7, 'n8': 8, 'n9': 9, 'na': 10, 'nb': 11, 'b0': 0, 'b1': 1}


def swap_cols(rsp):
    """ swaps old response matrix format for the new column ordering """
    ntemplate, nsky, nene, ndet = rsp.shape
    swapped = np.zeros((ntemplate, nsky, ndet, nene), dtype=rsp.dtype)
    for det in range(ndet):
        for ene in range(nene):
            swapped[:,:,det,ene] = rsp[:,:,ene,det]
    return swapped


def getGbmDetectorType(detector):
    
    if GbmDetectors.from_str(detector).is_nai():
        det_type = 'nai'
    elif GbmDetectors.from_str(detector).is_bgo():
        det_type = 'bgo'
    else:
        raise ValueError(f'Detector {detector} not recognized.')

    return det_type

class ResponseGenerator:

    def __init__(self, detectors, skygrid, spacecraft_frames, t0, templates_directory):

        self.detectors = detectors
        self.skygrid = skygrid
        self.spacecraft_frames = spacecraft_frames
        self.t0 = t0
        self.templates_directory = templates_directory

    def get(self, tstart, tstop, earthmask=False):

        # center of the time interval
        tcenter = 0.5*(tstart+tstop) + self.t0
        tcenter = Time(tcenter, format='fermi')

        # get geocenter coordinates
        spacecraft_frame = self.spacecraft_frames.at(tcenter)
        geo_azimuth, geo_zenith, geo_radius = getGeoCoordinates(spacecraft_frame)
            
        # dictionary to store the responses
        responses = {}

        # loop over the detector types and get the cumulative response accounting for the atmospheric scattering
        for detector in self.detectors:
            responses[detector] = DirectTemplates(self.templates_directory, detector).get() + AtmosphericTemplates(self.templates_directory, detector).get(geo_azimuth, geo_zenith)
            
        # get number of templates and skygrid points
        ntemplate, nsky, _, = responses.get(self.detectors[0]).shape

        # concatenate responses
        response = np.concatenate([value.reshape(ntemplate, nsky, -1) for value in responses.values()], axis=2)

        # filter out the response grid points that are Earth-occulted
        if earthmask:
            earthmask = createEarthMask(self.skygrid._points, geo_azimuth, geo_zenith, geo_radius)
            response = response[:, earthmask, :]

        return response
    
class FullSkyTemplates():

    def __init__(self):
        self.response = None

    @property
    def num_templates(self):
        return self.response.shape[0]

    @property
    def num_skygrid(self):
        return self.response.shape[1]

    @property
    def num_channnels(self):
        return self.response.shape[2]

    @property
    def num_detectors(self):
        return self.response.shape[3]

    def get(self):
        return self.response

class DirectTemplates(FullSkyTemplates):

    def __init__(self, templates_directory, detector):

        super().__init__()

        # load response for a specific detector
        template = os.path.join(templates_directory, 'direct', getGbmDetectorType(detector)+ '.npy')
        self.response = np.load(template)[:, :, :, det_index[detector]]

class AtmosphericTemplates(FullSkyTemplates):
    
    zen_margin = 5.0
    rocking_zen = 130.0
    
    def __init__(self, templates_directory, detector):

        super().__init__()

        self.path = os.path.join(templates_directory, 'atmo_' + getGbmDetectorType(detector))
        self.data = np.load(os.path.join(self.path, 'atmrates_az0_zen130.npy'))[:, :, :, det_index[detector]]
        self.azimuths = self.get_azimuths()
        
    def get(self, geo_az, geo_zen):
        
        # if not near the rocking angle, return zeros
        if np.abs(geo_zen-self.rocking_zen) > self.zen_margin:
            data = np.zeros_like(self.data)
        else:       
            # get the responses, interpolated in geo_az
            data = self.get_weighted(geo_az)

        return data
    
    def get_azimuths(self):
        
        # get the available azimuths
        files = os.listdir(self.path)
        az = [float(file.split('_')[1][2:]) for file in files]
        
        return np.array(sorted(az))
    
    def get_weighted(self, az):
        
        # the response to the atmospheric scattering is weighted with an azimuth-based interpolatiion
        
        # get the closest two responses in azimuth
        if az > 355:
            azimuths = np.array([355.0, 360.0])
        else:
            azimuths = self.azimuths
        idx = np.argsort(np.abs(az-azimuths))
        azimuths = azimuths[idx[:2]]
        azimuths[azimuths == 360.0] = 0.0
        
        # load the two responses for a specific detector
        templates = [os.path.join(self.path, 
                     'atmrates_az{}_zen130.npy'.format(int(a))) for a in azimuths]
        responses = [np.load(template)[:, :, :, det_index[detector]] for template in templates]
        
        # weight the responses to produce an interpolated response
        width = np.abs(azimuths[0]-azimuths[1])
        dtheta = np.abs(az-azimuths)
        weights = 1.0-(dtheta/width)
        weighted_response = responses[0]*weights[0] + responses[1]*weights[1]
        
        return weighted_response


class BaseResponseGenerator(ABC):
    def __init__(self, detectors, skygrid, spacecraft_frames):
        self.detectors = detectors
        self.skygrid = skygrid
        self.spacecraft_frames = spacecraft_frames


    @abstractmethod
    def load_response(self, tstart, tstop):
        pass


class GBMResponseGenerator(BaseResponseGenerator):
    zen_margin = 5.0
    rocking_zen = 130.0

    def __init__(self, detectors, skygrid, spacecraft_frames, t0, templates_directory):
        super().__init__(detectors, skygrid, spacecraft_frames)
        self.t0 = t0
        self.templates_directory = templates_directory

    def load_response(self, tstart, tstop):
        tcenter = 0.5 * (tstart + tstop) + self.t0
        tcenter = Time(tcenter, format='fermi')
        spacecraft_frame = self.spacecraft_frames.at(tcenter)
        geo_azimuth, geo_zenith, geo_radius = getGeoCoordinates(spacecraft_frame)

        responses = []

        for detector in self.detectors:
            direct = self.load_direct_template(detector)
            atmo = self.get_atmospheric_response(detector, geo_azimuth, geo_zenith)
            responses.append(direct + atmo)

        response = np.stack(responses, axis=2)

        earthmask = createEarthMask(self.skygrid._points, geo_azimuth, geo_zenith, geo_radius)

        # TODO This needs to be changed
        response = response[0:3, :, :, :]

        return response, earthmask

    def load_direct_template(self, detector):
        template_file = os.path.join(self.templates_directory, 'direct', f"{getGbmDetectorType(detector)}.npy")
        data = swap_cols(np.load(template_file))
        return data[:, :, det_index[detector]]

    def get_atmospheric_response(self, detector, geo_az, geo_zen):
        if np.abs(geo_zen - self.rocking_zen) > self.zen_margin:
            template_file = os.path.join(self.templates_directory, 'atmo_' + getGbmDetectorType(detector), 'atmrates_az0_zen130.npy')
            original_data = swap_cols(np.load(template_file))
            zero_shape = original_data[:, :, det_index[detector]].shape
            return np.zeros(zero_shape)

        azimuths = self.get_available_azimuths(detector)
        azimuths = np.array([355.0, 360.0]) if geo_az > 355 else azimuths

        idx = np.argsort(np.abs(geo_az - azimuths))[:2]
        nearest_az = azimuths[idx]
        nearest_az[nearest_az == 360.0] = 0.0

        responses = swap_cols([self.load_atmo_template(detector, a) for a in nearest_az])
        width = np.abs(nearest_az[0] - nearest_az[1])
        dtheta = np.abs(geo_az - nearest_az)
        weights = 1.0 - (dtheta / width)
        weighted_response = responses[0] * weights[0] + responses[1] * weights[1]

        return weighted_response

    def get_available_azimuths(self, detector):
        path = os.path.join(self.templates_directory, 'atmo_' + getGbmDetectorType(detector))
        files = os.listdir(path)
        azimuths = [float(f.split('_')[1][2:]) for f in files if f.startswith('atmrates_az')]
        return np.array(sorted(azimuths))

    def load_atmo_template(self, detector, azimuth):
        file_path = os.path.join(self.templates_directory, 'atmo_' + getGbmDetectorType(detector), f'atmrates_az{int(azimuth)}_zen130.npy')
        data = swap_cols(np.load(file_path))
        return data[:, :, det_index[detector]]