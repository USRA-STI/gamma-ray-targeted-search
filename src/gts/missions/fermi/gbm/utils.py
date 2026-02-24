#! python
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
import glob
import numpy as np
import healpy as hp

from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.localization import GbmHealPix
from gdt.missions.fermi.gbm.finders import ContinuousFinder, TriggerFinder

from gts.core.utils import update_tte_trigtime, grid_to_healpix
from gts.core.skymap import O3_DGAUSS_Model
from gts.core.search import TargetedSearch
from gts.core.results import Results


def GetData(trigger_id, settings, data_directory, protocol='HTTPS'):
    """ Method for downloading data needed by the targeted search

    Args:
        trigger_id (str, :class:`Time`): GBM trigger ID string (burst number) for analyzing triggered data OR
                                         a Time() object for analyzing continuous data
        data_directory (str): Directory for downloaded data. Data will appear in a subfolder formatted as
                              'data/trigger_id' for triggered data and 'data/#########.###' for continuous data.
        protocol (str): Download protocol. Can be 'HTTPS' or 'FTP'. 'AWS' is unsupported.

    Returns:
        (Time, [str, str, ...], str): tuple with Time() formatted trigger time, 
                                      list of TTE file paths, and position history path
    """
    ftp = None

    # boolean for specifying requested data type (triggered or continuous)
    triggered = isinstance(trigger_id, str)

    # format file paths
    sub_dir = trigger_id if triggered else "%.3f" % trigger_id.fermi
    path = f"{data_directory}/{sub_dir}"
    tte_wildcard = f"{path}/*tte_??_*.fit*"
    poshist_wildcard = f"{path}/glg_poshist_all_*.fit"
    
    # check for files
    tte_files = []
    for det in settings['detectors']:
        tte_files.extend(glob.glob(tte_wildcard.replace("??", det)))
    poshist_files = sorted(glob.glob(poshist_wildcard))

    if len(tte_files) < len(settings['detectors']):
        finder = TriggerFinder(trigger_id, protocol=protocol) if triggered else ContinuousFinder(trigger_id, protocol=protocol)
        tte_files = [finder.get_tte(path, dets=[det])[0] for det in settings['detectors']]

    # get trigtime from first triggered TTE file when using triggered files
    if triggered:
        trigtime = Time(GbmTte.open(tte_files[0]).headers[0]['TRIGTIME'], format='fermi')
    else:
        trigtime = trigger_id # trigger_id is already a Time() object for continuous case

    # ensure we have a position history file
    if not len(poshist_files):
        finder = ContinuousFinder(trigtime, protocol=protocol)
        finder.get_poshist(path)
        poshist_files = sorted(glob.glob(poshist_wildcard))
            
    if len(tte_files) != len(settings['detectors']) or not len(poshist_files):
        raise ValueError("Could not download or locate files. Check ")

    # only return first poshist for now.
    # Need to work on crossover at day boundary.
    return trigtime, tte_files, poshist_files[0]

def GetGbmLocalization(search, result, time_ref, include_systematic=True):
    """Compute GBM location with systematic error modeled as
    a double Gaussian (core + tail) shape.

    Args:
        search (TargetedSearch): Search object
        result (Results): Result object
        time_ref (Time): Reference time for the tstart value of result
        include_systematic (bool): Include systematic error when True

    Returns:
        (GbmHealPix)
    """
    # recompute likelihood without sky masking for this timebin
    search.calculate_likelihood(result['tstart'], result['tstart'] + result['duration'], sky_mask=False)

    # compute sky probability for max template
    prob = np.exp(search.like.llr - np.max(search.like.llr))[result['template'], :]

    # project to NSIDE 64 healpix
    proj_prob, _ = grid_to_healpix(
        prob, search.like_points, search.like_frame, nside_out=64)

    # upscale to NSIDE 128
    hires_nside = 128
    hires_npix = hp.nside2npix(hires_nside)
    theta, phi = hp.pix2ang(hires_nside, np.arange(hires_npix))
    upscaled_prob = hp.get_interp_val(proj_prob, theta, phi)

    # build GbmHealpix object
    loc = GbmHealPix.from_data(upscaled_prob, trigtime=time_ref.fermi + result['tstart'],
                               quaternion=search.like_frame.quaternion, scpos=search.like_frame.obsgeoloc)

    # apply systematic error
    if include_systematic:
        systematic = (O3_DGAUSS_Model, result['in_rock'], result['zen'])
        loc = loc.convolve(*systematic)

    # remove Earth region
    loc.remove_earth()

    return loc
