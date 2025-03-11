# This workflow begins directly with Phaii data,
# demonstrating how the search can be done without TTE data.
# Background rates are estimated with a first order
# polynomial fit.

import time as unix_time

####################
# Step 1. SETTINGS #
####################

import numpy as np

from gdt.missions.fermi.gbm.detectors import GbmDetectors

nai_edges = np.array([0, 8, 20, 33, 51, 85, 106, 127, 128])
bgo_edges = np.array([0, 8, 21, 40, 65, 90, 112, 124, 128])

settings = {
    'win_width': 60,
    'min_loglr': 5,
    'min_dur': 0.064, 'max_dur': 8.192,
    'min_step': 0.064,'num_steps': 8,
    'detectors':
        #{det: {'channel_edges': nai_edges, 'search_channels': [1, 2, 3, 4, 5, 6]} for det in ["n0"]}
        {det.name: {'channel_edges': nai_edges, 'search_channels': [1, 2, 3, 4, 5, 6]} for det in GbmDetectors.nai()} |
        {det.name: {'channel_edges': bgo_edges, 'search_channels': [0, 1, 2, 3, 4, 5, 6, 7]} for det in GbmDetectors.bgo()},
}

# these items could be properties / mapping methods for a settings class
detectors = list(settings['detectors'].keys())
channel_edges = {det: settings['detectors'][det]['channel_edges'] for det in detectors}
search_channels = {det: settings['detectors'][det]['search_channels'] for det in detectors}
time_range = np.array([-1, 1]) * max([0.5 * settings['win_width'] + settings['max_dur'] + 1.024, 30])
bkgd_range = [-40, 40]
phaii_resolution = settings['min_dur']
channel_mask = np.ravel(
    [[channel in search_channels[det] for channel in range(len(channel_edges[det]) - 1)] for det in detectors])

############################
# Step 2. Open Phaii files #
############################

# start by checking if we need to download files
import glob

paths = glob.glob("glg_ctime_*_bn170817529_v00.pha")

if len(paths) < len(detectors):
    from gdt.missions.fermi.gbm.finders import TriggerFinder

    finder = TriggerFinder("170817529")
    finder.get_ctime(".")

# open files
from rich.progress import track
from gdt.core.collection import DataCollection
from gdt.missions.fermi.gbm.phaii import GbmPhaii
from data import CountMatrix

phaii_data = []
for det in track(detectors, description="Opening Phaii files"):
    path = f"glg_ctime_{det}_bn170817529_v00.pha"
    phaii_data.append(GbmPhaii.open(path))

phaiis = DataCollection.from_list(phaii_data, names=detectors)

data = CountMatrix(phaiis)
clock0 = unix_time.time()
counts, exposure = data.counts(-0.256, 0.256) # relative to GRB time, not GW
clock1 = unix_time.time()

print("\nData:")
print("  counts", counts.reshape((len(detectors), 8)))
print("  exposure", exposure)
print("  retrieved in %.6f sec" % (clock1 - clock0))

#####################################
# Step 3. Polynomial Background Fit #
#####################################

from background import BackgroundRatesMatrix
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial

# initialize the background fitters
clock0 = unix_time.time()
backfitters = DataCollection.from_list(
    [BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=[bkgd_range]) for phaii in phaiis],
    names=detectors)
backfitters.fit(order=1)

print("\nBackground Fit took %.1f sec" % (unix_time.time() - clock0))

background = BackgroundRatesMatrix(backfitters) 
bkgd_counts, bkgd_var, good = background.counts(-0.256, 0.256, exposure) # relative to GRB time, not GW

print("\nBackground:")
print("  counts", bkgd_counts.reshape((len(detectors), 8)))
print("  variance", bkgd_var.reshape((len(detectors), 8)))
print("  good", good.reshape((len(detectors), 8)))

########################
# Step 4. Sanity Check #
########################

print("\nSanity Checks:")
for i, det in enumerate(detectors):
    match_det = data.detectors[i] == background.detectors[i]
    match_ebounds = data.ebounds[i].low_edges() == background.ebounds[i].low_edges() \
        and data.ebounds[i].high_edges() == background.ebounds[i].high_edges()
    print("  ", det, match_det, match_ebounds)

#########################
# Step 5. Load Response #
#########################

# Note: this section needs to be replaced by a response class

import gts
import utils

kwargs = {'templates': [0, 1, 2], 'channels': [1, 2, 3, 4, 5, 6]}
nai_response = gts.loadResponse('templates/GBM/direct/nai.npy', **kwargs)
nai_response += gts.loadResponse('templates/GBM/atmo_nai/atmrates_az140_zen130.npy', **kwargs)

kwargs = {'templates': [0, 1, 2], 'channels': [0, 1, 2, 3, 4, 5, 6, 7]}
bgo_response = gts.loadResponse('templates/GBM/direct/bgo.npy', **kwargs)
bgo_response += gts.loadResponse('templates/GBM/atmo_bgo/atmrates_az140_zen130.npy', **kwargs)

def swap_cols(rsp):
    """ swaps old response matrix format for the new column ordering """
    ntemplate, nsky, nene, ndet = rsp.shape
    swapped = np.zeros((ntemplate, nsky, ndet, nene), dtype=rsp.dtype)
    for det in range(ndet):
        for ene in range(nene):
            swapped[:,:,det,ene] = rsp[:,:,ene,det]
    return swapped

nai_response = swap_cols(nai_response)
bgo_response = swap_cols(bgo_response)

ntemplate, nsky, _, _ = nai_response.shape
rsp = np.concatenate((nai_response.reshape(ntemplate,nsky,-1),
                      bgo_response.reshape(ntemplate,nsky,-1)), axis=2)

skyGrid = utils.SkyGrid(5)
geo_azimuth = np.radians(140.38715981849026)
geo_zenith = np.radians(129.98693857220206)
geo_radius = np.radians(67.34705159615056)
earthmask = utils.createEarthMask(skyGrid._points, geo_azimuth, geo_zenith, geo_radius)
masked_rsp = rsp[:,earthmask,:]

##################################
# Step 6. Likelihood Calculation #
##################################

from likelihood import Likelihood

mask = channel_mask & good
masked_counts = counts[mask]
masked_bkgd_counts = bkgd_counts[mask]
masked_bkgd_var = bkgd_var[mask]

like = Likelihood(ntemplate, skyGrid.size)
like.calculate(masked_counts, masked_bkgd_counts, masked_bkgd_var, masked_rsp)

print("\nLikelihood:")
print("  Max template ", like.max_template)
print("  Marginal log-likelihood ratio ", like.marginal_llr)

