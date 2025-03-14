# This workflow demonstrates a workflow which skips the step
# that bins data into a Phaii format. This should be used
# by multi-mission searches to handle the correct phasing
# of counts between different spacecraft.

# NOTE: Run "Work in Progress.ipynb" before running this script
#       to download the necessary data files.

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
bkgd_range = [-500, 500]
phaii_resolution = settings['min_dur']
channel_mask = np.ravel(
    [[channel in search_channels[det] for channel in range(len(channel_edges[det]) - 1)] for det in detectors])

##########################
# Step 2. TTE Prepartion #
##########################

from rich.progress import track
from data import update_tte_trigtime
from gdt.core.collection import DataCollection
from gdt.core.binning.binned import rebin_by_edge_index
from gdt.missions.fermi.gbm.tte import GbmTte

t0 = 524666469.44569993
tte_data = []
for det in track(detectors, description="Opening TTE files"):
    path = f"glg_tte_{det}_170817_12z_v00.fit.gz"
    tte = update_tte_trigtime(GbmTte.open(path), t0)
    tte = tte.rebin_energy(rebin_by_edge_index, channel_edges[det])
    tte_data.append(tte)

ttes = DataCollection.from_list(tte_data, names=detectors)

################################
# Step 3. CountMatrix Creation #
################################

from data import CountMatrix

# example showing CountMatrix creation direct from from TTE
data = CountMatrix(ttes)
clock0 = unix_time.time()
counts, exposure = data.counts(1.728, 2.240)
clock1 = unix_time.time()

print("\nData:")
print("  counts", counts.reshape((len(detectors), 8)))
print("  exposure", exposure)
print("  retrieved in %.6f sec" % (clock1 - clock0))

###################################
# Step 4. Unbinned Background Fit #
###################################

from background import BackgroundRatesMatrix
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.unbinned import NaivePoisson

# initialize the background fitters
clock0 = unix_time.time()
backfitters = DataCollection.from_list(
    [BackgroundFitter.from_tte(tte.slice_time(bkgd_range), NaivePoisson) for tte in ttes],
    names=detectors)
backfitters.fit(window_width=125., fast=True)

print("\nBackground Fit took %.1f sec" % (unix_time.time() - clock0))

background = BackgroundRatesMatrix(backfitters) 
bkgd_counts, bkgd_var, good = background.counts(1.728, 2.240, exposure)

print("\nBackground:")
print("  counts", bkgd_counts.reshape((len(detectors), 8)))
print("  variance", bkgd_var.reshape((len(detectors), 8)))
print("  good", good.reshape((len(detectors), 8)))

########################
# Step 5. Sanity Check #
########################

print("\nSanity Checks:")
for i, det in enumerate(detectors):
    match_det = data.detectors[i] == background.detectors[i]
    match_ebounds = data.ebounds[i].low_edges() == background.ebounds[i].low_edges() \
        and data.ebounds[i].high_edges() == background.ebounds[i].high_edges()
    print("  ", det, match_det, match_ebounds)

#########################
# Step 6. Load Response #
#########################

# Note: this section needs to be replaced by a response class

import gts
import utils

kwargs = {'templates': [0, 1, 2], 'channels': [0, 1, 2, 3, 4, 5, 6, 7]}
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
# Step 7. Likelihood Calculation #
##################################

from likelihood import Likelihood

mask = channel_mask & good
masked_counts = counts[mask]
masked_bkgd_counts = bkgd_counts[mask]
masked_bkgd_var = bkgd_var[mask]

masked_rsp = rsp[:,:,mask] # apply same detector channel mask to response

like = Likelihood(ntemplate, skyGrid.size)
like.calculate(masked_counts, masked_bkgd_counts, masked_bkgd_var, masked_rsp)

print("\nLikelihood:")
print("  Max template ", like.max_template)
print("  Marginal log-likelihood ratio ", like.marginal_llr)

