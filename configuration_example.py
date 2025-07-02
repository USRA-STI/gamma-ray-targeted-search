# Copyright 2017-2025 by Universities Space Research Association (USRA). All rights reserved.
#
# Developed by: William Cleveland, Adam Goldstein, and Alex Goberna
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
import numpy as np
import time as unix_time

from rich.progress import track

from configuration import InstrumentConfiguration, SearchConfiguration
from response import GBMResponseGenerator
from search import TargetedScanner
from results import Results
from utils import SkyGrid

import gts
import utils
import plots

from gdt.core.collection import DataCollection
from gdt.core.binning.binned import rebin_by_edge_index
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.background.unbinned import NaivePoisson
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.detectors import GbmDetectors



nai_edges = [0, 8, 20, 33, 51, 85, 106, 127, 128]
bgo_edges = [0, 8, 21, 40, 65, 90, 112, 124, 128]

nai_configs = {det.name: {'channel_edges': nai_edges.copy(), 'search_channels': [1, 2, 3, 4, 5, 6]} for det in GbmDetectors.nai()}
bgo_configs = {det.name: {'channel_edges': bgo_edges.copy(), 'search_channels': [0, 1, 2, 3, 4, 5, 6, 7]} for det in GbmDetectors.bgo()}

det_configs = nai_configs | bgo_configs
gbm_config = InstrumentConfiguration('gbm', det_configs)

t0 = 524666469.44569993

search_config = SearchConfiguration(win_width=10, instruments=[gbm_config])

time_range = search_config.time_range

skygrid = utils.SkyGrid(search_config['skygrid_resolution'])

########################################################################################################################
####################################### Getting the data (user responsibility) #########################################

# Load GBM detector and background data, and instrument's spacecraft_frames
channel_edges = gbm_config['channel_edges']
tte_data = []
for det in track(gbm_config['detector_names'], description="Opening TTE files"):
    path = f"data/gbm/524666469.429/glg_tte_{det}_170817_12z_v00.fit.gz"
    tte = utils.update_tte_trigtime(GbmTte.open(path), t0)
    tte = tte.rebin_energy(rebin_by_edge_index, np.array(channel_edges[det]))
    tte_data.append(tte)

ttes = DataCollection.from_list(tte_data, names=gbm_config['detector_names'])

phaii_resolution = search_config['min_dur']
clock0 = unix_time.time()
phaii_list = ttes.to_phaii(bin_by_time, phaii_resolution, time_ref=0, time_range=time_range)
phaiis = DataCollection.from_list(phaii_list, names=gbm_config['detector_names'])
print("\nPhaii binning took %.1f sec" % (unix_time.time() - clock0))

bkgd_fit_ranges = [-30, 30]
clock0 = unix_time.time()
backfitters = DataCollection.from_list(
    [BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=[bkgd_fit_ranges]) for phaii in phaiis],
    names=gbm_config['detector_names'])
backfitters.fit(order=1)
print("\nBackground Fit took %.1f sec" % (unix_time.time() - clock0))

#bkgd_range = [-30, 30]
#backfitters = DataCollection.from_list(
#    [BackgroundFitter.from_tte(tte.slice_time(bkgd_range), NaivePoisson) for tte in ttes],
#    names=gbm_config['detector_names'])
#backfitters.fit(window_width=125., fast=True)


poshist = GbmPosHist.open("data/gbm/524666469.429/glg_poshist_all_170817_v01.fit")


##### NOTE: Somewhere in this section, handle time conversion to a coordinated time across instruments


########################################################################################################################
####################################### Creating the search components #################################################

spacecraft_frames = poshist.get_spacecraft_frame()

response = GBMResponseGenerator(phaiis.items, skygrid, spacecraft_frames, t0, 'templates/GBM')

def goodness_of_fit(counts, background_rates):
    return np.ones_like(counts[-1], dtype=bool)

backup_fitters = []

scanner = TargetedScanner(search_config, skygrid)
scanner.add_instrument('gbm', phaiis, backfitters, response, spacecraft_frames, goodness_of_fit, backup_fitters)

result_inputs = scanner.run_search(t0)
results = Results.create(len(result_inputs), template_names=["soft", "norm", "hard"])
for i, result in enumerate(result_inputs):
    results.data[i] = result
results.save(".", "results.npz")

opened_results = Results.open("results.npz")
opened_results.data.sort(order='duration')
for entry in opened_results.data:
    print(entry)


# results.data = result_inputs

# filtered_results = results.remove_pe()
# filtered_results = filtered_results.downselect(threshold=search_config['min_loglr'], no_empty=True)
# filtered_results = filtered_results.downselect(combine_spec=False, fixedwin=search_config['win_width'])
# filtered_results.remove_dur_spec(8.192, 'soft')
#
# print('\nFound the following {} candidates:'.format(filtered_results.size))
# filtered_results.write()
# print('')
#
# print(filtered_results.durations)
# print(filtered_results.templates)
# print(filtered_results._data.shape)
#
#
# w = plots.Waterfall(results, t0)
# w.plot_loglr(val_min=3.0)
