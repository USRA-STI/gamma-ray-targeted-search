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
import os
import sys
import glob
import time
import numpy as np
import healpy as hp
import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rich.progress import Progress, TextColumn, TaskProgressColumn, TimeRemainingColumn
from astropy.coordinates import SkyCoord, get_sun
from gdt.core.plot.sky import EquatorialPlot
from gdt.core.collection import DataCollection
from gdt.core.binning.binned import rebin_by_edge_index
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.background.unbinned import NaivePoisson
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.saa import GbmSaa
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.missions.fermi.gbm.localization import GbmHealPix
from gdt.missions.fermi.gbm.finders import ContinuousFinder, TriggerFinder

from data import FitStatus
from utils import SkyGrid, update_tte_trigtime, grid_to_healpix
from plots import TargetedLightcurves, Waterfall, plot_orbit
from skymap import O3_DGAUSS_Model, LigoHealPix
from search import TargetedSearch
from results import Results, calculate_top_snr, calculate_pe_variables, calculate_marginal_flux, calculate_coinclr
from filters import remove_pe, remove_dur_spec, downselect
from response import GbmResponse
from configuration import InstrumentConfiguration, SearchConfiguration

# treat n0-n5 and n6-nb as separate instruments
nai1 = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']
config1 = {det: {'channel_edges': [0, 8, 20, 33, 51, 85, 106, 127, 128], 'search_channels': [1, 2, 3, 4, 5, 6]} for det in nai1}
inst_config1 = InstrumentConfiguration('instrument1', config1)

nai2 = ['n6', 'n7', 'n8', 'n9', 'na', 'nb']
config2 = {det: {'channel_edges': [0, 8, 20, 33, 51, 85, 106, 127, 128], 'search_channels': [1, 2, 3, 4, 5, 6]} for det in nai2}
inst_config2 = InstrumentConfiguration('instrument2', config2)

search_config = SearchConfiguration(instruments=[inst_config1, inst_config2], skygrid_resolution=45.0)
search_config.settings.update({
    'win_width': 60,
    'min_loglr': 5,
    'min_dur': 1.024, 'max_dur': 1.024,
    'min_step': 1.024,'num_steps': 1,
    'bkgd_range': [-500, 500], 'bkgd_window': 125, 
    'data_range': [-30.512, 30.512]})

search_config.write("search.yml")

skygrid = SkyGrid(search_config['skygrid_resolution'])

search = TargetedSearch(search_config, skygrid)

trigtime = Time(524666469.429, format='fermi')

print("Preparing data...")
for inst_config in [inst_config1, inst_config2]:

    print(f"  {inst_config['name']}")
    print(f"    Opening TTE")
    tte_data = []
    for det in inst_config['detector_names']:
        path = f"data/gbm/524666469.429/glg_tte_{det}_170817_12z_v00.fit.gz"
        tte = update_tte_trigtime(GbmTte.open(path), trigtime.fermi)
        tte = tte.rebin_energy(rebin_by_edge_index, np.array(inst_config['detectors'][det]['channel_edges']))
        tte_data.append(tte)
    ttes = DataCollection.from_list(tte_data, names=inst_config['detector_names'])

    print(f"    Opening PosHist")
    poshist = GbmPosHist.open("data/gbm/524666469.429/glg_poshist_all_170817_v01.fit")
    spacecraft_frames = poshist.get_spacecraft_frame()

    print(f"    Opening Response")
    response = GbmResponse(inst_config['detector_names'], SkyGrid(5.0), 'templates/GBM',
                           spacecraft_frames, trigtime.fermi, templates=[0, 1, 2])

    print(f"    Fitting background")
    backfitters = DataCollection.from_list(
        [BackgroundFitter.from_tte(tte.slice_time(search_config['bkgd_range']), NaivePoisson) for tte in ttes],
         names=inst_config['detector_names'])
    backfitters.fit(window_width=search_config['bkgd_window'], fast=True)

    goodness_of_fit = DataCollection.from_list(
        [FitStatus(len(edges) - 1) for det, edges in inst_config['channel_edges'].items()],
        names=inst_config['detector_names'])

    search.add_instrument(inst_config['name'], ttes, backfitters, goodness_of_fit, response)

search.calculate_likelihood(1.984 - 0.256, 1.984 + 0.256, sky_mask=True)

az_max, zen_max = search.like_points[:, search.like.max_location]
print("Best-fit (az %.1f deg, zen %.1f deg) marginal llr %.2f" % (np.degrees(az_max), np.degrees(zen_max), search.like.marginal_llr))
