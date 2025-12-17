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
from utils import SkyGrid, update_tte_trigtime
from plots import TargetedLightcurves, Waterfall, plot_orbit
from skymap import O3_DGAUSS_Model, LigoHealPix
from search import TargetedSearch
from results import Results, calculate_top_snr, calculate_pe_variables, calculate_marginal_flux
from filters import remove_pe, remove_dur_spec, downselect
from response import GbmResponse
from configuration import InstrumentConfiguration, SearchConfiguration

basedir = os.path.dirname(os.path.abspath(__file__))

def GetData(trigger_id, settings, data_directory):
    """ Method for downloading data needed by the targeted search

    Args:
        trigger_id (str, :class:`Time`): GBM trigger ID string (burst number) for analyzing triggered data OR
                                         a Time() object for analyzing continuous data
        data_directory (str): Directory for downloaded data. Data will appear in a subfolder formatted as
                              'data/trigger_id' for triggered data and 'data/#########.###' for continuous data.

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
        ftp = TriggerFinder(trigger_id) if triggered else ContinuousFinder(trigger_id)
        tte_files = ftp.get_tte(path, dets=settings['detectors'])

    # get trigtime from first triggered TTE file when using triggered files
    if triggered:
        trigtime = Time(GbmTte.open(tte_files[0]).headers[0]['TRIGTIME'], format='fermi')
    else:
        trigtime = trigger_id # trigger_id is already a Time() object for continuous case

    # ensure we have a position history file
    if not len(poshist_files):
        if ftp is None or triggered:
            ftp = ContinuousFinder(trigger_id)
        ftp.get_poshist(path)
        poshist_files = sorted(glob.glob(poshist_wildcard))
            
    if len(tte_files) != len(settings['detectors']) or not len(poshist_files):
        raise ValueError("Could not download or locate files. Check ")

    # only return first poshist for now.
    # Need to work on crossover at day boundary.
    return trigtime, tte_files, poshist_files[0]

def main():

    parser = argparse.ArgumentParser("gbm_targeted_search.py", "Script for performing the full GBM targeted search")
    parser.add_argument("-t", "--time", default=None, help="Time for continuous data search.")
    parser.add_argument("-b", "--burst-number", default=None, help="GBM burst number for on-board trigger search.")
    parser.add_argument("-f", "--format", type=str, default=None, choices=[None, 'gps', 'fermi', 'datetime'], help="Format of --trigger option.")
    parser.add_argument("-w", "--search-window-width", default=60, type=float, help="Search window around trigger time in seconds. The search will run from -width/2 until +width/2.")
    parser.add_argument("--min-dur", default=0.064, type=float, help="Minimum duration of GRB transient in seconds.")
    parser.add_argument("--max-dur", default=8.192, type=float, help="Maximum duration of GRB transient in seconds.")
    parser.add_argument("--min-step", default=0.064, type=float, help="Minimum time step size in seconds used to move duration window.")
    parser.add_argument("--num-steps", default=8, type=int, help="Sets duration window step size using duration/num_steps for steps larger than --min-step.")
    parser.add_argument("-s", "--skymap", default=None, type=str, help="Optional skymap file.")
    parser.add_argument("-o", "--results-dir", default=".", type=str, help="Directory for results output")
    parser.add_argument("--flatten", action='store_true', help="Flatten multiorder skymaps.")
    
    print("\n"  + " ".join(sys.argv) +  "\n")

    args = parser.parse_args()

    progress = Progress(TextColumn("[progress.description]{task.description}"),
                        TaskProgressColumn(), TimeRemainingColumn(elapsed_when_finished=True))

    # default behavior
    trigger = args.burst_number

    if args.time is None and args.skymap is None and args.burst_number is None:
        raise ValueError("User must provide at least --time, --skymap, or --burst-number")

    if args.format is None and args.time is not None:
        raise ValueError("User must specify time format with --format")

    if args.skymap:
        args.skymap = LigoHealPix.open(args.skymap, min_nside=128, flatten=args.flatten, prob_only=False)
        if args.time is None and args.burst_number is None:
            args.time = args.skymap.trigtime
            args.format = 'datetime'

    # apply trigger formatting for Time() object trigger types.
    # Note: setting --time will over-ride skymap time.
    if args.time:
        if args.format == 'datetime':
            value = datetime.datetime.fromisoformat(args.time)
        else:
            value = float(args.time)
        trigger = Time(value, format=args.format)

    nai_configs = {det.name: {'channel_edges': [0, 8, 20, 33, 51, 85, 106, 127, 128], 'search_channels': [1, 2, 3, 4, 5, 6]} for det in GbmDetectors.nai()}
    bgo_configs = {det.name: {'channel_edges': [0, 8, 21, 40, 65, 90, 112, 124, 128], 'search_channels': [0, 1, 2, 3, 4, 5, 6, 7]} for det in GbmDetectors.bgo()}
    gbm_config = InstrumentConfiguration('gbm', nai_configs | bgo_configs)

    search_config = SearchConfiguration(instruments=[gbm_config])
    search_config.settings.update({
         'win_width': args.search_window_width,
         'min_loglr': 5,
         'min_dur': args.min_dur, 'max_dur': args.max_dur,
         'min_step': args.min_step,'num_steps': args.num_steps,
         'bkgd_range': [-500, 500], 'bkgd_window': 125.0,
         'data_range': np.array([-0.5, 0.5]) * (args.search_window_width + args.max_dur)})

    trigtime, tte_files, poshist_file = GetData(trigger, gbm_config, "data/gbm")

    progress.start()
    task = progress.add_task("Opening TTE..." , total=len(gbm_config['detectors']))

    tte_data = []
    for i, det_config in enumerate(gbm_config['detectors'].values()):
        tte = update_tte_trigtime(GbmTte.open(tte_files[i]), trigtime.value)
        tte = tte.rebin_energy(rebin_by_edge_index, np.array(det_config['channel_edges']))
        tte_data.append(tte)
        progress.update(task, advance=1)
    ttes = DataCollection.from_list(tte_data, names=gbm_config['detector_names'])

    progress.stop()
    progress.remove_task(task)

    print("Binning TTE for search...")
    phaiis = DataCollection.from_list(
         ttes.to_phaii(bin_by_time, search_config['time_resolution'], time_ref=0, time_range=search_config['data_range']),
         names=gbm_config['detector_names'])

    print("Fitting background...")
    backfitters = DataCollection.from_list(
        [BackgroundFitter.from_tte(tte.slice_time(search_config['bkgd_range']), NaivePoisson) for tte in ttes],
        names=gbm_config['detector_names'])
    backfitters.fit(window_width=search_config['bkgd_window'], fast=True)
    
    goodness_of_fit = DataCollection.from_list(
        [FitStatus(len(edges) - 1) for det, edges in gbm_config["channel_edges"].items()],
        names=gbm_config['detector_names'])

    print("Opening poshist...")
    poshist = GbmPosHist.open(poshist_file)
    spacecraft_frames = poshist.get_spacecraft_frame()

    print("Opening the response...")
    # Get the response for hard, normal, soft spectral templates
    skygrid = SkyGrid(search_config['skygrid_resolution'])
    response = GbmResponse(phaiis.items, skygrid, 'templates/GBM', spacecraft_frames, ttes.get_item("n0").trigtime, templates=[0, 1, 2])

    print("Initializing search...")
    search = TargetedSearch(search_config, skygrid)
    search.add_instrument('gbm', phaiis, backfitters, goodness_of_fit, response)

    snr_channels = gbm_config.select_channels({det.name: [3, 4] for det in GbmDetectors.nai()})
    search.add_calculation([("snr1", "<f8"), ("snr0", "<f8")], calculate_top_snr, instrument="gbm", channels=snr_channels, n=2)

    pe_channels = gbm_config.select_channels({det.name: [0, 1] for det in GbmDetectors.nai()})
    search.add_calculation([("pe0", "<f8"), ("pe1", "<f8"), ("pe2", "<f8")], calculate_pe_variables, instrument="gbm", channels=pe_channels)

    search.add_calculation([("in_rock", "<i8")], lambda search, result: search.instrument_data['gbm'].response.in_rock)

    search.add_calculation([(f"marginal_flux{i}", "<f8") for i in range(3)] +
                           [(f"marginal_flux_sig{i}", "<f8") for i in range(3)], calculate_marginal_flux, durations=[1.024])

    timebins = search.get_timebins()
    response.preprocess(timebins)
    progress.start()
    results = search.run(timebins, progress=progress)
    progress.stop()

    # append common coordinate transformations
    frames = search.instrument_data['gbm'].response._preprocessed['frames']

    coordinate_max = SkyCoord(results['az'], 0.5 * np.pi - results['zen'], frame=frames, unit='rad')
    coordinate_sun = get_sun(Time(trigtime, format='fermi')) # TODO: use central time of bin instead of trigtime

    results.append_fields(
        ["ra", "dec", "sun_angle", "earth_angle"],
        [coordinate_max.icrs.ra.radian,
         coordinate_max.icrs.dec.radian,
         coordinate_sun.separation(coordinate_max, origin_mismatch="ignore").radian,
         frames.geocenter.separation(coordinate_max, origin_mismatch="ignore").radian]
    )
    results.append_fields(["in_gti"], [np.ones(results.size, dtype=int)])

    # filter results to produce up to 3 top candidates
    filtered_results = remove_pe(results)
    filtered_results = downselect(filtered_results, threshold=search_config['min_loglr'], no_empty=True)
    filtered_results = downselect(filtered_results, combine_spec=False, fixedwin=search_config['win_width'])
    filtered_results = remove_dur_spec(filtered_results, 8.192, 2)
    filtered_results.save(args.results_dir, 'filtered_results.npz')

    # TO DO ADD coinclr calc

    # report the results
    print('\nFound {} candidates...\n'.format(filtered_results.size))
    print('Total number of bins: {}'.format(filtered_results.size))
    print('In GTI: {}'.format(np.sum(filtered_results['in_gti'])))
    print('Used atmoscat: {}'.format(np.sum(filtered_results['in_rock'])))
    print('Pre-filtered: {}'.format(np.sum(filtered_results['like_status'] == 2)))
    print(
        "--------------------------------------------------------------------------------------------------------------------------------------------------")
    print(
        "    tcent    duration  gti rock good  az   zen   ra    dec  spec ampli snr  snr0  snr1  chisq chisq+ sun earth    logLR  coincLR   PE0   PE1   PE2")
    print(
        "--------------------------------------------------------------------------------------------------------------------------------------------------")

    keys = ['tstart', 'duration', 'in_gti', 'in_rock', 'like_status', 'az', 'zen', 'ra', 'dec', 'template', 'flux_amplitude',
            'like_snr', 'snr0', 'snr1', 'reduced_chisq', 'chiplusdof', 'sun_angle', 'earth_angle', 'loglr', 'loglr', 'pe0', 'pe1', 'pe2']

    for values in filtered_results.to_list(keys, units={key: np.degrees(1) for key in ['az', 'zen', 'ra', 'dec', 'sun_angle', 'earth_angle']}):
        values[0] = values[0] + 0.5 * values[1] # convert to tcent
        print(
            "%13.3f %7.3f %3d %4d %4d  %5.1f %5.1f %5.1f %5.1f %1d %5.2f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %8.2f %8.2f %5.1f %5.1f %5.1f" % tuple(values))

    print('\nCreating the following plots:')

    print('\nOrbital plot...')
    orbit_filename = os.path.join(args.results_dir, 'Orbit.png')
    plot_orbit(spacecraft_frames, trigtime, orbit_filename, GbmSaa())
    print('Done.')

    print('\nWaterfall plots...')
    w = Waterfall(results, trigtime)
    loglr_filename = os.path.join(args.results_dir, 'Loglr.png')
    w.plot_loglr(loglr_filename, val_min=3.0)
    loglr_spec_filename = os.path.join(args.results_dir, 'Loglr_spec.png')
    w.plot_loglr(loglr_spec_filename, val_min=3.0, spectra=True)
    print('Done.')

    print('\nLight curve plots...')
    nai = list(nai_configs.keys())
    bgo = list(bgo_configs.keys())
    time_range = search_config['search_range']
    lcplotter = TargetedLightcurves(search.instrument_data['gbm'], trigtime)
    for i in range(filtered_results.size):
        print('Light curves for Event {}.'.format(i+1))
        duration, tstart = filtered_results['duration'][i], filtered_results['tstart'][i]
        [lcplotter.plot_summed(duration, time_range=time_range, event_time=tstart, **kwargs) for kwargs in [
            {'filename': os.path.join(args.results_dir, f'Event{i}_Summed_All_NaI_Chan1-6.png'), 'detectors': nai, 'channel_range': (1, 6)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Summed_Right_NaI_Chan3-4.png'), 'detectors': nai[:6], 'channel_range': (3, 4)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Summed_Left_NaI_Chan3-4.png'), 'detectors': nai[6:], 'channel_range': (3, 4)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Summed_All_BGO_Chan0-3.png'), 'detectors': bgo, 'channel_range': (0, 3)}]]
        [lcplotter.plot_channels(duration, time_range=time_range, event_time=tstart, **kwargs) for kwargs in [
            {'filename': os.path.join(args.results_dir, f'Event{i}_Channel_All_NaI_Chan0-7.png'), 'detectors': nai, 'channel_range': (0, 7)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Channel_Right_NaI_Chan0-7.png'), 'detectors': nai[:6], 'channel_range': (0, 7)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Channel_Left_NaI_Chan0-7.png'), 'detectors': nai[6:], 'channel_range': (0, 7)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Channel_All_BGO_Chan0-3.png'), 'detectors': bgo, 'channel_range': (0, 3)}]]
        [lcplotter.plot_detectors(duration, time_range=time_range, event_time=tstart, **kwargs) for kwargs in [
            {'filename': os.path.join(args.results_dir, f'Event{i}_Detector_All_NaI_Chan1-6.png'), 'detectors': nai, 'channel_range': (1, 6)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Detector_All_NaI_Chan1-2.png'), 'detectors': nai, 'channel_range': (1, 2)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Detector_All_NaI_Chan3-4.png'), 'detectors': nai, 'channel_range': (3, 4)},
            {'filename': os.path.join(args.results_dir, f'Event{i}_Detector_All_BGO_Chan1-6.png'), 'detectors': bgo, 'channel_range': (1, 6)}]]
    print('Done.')

    print('\nLocalizations...')
    for i in range(filtered_results.size):

        # event information
        t = filtered_results.times[i]
        duration = filtered_results.durations[i]
        zen = np.array(filtered_results.locs_sc)[1][i]
        template = filtered_results.templates[i]

        # localization
        systematic = (O3_DGAUSS_Model, atmoscat, zen) 
        loc = gts.createLocalization(t, duration, template, search, GbmHealPix, systematic, remove_earth=True)
        loc.write(args.results_dir, filename='Event{}_healpix.fit'.format(i+1), overwrite=True)

        skyplot = EquatorialPlot()
        skyplot.add_localization(loc, clevels=[0.90, 0.50], gradient=False)
        plt.savefig('Event{}_skymap.png'.format(i+1), dpi=300)
        plt.clf()

        # combined localization
        if search['skymap'] is not None:
            region_prob = loc.region_probability(search['skymap']) * 100.0
            print('\t Event {0} Spatial Association: {1:3.1f}%'.format(i+1, region_prob))
            if region_prob > 50.0:
                combined = loc.multiply(loc, search['skymap'])
                # run from_data to fix _frame member. To do: fix bug in GDT
                combined = GbmHealPix.from_data(combined.prob, trigtime=loc.trigtime, scpos=loc.scpos, quaternion=loc.quaternion)
                combined.write(args.results_dir, 
                               filename='Event{}_healpix_combined.fit'.format(i+1), overwrite=True)

                skyplot = EquatorialPlot()
                skyplot.add_localization(combined, clevels=[0.9, 0.5], gradient=False)
                plt.savefig('Event{}_skymap_combined.png'.format(i+1), dpi=300)
                plt.clf()
    print('Done.')

if __name__ == "__main__":

    main()
