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
import glob
import numpy as np
import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gts
import utils
import plots

from skymap import O3_DGAUSS_Model, LigoHealPix

from gdt.core.plot.sky import EquatorialPlot
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.saa import GbmSaa
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.finders import TriggerFtp, ContinuousFtp
from gdt.missions.fermi.gbm.localization import GbmHealPix

basedir = os.path.dirname(os.path.abspath(__file__))

def getData(trigger_id, data_directory):
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
    # boolean for specifying requested data type (triggered or continuous)
    triggered = isinstance(trigger_id, str)

    # format file paths
    sub_dir = trigger_id if triggered else "%.3f" % trigger_id.fermi
    path = f"{data_directory}/{sub_dir}"
    tte_wildcard = f"{path}/*tte_n?_*.fit*"
    poshist_wildcard = f"{path}/glg_poshist_all_*.fit"
    
    # check for files
    tte_files = sorted(glob.glob(tte_wildcard))
    poshist_files = sorted(glob.glob(poshist_wildcard))

    # ensure we have 12 nai TTE files. We'll add BGO in the future.
    if len(tte_files) < 12:
        ftp = TriggerFtp(trigger_id) if triggered else ContinuousFtp(trigger_id)
        ftp.get_tte(path)
        tte_files = sorted(glob.glob(tte_wildcard))

    # get trigtime from first triggered TTE file when using triggered files
    if triggered:
        trigtime = Time(GbmTte.open(tte_files[0]).headers[0]['TRIGTIME'], format='fermi')
    else:
        trigtime = trigger_id # trigger_id is already a Time() object for continuous case

    # ensure we have a position history file
    if not len(poshist_files):
        if triggered:
            # need to update ftp object because poshist are from continuous file set
            ftp = ContinuousFtp(trigtime)
        ftp.get_poshist(path)
        poshist_files = sorted(glob.glob(poshist_wildcard))
            
    if len(tte_files) != 12 or not len(poshist_files):
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
    
    args = parser.parse_args()

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

    # get trigtime and files for the search
    trigtime, tte_files, poshist_file = getData(trigger, "data/gbm")

    # Load the tte data into memory
    print("opening TTE")
    tte_data = []
    for tte_file in tte_files:
        tte = GbmTte.open(tte_file)
        tte_data.append(tte)

    print("re-binning TTE for search")
    # Convert the tte data to binned phaii data using a time range of at least +/-30 seconds
    time_range = np.array([-1, 1]) * max([0.5 * args.search_window_width + args.max_dur + 1.024, 30])
    channel_edges = [8, 20, 33, 51, 85, 106, 127]
    pha2_data = gts.preparePha2Data(tte_data, channel_edges, t0=trigtime.fermi, time_range=time_range)

    print("opening poshist")
    # Get the spacecraft frame
    poshist = GbmPosHist.open(poshist_file)
    spacecraft_frames = poshist.get_spacecraft_frame()

    # Get the response for hard, normal, soft spectral templates
    print("opening the response files")
    kwargs = {'templates': [0, 1, 2], 'channels': [1, 2, 3, 4, 5, 6]}
    direct_path = os.path.join(basedir, 'templates/GBM/direct/nai.npy')
    response = gts.loadResponse(direct_path, **kwargs)

    atmoscat = 0
    az, zen = utils.getGeoCoordinates(spacecraft_frames.at(trigtime), unit='deg')[:2]
    if 125.0 < zen and zen < 135.0:
        # add atmospheric scattering component
        atmoscat = 1
        allowed_az = np.arange(0, 361, 5)
        closest = allowed_az[np.fabs(allowed_az - az).argmin()] % 360
        atmo_path = os.path.join(basedir, f'templates/GBM/atmo_nai/atmrates_az{closest}_zen130.npy')
        response += gts.loadResponse(atmo_path, **kwargs)

    print("running the search")
    # Run the search
    settings = {
        'win_width': args.search_window_width,
        'min_loglr': 5,
        'min_dur': args.min_dur, 'max_dur': args.max_dur,
        'min_step': args.min_step,'num_steps': args.num_steps,
    }
    search = gts.runSearch(pha2_data, response, spacecraft_frames, t0=trigtime,
                           background_range=time_range, skymap=args.skymap, settings=settings,
                           results_dir=args.results_dir)

    # filter results to produce up to 3 top candidates
    filtered_results = search['results'].remove_pe()
    filtered_results = filtered_results.downselect(threshold=settings['min_loglr'], no_empty=True)
    filtered_results = filtered_results.downselect(combine_spec=False, fixedwin=settings['win_width'])
    filtered_results.remove_dur_spec(8.192, 'soft')
    filtered_results.save(args.results_dir, 'filtered_results.npz')

    # report the results
    print('\nFound the following {} candidates:'.format(filtered_results.size))
    filtered_results.write()
    print('')

    print('\nCreating the following plots:')

    print('\nOrbital plot...')
    orbit_filename = os.path.join(args.results_dir, 'Orbit.png')
    plots.plot_orbit(spacecraft_frames, trigtime, orbit_filename, GbmSaa())
    print('Done.')

    print('\nWaterfall plots...')
    w = plots.Waterfall(search['results'], trigtime)
    loglr_filename = os.path.join(args.results_dir, 'Loglr.png')
    w.plot_loglr(loglr_filename, val_min=3.0)
    loglr_spec_filename = os.path.join(args.results_dir, 'Loglr_spec.png')
    w.plot_loglr(loglr_spec_filename, val_min=3.0, spectra=True)
    print('Done.')

    print('\nLight curve plots...')
    lcplotter = plots.TargetedLightcurves(search['data'], search['background'], trigtime)
    lc_detectors_filename = os.path.join(args.results_dir, 'Event{}_lightcurve_detectors.png')
    lc_summed_filename = os.path.join(args.results_dir, 'Event{}_lightcurve_summed.png')
    lc_channel_filename = os.path.join(args.results_dir, 'Event{}_lightcurve_channels.png')
    for i in range(filtered_results.size):
        print('Light curves for Event {}.'.format(i+1))
        duration = filtered_results.durations[i]
        event_time = filtered_results.times[i] - 0.5 * duration
        lcplotter.plot_detectors(duration, lc_detectors_filename.format(i+1), event_time=event_time)
        lcplotter.plot_channels(duration, lc_channel_filename.format(i+1), event_time=event_time)
        lcplotter.plot_summed(duration, lc_summed_filename.format(i+1), event_time=event_time)
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
