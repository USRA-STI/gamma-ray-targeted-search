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
import argparse

from gts.core.skymap import LigoHealPix


class Parser(argparse.ArgumentParser):
    """Parser class for core arguments shared between most analysis scripts


    Attributes:
        protocols (list[str]): List of available download protocols

    Public Methods:
        __init__: Constructor defining core arguments
        parse_args: Parses argument with additional sanity checks
    """
    protocols = ['HTTPS', 'FTP']

    def __init__(self, *args, **kwargs):
        """Constructor"""
        super().__init__(*args, **kwargs)

        self.add_argument('-t', '--time', default=None, help="Time for continuous data search.")
        self.add_argument('-d', '--dir', type=str, default=".", help="Base directory with templates folder.")
        self.add_argument('-b', '--burst-number', default=None, help="Burst ID number.")
        self.add_argument('-f', '--format', type=str, default=None, choices=[None, 'gps', 'fermi', 'datetime'], help="Format of --time option.")
        self.add_argument('-w', '--search-window-width', default=60, type=float, help="Search window around trigger time in seconds. The search will run from -width/2 until +width/2.")
        self.add_argument('--min-dur', default=0.064, type=float, help="Minimum duration of GRB transient in seconds.")
        self.add_argument('--max-dur', default=8.192, type=float, help="Maximum duration of GRB transient in seconds.")
        self.add_argument('--min-step', default=0.064, type=float, help="Minimum time step size in seconds used to move duration window.")
        self.add_argument('--num-steps', default=8, type=int, help="Sets duration window step size using duration/num_steps for steps larger than --min-step.")
        self.add_argument('-T', '--templates', default=["hard", "norm", "soft"], type=str, nargs="+", help="Spectral templates for search.")
        self.add_argument('-s', '--skymap', default=None, type=str, help="Optional skymap file.")
        self.add_argument('-o', '--results-dir', default='.', type=str, help="Directory for results output.")
        self.add_argument('-p', '--protocol', default='HTTPS', type=str, choices=self.protocols, help="Download Protocol.")
        self.add_argument('-R', '--data-range', default=[-500, 500], nargs=2, type=float, help="Data range.")
        self.add_argument('-r', '--background-range', default=[-500, 500], nargs="+", type=float, help="Background fit range(s).")
        self.add_argument('-x', '--background-window', default=125.0, type=float, help="NaivePossion background window.")
        self.add_argument('-y', '--background-poly', default=None, type=int, help="Polynomial background order.")
        self.add_argument('-z', '--background-robo', action='store_true', help="RoboLowess background.")
        self.add_argument('--min-nside', default=128, type=int, help="Minimum nside for skymaps.")
        self.add_argument('--flatten', action='store_true', help="Flatten multiorder skymaps.")
        self.add_argument('--prob-only', action='store_true', help="Only use probability column from GW skymaps.")

    def parse_args(self, *args, **kwargs):
        """Argument parser

        Args:
            See argparse.ArgumentParser

        Returns:
            (argparse.Namespace)
        """
        args = super().parse_args(*args, **kwargs)

        if args.time is None and args.skymap is None and args.burst_number is None:
            raise ValueError("User must provide at least --time, --skymap, or --burst-number")

        if args.format is None and args.time is not None:
            raise ValueError("User must specify time format with --format")

        if args.background_poly is None and len(args.background_range) != 2:
            raise ValueError("User must provide two values to --background-range for NaivePoisson fit")

        if args.background_poly is not None and len(args.background_range) % 2 != 0:
            raise ValueError("User must provide an even number of values to --background-range for Polynomial fit")

        if args.skymap:
            args.skymap = LigoHealPix.open(args.skymap, min_nside=args.min_nside, flatten=args.flatten, prob_only=args.prob_only)
            if args.time is None and args.burst_number is None:
                args.time = args.skymap.trigtime
                args.format = 'datetime'

        if args.background_poly:
            # reformat as separate fit intervals for the background polynomial
            args.background_range = [
                (args.background_range[i], args.background_range[i+1]) for i in range(0, len(args.background_range), 2)]

        return args
