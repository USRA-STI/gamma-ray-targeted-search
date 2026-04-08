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
import os

from pathlib import Path
from importlib.resources import files

__version__ = "2.0.1"

_gts_data = files('gts.data')

suite_path = Path(__file__).parent.parent

if 'GTS_BASE' in os.environ:
    base_path = Path(os.environ['GTS_BASE'])
else:
    base_path = Path.home().joinpath('.gammaray_targeted_search', __version__)

cache_path = base_path.joinpath('cache')

if 'GTS_DATA' in os.environ:
    data_path = Path(os.environ['GTS_DATA'])
else:
    data_path = base_path.joinpath('data')

data_path.mkdir(parents=True, exist_ok=True)

if 'GTS_TEMPLATES' in os.environ:
    templates_path = Path(os.environ['GTS_TEMPLATES'])
else:
    templates_path = base_path.joinpath('templates')

templates_path.mkdir(parents=True, exist_ok=True)
