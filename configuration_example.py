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

from configuration import DetectorConfiguration, InstrumentConfiguration, SearchConfiguration
from gdt.missions.fermi.gbm.detectors import GbmDetectors

nai_edges = [0, 8, 20, 33, 51, 85, 106, 127, 128]
bgo_edges = [0, 8, 21, 40, 65, 90, 112, 124, 128]

nai_configs = [DetectorConfiguration(nai_edges.copy(), [1, 2, 3, 4, 5, 6]) for det in GbmDetectors.nai()]
bgo_configs = [DetectorConfiguration(bgo_edges.copy(), [0, 1, 2, 3, 4, 5, 6, 7]) for det in GbmDetectors.bgo()]
det_configs = nai_configs + bgo_configs
dets = GbmDetectors.nai() + GbmDetectors.bgo()
det_names = [det.name for det in dets]

gbm_config = InstrumentConfiguration(det_names, det_configs)
search_settings = SearchConfiguration.build_search_settings()
search_config = SearchConfiguration(search_settings, 'gbm', ['gbm'], [gbm_config])

search_config.save('test.yaml')
new_config = SearchConfiguration.open('test.yaml')
print(new_config.reference_instrument)
print(new_config.instruments)
print(new_config.time_range)
