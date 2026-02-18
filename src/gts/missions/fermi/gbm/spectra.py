# Copyright 2017-2022 by Universities Space Research Association (USRA). All rights reserved.
#
# Developed by: William Cleveland, Adam Goldstein, and Suman Bala
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
# Very closely based on the gamma-ray burst targeted search (gbuts).
# Written by:
#               Lindy Blackburn
#               Center for Astrophysics (CfA) | Harvard & Smithsonian
#               https://github.com/lindyblackburn/gbuts
#
# Included in the generalized targeted search (gts) with permission from Lindy.
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

from gdt.core.spectra.functions import Band, Comptonized, BlackBody

# Soft Spectral Template describing lower 1/3rd of GBM GRBs
soft = Band()
soft.default_values = [
      1.0, # Amplitude
     70.0, # Epeak [keV]
     -1.9, # alpha
     -3.7, # beta
    100.0, # Epiv [keV]
]

# Normal Spectral Template describing middle 1/3rd of GBM GRBs
norm = Band()
norm.default_values = [
      1.0, # Amplitude
    230.0, # Epeak [keV]
     -1.0, # alpha
     -2.3, # beta
    100.0, # Epiv [keV]
]

# Hard Spectral Template describing upper 1/3rd of GBM GRBs
hard = Comptonized()
hard.default_values = [
       1.0, # Amplitude
    1500.0, # Epeak [keV]
      -0.5, # index
     100.0, # Epiv [keV]
]

# Blackbody spectral template
blackbody = BlackBody()
blackbody.default_values = [
     1.0, # Amplitude
    10.0, # Temperature (kT) [keV]
]
