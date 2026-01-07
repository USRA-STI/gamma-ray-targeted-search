# GTS

The Gamma-ray Targeted Search (GTS) is a generalized, mission agnostic, version of the Fermi-GBM Targeted Search that is built around the Gamma-ray Data Tools. 

<URL>

## Installation

To use this code, you will need:

    * python3.9 or higher
    * the dependencies included in `requirements.txt`

You can install the dependencies with pip:
```
pip3 install pip --upgrade
pip3 install -r requirements.txt
```
To run the GBM examples you will also need to download the GBM detector responses
from https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/ and untar them in the GTS directory
using
```
tar -xf templates.tar.gz
```

## Usage 

See the **GBM Example.ipynb** and **GW170817 Example.ipynb** notebooks for
quick examples of how to run the Gamma-ray Targeted Search using triggered and
continuous GBM data downloaded from https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/.

A separate command line script `gbm_targeted_search.py` is also provided
to demonstrate how the GBM search can be setup to run over any time/skymap.
To run it at a specific time do:
```
python3 gbm_targeted_search.py --time 2017-08-17T12:41:04.429126 --format datetime
```
See `python3 gbm_targeted_search.py --help` for addition details.

Example code for running these searches is included below for developers
looking to dive directly into the code. Note that you will need to
download the necessary files for GBM burst number 160408268 from
https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/ to a local folder
named `data/GBM` to run it.

```python
import glob
import numpy as np

from gdt.core.collection import DataCollection
from gdt.core.binning.binned import rebin_by_edge_index
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.missions.fermi.gbm.finders import TriggerFinder
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.trigdat import Trigdat
from gdt.missions.fermi.time import Time

from data import FitStatus
from utils import SkyGrid
from search import TargetedSearch
from response import GbmResponse
from configuration import InstrumentConfiguration, SearchConfiguration

# Setup configuration objects
nai_configs = {det.name: {'channel_edges': [0, 8, 20, 33, 51, 85, 106, 127, 128], 'search_channels': [1, 2, 3, 4, 5, 6]} for det in GbmDetectors.nai()}
gbm_config = InstrumentConfiguration('gbm', nai_configs)
search_config = SearchConfiguration(instruments=[gbm_config])

# Get the tte data
bn = '160408268'
finder = TriggerFinder(bn)
finder.get_tte("data/gbm", dets=gbm_config['detector_names'])
tte_files = sorted(glob.glob(f'data/gbm/glg_tte_n?_bn{bn}_v??.fit'))

# Get the trigdat file too since it contains spacecraft position history for this burst
finder.get_trigdat('data/gbm')
trigdat_file = glob.glob(f'data/gbm/glg_trigdat_all_bn{bn}_v??.fit')[-1]

# Load the tte data into memory
tte_data = []
for i, det_config in enumerate(gbm_config['detectors'].values()):
    tte = GbmTte.open(tte_files[i])
    tte = tte.rebin_energy(rebin_by_edge_index, np.array(det_config['channel_edges']))
    tte_data.append(tte)
ttes = DataCollection.from_list(tte_data, names=gbm_config['detector_names'])

trigtime = Time(ttes.get_item('n0').trigtime, format='fermi')

phaiis = DataCollection.from_list(
     ttes.to_phaii(bin_by_time, search_config['time_resolution'], time_ref=0, time_range=search_config['search_range']),
     names=gbm_config['detector_names'])

# Fit background
backfitters = DataCollection.from_list(
    [BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=[(-30, 30)]) for phaii in phaiis],
    names=gbm_config['detector_names'])
backfitters.fit(order=1)

# placeholder goodness-of-fit to be replaced by chi square test in the future
goodness_of_fit = DataCollection.from_list(
        [FitStatus(len(edges) - 1) for det, edges in gbm_config['channel_edges'].items()],
        names=gbm_config['detector_names'])

# Load the detector responses
trigdat = Trigdat.open(trigdat_file)
response = GbmResponse(gbm_config['detector_names'], SkyGrid(5.0),
                       'templates/GBM', trigdat.poshist, trigtime.fermi, templates=[0, 1, 2])

# Run the search
search = TargetedSearch(search_config, SkyGrid(search_config['skygrid_resolution']))
search.add_instrument('gbm', phaiis, backfitters, goodness_of_fit, response)
results = search.run(search.get_timebins())
```
