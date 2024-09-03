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
import gts
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.finders import TriggerFtp, ContinuousFtp
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.trigdat import Trigdat
from gdt.missions.fermi.time import Time
import glob

# Get the tte data
bn = '160408268'
ftp = TriggerFtp(bn)
ftp.get_tte("data/gbm", dets=['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb'])
tte_files = sorted(glob.glob(f'data/gbm/glg_tte_n?_bn{bn}_v??.fit'))

# Get the trigdat file too since it contains spacecraft position history for this burst
ftp.get_trigdat('data/gbm')
trigdat_file = glob.glob(f'data/gbm/glg_trigdat_all_bn{bn}_v??.fit')[-1]

# Load the tte data into memory
tte_data = []
for tte_file in tte_files:
    tte = GbmTte.open(tte_file)
    tte_data.append(tte)

# Bin the tte data
channel_edges = [8, 20, 33, 51, 85, 106, 127]
pha2_data = gts.preparePha2Data(tte_data, channel_edges)

# Load the detector responses
response = gts.loadResponse('templates/GBM/direct/nai.npy', templates=[0, 1, 2], channels=[1, 2, 3, 4, 5, 6])

# Get trigger time and spacecraft frames
trigdat = Trigdat.open(trigdat_file)
spacecraft_frames = trigdat.poshist
trigtime = Time(pha2_data[0].trigtime, format='fermi')

# Run the search
search = gts.runSearch(pha2_data, response, spacecraft_frames, trigtime, background_range=[-30, 30])
```
