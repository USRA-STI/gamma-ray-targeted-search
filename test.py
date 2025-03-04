
import numpy as np

from rich.progress import track
from gdt.core.collection import DataCollection
from gdt.core.binning.binned import rebin_by_edge_index
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.missions.fermi.gbm.tte import GbmTte

from data import update_tte_trigtime, PhaiiCountMatrix
from background import BackgroundInterpMatrix

nai_edges = np.array([0, 8, 20, 33, 51, 85, 106, 127, 128])
bgo_edges = np.array([0, 8, 21, 40, 65, 90, 112, 124, 128])

settings = {
    'win_width': 60,
    'min_loglr': 5,
    'min_dur': 0.064, 'max_dur': 8.192,
    'min_step': 0.064,'num_steps': 8,
    'detectors':
        {det.name: {'channel_edges': nai_edges, 'search_channels': [1, 2, 3, 4, 5, 6]} for det in GbmDetectors.nai()} |
        {det.name: {'channel_edges': bgo_edges, 'search_channels': [0, 1, 2, 3, 4, 5, 6, 7]} for det in GbmDetectors.bgo()},
}

# these items could be properties / mapping methods for a settings class
detectors = list(settings['detectors'].keys())
channel_edges = {det: settings['detectors'][det]['channel_edges'] for det in detectors}
time_range = np.array([-1, 1]) * max([0.5 * settings['win_width'] + settings['max_dur'] + 1.024, 30])
bkgd_range = [-40, 40]
phaii_resolution = settings['min_dur']

t0 = 524666469.446
tte_data = []
for det in track(detectors, description="Opening TTE files"):
    path = f"data/gbm/524666469.429/glg_tte_{det}_170817_12z_v00.fit.gz"
    tte = update_tte_trigtime(GbmTte.open(path), t0)
    tte = tte.rebin_energy(rebin_by_edge_index, channel_edges[det])
    tte_data.append(tte)

ttes = DataCollection.from_list(tte_data, names=detectors)

print("binning")
from gdt.core.binning.unbinned import bin_by_time

phaiis = DataCollection.from_list(
    ttes.to_phaii(bin_by_time, phaii_resolution, time_ref=0, time_range=time_range),
    names=detectors)

from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial

# initialize the background fitters
print("fitting background")
bkgd_range = [(-40, 40)]
backfitters = DataCollection.from_list(
    [BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_range) for phaii in phaiis],
    names=detectors)
backfitters.fit(order=1)


data = PhaiiCountMatrix(phaiis)
counts, exposure = data.counts(0, 1.024)
print("counts", counts)
print("exposure", exposure)

background = BackgroundInterpMatrix(backfitters, time_range=bkgd_range[0]) 
rates, good = background.rates(0.512)
print("rates", rates)
print("good", good)
print(background.detectors)
