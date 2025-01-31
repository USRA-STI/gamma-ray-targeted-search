import glob
import numpy as np

from gdt.core.data_primitives import Gti
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.binning.binned import rebin_by_edge_index
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.finders import TriggerFtp, ContinuousFtp

class Detectors():

    def __init__():

        self.detectors = {
            
            'nai': 
            {
                'number': 12,
                'wildcard': 'glg_tte_n??_bn??.fit'
            },
            
            'bgo':
            {
                'number': 2,
                'wildcard': 'glg_tte_b??_bn??.fit'
            }
        }

    def set_mask(data_directory, files):

        for detector in self.detectors:

            # create mask
            detector_files = sorted(glob.glob(f'{data_directory}+/{self.detectors[detector]["wildcard"]}'))
            detector_mask = [True if file in detector_files else False for file in files]
            
            # save mask
            self.detectors[key]['mask'] = detector_mask

class Data():
    
    '''
    General class of data. Currently supported data type are tte data of Fermi-GBM. 
    Binned Phaii files are produced for each NaI according to binning requirements. 
    '''

    def __init__(self, trigger, data_directory='data/gbm', search_window_width=60, 
                 max_dur=8.192, resolution=0.064):

        self.trigger = trigger
        self.data_directory = data_directory
        self.search_window_width = search_window_width
        self.max_dur = max_dur
        self.resolution = resolution
        # set detectors channels
        self.channels = [0,1,2,3,4,5,6,7]
        self.channels_edges = [8, 20, 33, 51, 85, 106, 127]

    def get(self):
        
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

        trigger_id     = self.trigger
        data_directory = self.data_directory
        
        # boolean for specifying requested data type (triggered or continuous)
        triggered = isinstance(trigger_id, str)
    
        # format file paths
        sub_dir = trigger_id if triggered else "%.3f" % trigger_id.fermi
        path = f"{data_directory}/{sub_dir}"

        tte_wildcard = f"{path}/*tte*.fit*"
        poshist_wildcard = f"{path}/glg_poshist_all_*.fit"
        
        # check for files
        tte_files = sorted(glob.glob(tte_wildcard))
        poshist_files = sorted(glob.glob(poshist_wildcard))
    
        # ensure we have 14 nai TTE files (12 NaIs and 2 BGOs). 
        if len(tte_files) < 14:
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
                
        if len(tte_files) != 14 or not len(poshist_files):
            raise ValueError("Could not download or locate files. Check ")
       
        # only return first poshist for now.
        # Need to work on crossover at day boundary.
        self.trigtime     = trigtime
        self.tte_files    = tte_files
        self.poshist_file = poshist_files[0]
        
    def load(self):

        # Load the tte data into memory
        print("opening TTE")
        self.tte_data = []
        for tte_file in self.tte_files:
            tte = GbmTte.open(tte_file)
            self.tte_data.append(tte)
        
    def bin(self):
        
        print("re-binning TTE for search")
        # Convert the tte data to binned phaii data using a time range of at least +/-30 seconds
        self.time_range = np.array([-1, 1]) * max([0.5 * self.search_window_width + self.max_dur + 1.024, 30])      
        
        """ Function for preparing binned phaii data from time tagged events
    
        Args:
            tte_data (list): list of opened Tte data objects from a mission
            channel_edges (list): list of energy channel edges to use when binning data by energy index
            time_range (list): start and stop time used to select data around t0
            t0 (float or Time class): trigger time to use. Use trigtime of the Tte file when None.
            resolution (float): time resolution used when binning the Tte data in time.
                                This will set the minimum searchable duration of the search.
    
        Returns:
            list: list of PHAII data objects which represent instrument counts binned in energy as a function of time
        """

        # Download data and load them in memory
        self.get()
        self.load()
        
        # Create a list to contain the phaii product for each detector
        pha2_data = []
    
        # Make sure the channel edges is a numpy array
        channel_edges = np.array(self.channels_edges)

        t0 = self.trigtime.fermi
        resolution = self.resolution

        # Loop through each TTE file and create a phaii file with the supplied channel edges
        for tte in self.tte_data:
    
            trigtime = tte.trigtime
    
            if trigtime is None and t0 is None:
                raise ValueError("t0 time is required when using continuous TTE files")
            if t0 is not None:
                # calculate offset to new trigger time
                offset = t0 if trigtime is None else t0 - trigtime
                # apply offset to event times
                tte.data._events['TIME'] -= offset
                # apply offset to good time interval bounds
                gti_start, gti_stop = np.transpose(tte.gti.as_list()) - offset
                tte._gti = Gti.from_bounds(gti_start, gti_stop)
                # update trigtime here but set it after rebin_energy to
                # avoid header mismatch in continuous tte files
                trigtime = t0
    
            # Bin the TTE data by time and energy
            phaii = tte.to_phaii(bin_by_time, resolution, time_ref=0, time_range=self.time_range)
            phaii = phaii.rebin_energy(rebin_by_edge_index, channel_edges)
            phaii._trigtime = trigtime
    
            pha2_data.append(phaii)

        self.pha2_data = pha2_data

        return pha2_data

    def getCounts(self, timebin, channels=None):
        """ Retrieve observed counts computed over a specific time bin
    
        Args:
            timebin (tuple): tuple with (bin start time, bin duration)
            channels (list): list of channel indices to use when selecting a subset of detectors
    
        Returns:
            np.ndarray: array of counts for each detector
        """

        # Get the time bin information
        tstart = timebin[0]
        duration = timebin[1]
        
        tstop = tstart + duration
        time_range = np.array([tstart, tstop])

        # get pha2 data
        pha2_data = self.pha2_data
    
        # Determine the number of detectors and channels
        n_detectors = len(pha2_data)
    
        if channels is None:
            n_channels = len(pha2_data[0].data.chan_widths)
    
        # Create an array to contain the count data
        counts = np.zeros((n_detectors, n_channels))
    
        # Loop through each pha2 file and extract and record the number of counts in the time bin
        for index in range(len(pha2_data)):
    
            # Integrate the phaii data over time to produce a count spectrum
            phaii = pha2_data[index]
            channel_counts = phaii.to_spectrum(time_range=time_range).counts
    
            # Fill the counts array
            counts[index, :] = channel_counts[channels]
    
        return counts