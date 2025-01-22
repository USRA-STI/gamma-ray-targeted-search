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
        self.pha2_data = []
    
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
    
            self.pha2_data.append(phaii)

    def getTimeBins(self, settings):
        """ Calculate the time bins used in the search. These represent the different
        emission durations of the search shifted across the full search range using
        a given step size.
    
        Args:
            pha2_data (list): PHAII data objects for each detector
            settings (dict): search settings for duration range, search range, and step size
    
        Returns:
            list: list with values for the start times and durations of each search bin
        """

        win_width = settings['win_width']
        min_dur = settings['min_dur']
        max_dur = settings['max_dur']
        min_step = settings['min_step']
        num_steps = settings['num_steps']
    
        search_range = (-win_width/2.0, win_width/2.0)
    
        # Durations to search in powers of two
        log2maxdur = np.round(np.log2(max_dur))
        log2mindur = np.round(np.log2(min_dur))
        durations = 1.024 * 2. ** np.arange(log2mindur, log2maxdur + 1, 1)
    
        # Get one of the phaii files to determine the proper data binning
        phaii = self.pha2_data[0]
    
        # The search bins before 0
        tstart1 = phaii.data.slice_time(search_range[0] - durations.max()/2.0, 0).tstart
        timebins1 = []
        if len(tstart1):
            timebins1 = ((t, dur) for dur in durations \
                         for t in np.arange(0, tstart1[0], \
                                            -max(min_step, dur/num_steps)) \
                         if t >= search_range[0]-dur/2.0)     
    
        # The search bins after 0, inclusive
        tstart2 = phaii.data.slice_time(0, search_range[1]).tstart
        timebins2 = []
        if len(tstart2):
            timebins2 = ((t, dur) for dur in durations 
                         for t in np.arange(0, tstart2[-1], \
                                            max(min_step, dur/num_steps)) \
                         if t+dur/2.0 <= search_range[-1])        
    
        # Combine the search windows. Format: (tstart, duration)
        timebins = sorted(timebins1)
        timebins.extend(sorted(timebins2))

        # Get the center of the timebin and the bin duration
        self.tstarts = [timebin[0] for timebin in timebins]
        self.durations = [timebin[1] for timebin in timebins]
        self.tcenters = [tstart+duration/2.0 for tstart, duration in zip(self.tstarts, self.durations)]

        # save timebins to self
        self.timebins = timebins
        
        return timebins

    def getCounts(self, pha2_data, timebin, channels=None):
        """ Retrieve observed counts computed over a specific time bin
    
        Args:
            pha2_data (list): list of PHAII data objects for each detector
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
    
        # Determine the number of detectors and channels
        n_detectors = len(pha2_data)
    
        if channels is None:
            n_channels = len(pha2_data[0].data.chan_widths)
    
        # Create an array to contain the count data
        # Note: rows are detectors, columns are channels. 
        # The flattened array will ordered by detectors: 
        # [det_0_ch_0, det_0_ch1, ..., det_n_ch_0, det_n_ch_1]
        counts = np.zeros((n_detectors, n_channels))
    
        # Loop through each pha2 file and extract and record the number of counts in the time bin
        for index in range(len(pha2_data)):
    
            # Integrate the phaii data over time to produce a count spectrum
            phaii = pha2_data[index]
            channel_counts = phaii.to_spectrum(time_range=time_range).counts
    
            # Fill the counts array
            counts[index, :] = channel_counts[channels]
    
        return counts

    def fitBackgrounds(self, pha2_data, time_range=(-30, 30), verbose=True, plot=False):
        """ Method for performing a first order polynomial background fit
    
        Args:
            pha2_data (list): PHAII data objects for each detector
            time_range (tuple): tuple with start and stop time of the fit region
            verbose (bool): show fit statistic when True
            plot (bool): display a plot of the fit when True
    
        Returns:
            list: list of background rates objects returned from the fit
        """
        if verbose == True:
            print('\nFitting backgrounds...')
            print('Background fit selection: %s sec to %s sec' % (time_range[0], time_range[1]))
            print('\nStat/DOF:')
    
            print('--------------------------- Channels ---------------------------')
    
        # Create a list to contain the background rates for each detector
        background_rates = []
    
        for phaii in pha2_data:
    
            # Fit the data
            fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=[time_range])
            fitter.fit(order=1)
    
            if verbose == True:
                # Round the elements of the array
                goodness_of_fit = np.round(fitter.statistic/fitter.dof, 2)
    
                # Print the elements in a table format with consistent column spacing
                col_width = 7  # Adjust as needed for wider numbers
                formatted_strings = [f"{value:>{col_width}.2f}" for value in goodness_of_fit]
                print(" ".join(formatted_strings))
    
            # Get the closest time edge to the search window
            tstart_closest = phaii.data.closest_time_edge(time_range[0], which='low')
            tstop_closest = phaii.data.closest_time_edge(time_range[1], which='high')
    
            # Get the index of the closest values and pad that index by an additional bin
            index_start = np.abs(phaii.data.tstart - tstart_closest).argmin()
            index_stop = np.abs(phaii.data.tstop - tstop_closest).argmin()
    
            # Interpolate the fit over the search range
            tstarts = phaii.data.tstart[index_start:index_stop]
            tstops = phaii.data.tstop[index_start:index_stop]
            back_rates = fitter.interpolate_bins(tstarts, tstops)        
    
            # Save the background object
            background_rates.append(back_rates)
    
            # Plot the fit
            if plot == True:
                lightcurve = phaii.to_lightcurve()
                lcplot = Lightcurve(data=lightcurve)
                lcplot.set_background(back_rates)
                plt.xlim(*time_range)
                plt.show()
    
        background_rates = np.array(background_rates)

        return background_rates

    def getBackgrounds(self, background_rates, timebin, channels=None):
        """ Retrieve background counts computed over a specific time bin
    
        Args:
            background_rates (list): list of background rates objects for each detector
            timebin (tuple): tuple with (bin start time, bin duration)
            channels (list): list of channel indices to use when selecting a subset of detectors
    
        Returns:
            (np.ndarray, np.ndarray): tuple with arrays of background counts and their uncertainties for each detector
        """
        tstart = timebin[0]
        duration = timebin[1]
        tstop = tstart + duration
        time_range = np.array([tstart, tstart + duration])
    
        # Determine the number of detectors
        n_detectors = len(background_rates)
    
        # Determine the number of channels 
        if channels is None:
            n_channels = len(background_rates[0].chan_widths)
    
        # Create an array to contain the background data
        # Note: rows are detectors, columns are channels. 
        # The flattened array will ordered by detectors: 
        # [det_0_ch_0, det_0_ch1, ..., det_n_ch_0, det_n_ch_1]
        backgrounds = np.zeros((n_detectors, n_channels))
        background_uncertainties = np.zeros((n_detectors, n_channels))
    
        for index in range(len(background_rates)):
    
            # Produce an background object that is integrated over the entire time slice
            background_rate = background_rates[index]
            background_rate_integrated = background_rate.integrate_time(tstart=tstart, tstop=tstop)
    
            # Eztract arrays of background counts and background counts uncertainty per channel
            background = background_rate_integrated.counts
            background_uncertainty = background_rate_integrated.count_uncertainty
    
            if isinstance(background_rate.count_uncertainty[0], np.float64):
                background_uncertainty = background_uncertainty.reshape(-1,1)
    
            # Fill the background and background uncertainy arrays
            backgrounds[index, :] = background[channels]
            background_uncertainties[index, :] = background_uncertainty[channels]
    
        return backgrounds, background_uncertainties

    def formatDataForSearch(self, index):

        """ Creates data format required by likelihood functions

        Parameters:
        -----------
        index: int
            Integer for the time bin index

        Returns:
        --------
        counts: array
            Numpy array with source counts for all detectors
        bkgd_counts: array
            Numpy array with background counts for all detectors
        bkgd_var: array
            Numpy array with background variance for all detectors
        fitmask: array
            Numpy array with status of each count.
            True == include in likelihood, False == exclude
        pe_veto: tuple
            Tuple with values needed for phoshorescence veto
        snrs: tuple
            Tuple with top 2 SNR values
        """

        # Get the center of the timebin and the bin duration
        tstart   = self.tstarts[index]
        duration = self.durations[index]

        # Get the timebin
        timebin = self.timebins[index]
        
        # Extract the counts and background data from the phaii data for this specific timebin
        counts = self.getCounts(self.pha2_data, timebin)
        
        # Fit the backgrounds and return the background objects for each detector
        background_rates = self.fitBackgrounds(self.pha2_data, time_range=self.time_range)
        background, background_error = self.getBackgrounds(background_rates, timebin)
            
        # Flatten the arrays
        counts = counts.ravel()
        background = background.ravel()
        background_error = background_error.ravel()

        return counts, background, background_error