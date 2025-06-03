from data import CountMatrix
from background import BackgroundRatesMatrix
# from response import ResponseGenerator
import numpy as np
import os




class InstrumentData:
    """Class for storing necessary data components for targeted search, for a single instrument.
    """
    def __init__(self, counter, background_counter, response_generator, spacecraft_frames):
        """Constructor"""
        self.counter = counter
        self.background_counter = background_counter
        self.response_generator = response_generator
        self.spacecraft_frames = spacecraft_frames


    # For reference instrument, counts will be computed as is from CountMatrix. For additional instruments, an offset
    # to the time bin will be required relative to the reference instrument, for each target sky position we compute
    #
    # Options: Calculate offset within this function
    #          Calculate offset externally, application of offset handled by higher-level search utility/function
    def counts(self, tstart: float, tstop: float, reference_frame=None):
        return self.counter.counts(tstart, tstop)

    # For reference instrument, background will be computed as is from BackgroundRatesMatrix. Otherwise, an offset
    # to the time bin will be required relative to the reference instrument, for each target sky position we compute
    #
    # Options: Calculate offset within this function
    #          Calculate offset externally, application of offset handled by higher-level search utility/function
    def background_rates(self, tstart: float, tstop: float, exposure, reference_frame=None):
        return self.background_counter.counts(tstart, tstop, exposure)


    # response[i, :, :, :] -> Response array for a specific template
    # response[:, i, :, :] -> Response array for a specific sky position
    # response[:, :, i, :] -> Response array for a specific energy bin
    # response[:, :, :, i] -> Response array for a specific detector
    # When combining instrument data, stack response matrix across the 3rd axis to add new detector responses from
    # additional instrument. Projection from additional instrument to reference instrument skygrid will be required
    #
    # Will the response need to be generated per skypos, given that for each skypos, the relative path of the energy
    # waves to the target instrument is different and would use a different spacecraft_frame?
    def load_response(self, tstart, tstop, skygrid, earthmask=False):
        # Note: Can this function take just the spacecraft frame itself rather than calculating it
        return self.response_generator.load_response(tstart, tstop)


    def load_skypos_response(self, tstart, tstop, target_skypos, reference_frame, earthmask=False):
        # Note: Can this function take just the spacecraft frame itself rather than calculating it
        pass


    def get_spacecraft_frame(self, time):
        frame_index = np.abs(self.spacecraft_frames.obstime.value - time).argmin()
        spacecraft_frame = self.spacecraft_frames[frame_index]

        return spacecraft_frame


    def get_timebin_offset(self, reference_frame, target_skypos):
        # Calculate offset based on target sky pos, reference_frame, finding the frame in this instance's frames that
        # would correspond to when the energy beam would reach this instrument
        # Return a float representing the timebin offset, along with the spacecraft frame associated with it.
        pass


class FullInstrumentData:
    """Class for storing necessary data components for targeted search, for a single instrument."""

    # Backup fitters should be an array of DataCollections of BackFitters to be used in case we find the background fit
    # is not suitable
    def __init__(self, data, fitters, response_generator, spacecraft_frames, goodness_of_fit, backup_fitters):
        # Sanity checks
        background_bounds = [fitter._data_obj.ebounds for fitter in fitters]
        for i, det in enumerate(data.items):
            match_det = data.items[i] == fitters.items[i] == response_generator.detectors[i]
            match_ebounds = data.ebounds()[i].low_edges() == background_bounds[i].low_edges() \
                            and data.ebounds()[i].high_edges() == background_bounds[i].high_edges()
            if not match_det:
                print("Warning: Detectors are not in order across input data, backfitters, or response generator."
                      "Please check to ensure correct ordering.")
            if not match_ebounds:
                print("Warning: Energy bounds do not match across input data and backfitters. Please check to ensure "
                      "inputs are valid.")

        self.data = data
        self.fitters = fitters
        self.response_generator = response_generator
        self.spacecraft_frames = spacecraft_frames
        self.goodness_of_fit = goodness_of_fit
        self.backup_fitters = backup_fitters


    @property
    def detectors(self):
        return self.data.items


    @property
    def ebounds(self):
        return self.data.ebounds()


    def counts(self, tstart, tstop):
        counts, exposure = [], []
        for spec in self.data.to_spectrum(time_range=(tstart, tstop)):
            counts.append(spec.counts)
            exposure.append(spec.exposure[0])

        return np.ravel(counts), np.ravel(exposure)


    def background_rates(self, tstart, tstop, exposure):
        tstart = np.atleast_1d(tstart)
        tstop = np.atleast_1d(tstop)

        counts, counts_var, good = [], [], []
        for i, fitter in enumerate(self.fitters):
            rates, rate_uncert = fitter._method.interpolate(tstart, tstop)
            counts.append(rates[0] * exposure[i])
            counts_var.append(0.5 * (rate_uncert[0] * exposure[i]) ** 2)
            # TODO Replace with correct TTE/PHAII counts for goodness of fit.
            good.append(self.goodness_of_fit(counts, rates))

        return np.ravel(counts), np.ravel(counts_var), np.ravel(good)


    def load_response(self, tstart, tstop, skygrid, earthmask=False):
        return self.response_generator.load_response(tstart, tstop)


    def load_skypos_response(self, tstart, tstop, target_skypos, reference_frame, earthmask=False):
        # Note: Can this function take just the spacecraft frame itself rather than calculating it
        pass


    def get_spacecraft_frame(self, time):
        frame_index = np.abs(self.spacecraft_frames.obstime.value - time).argmin()
        spacecraft_frame = self.spacecraft_frames[frame_index]

        return spacecraft_frame


    def get_timebin_offset(self, reference_frame, target_skypos):
        # Calculate offset based on target sky pos, reference_frame, finding the frame in this instance's frames that
        # would correspond to when the energy beam would reach this instrument
        # Return a float representing the timebin offset, along with the spacecraft frame associated with it.
        pass
