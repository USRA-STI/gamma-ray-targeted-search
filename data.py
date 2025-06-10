import numpy as np
import os


class InstrumentData:
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


    def format_data(self, instrument_config, tstart, tstop, skygrid, shape_data):
        n_templates = shape_data['n_templates']

        channel_mask = instrument_config.channel_mask
        counts, exposure = self.counts(tstart, tstop)

        bkgd_rates, bkgd_variance, good = self.background_rates(tstart, tstop, exposure)
        # TODO Stack counts, backgrounds across all Skygrid positions

        # Get full skygrid, templates response
        response, earthmask = self.load_response(tstart, tstop, skygrid)
        # What if instrument response skygrid != search skygrid?
        rsp_templates, n_skygrid, _, _ = response.shape

        # NOTE: Why get n_skygrid from response, when n_templates is static, and n_skygrid can be grabbed from
        # scanner's skygrid attribute?
        rsp = response.reshape(n_templates, n_skygrid, -1)
        rsp = rsp[:, earthmask, :]

        mask = channel_mask & good

        return {
            'counts': counts[mask],
            'background_rates': bkgd_rates[mask],
            'background_variance': bkgd_variance[mask],
            'response': rsp[:, :, mask]
        }


    def format_data_by_reference(self, instrument_config, tstart, tstop, reference_frame, skygrid, shape_data):
        n_templates = shape_data['n_templates']
        num_sky_positions = shape_data['num_sky_positions']
        n_energybins = shape_data['n_energybins']

        n_detectors = len(instrument_config.detectors)
        skygrid_counts = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
        skygrid_background = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
        skygrid_background_variance = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)
        response = np.zeros(n_templates, num_sky_positions, n_energybins, n_detectors)

        for i, skypos in enumerate(skygrid._points.T):
            offset = self.get_timebin_offset(reference_frame, skypos)
            counts, exposure = self.counts(tstart + offset, tstop + offset)
            bkgd_rates, bkgd_variance = self.background_rates(tstart + offset, tstop + offset, exposure)
            # This should return a matrix for each template, energy bin, and detector given a specific skypos
            skypos_response = self.load_skypos_response(tstart, tstop, skypos, reference_frame)
            # TODO reproject outputs to match reference

            # TODO Assign all values to the skygrid matrix representation

        return {
            'counts': skygrid_counts,
            'background_rates': skygrid_background,
            'background_variance': skygrid_background_variance,
            'response': response
        }
