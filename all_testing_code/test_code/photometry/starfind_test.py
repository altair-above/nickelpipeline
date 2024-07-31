

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.photometry.starfind import analyze_sources


adjust_global_logger('DEBUG', __name__)
logger = logging.getLogger(__name__)

# test_img = Path('test_img1.fits')
# phot_data_1 = analyze_sources(test_img, plot=True)

# test_img = Path('test_img2.fits')
# phot_data_2 = analyze_sources(test_img, plot=True)

image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1040.fits_red.fits')
phot_data_3 = analyze_sources(image, plot=True)



        # phot_tbl = self.psfphot(data, mask=mask, error=error,
        #                         init_params=init_params)
        # # phot_tbl = phot_tbl[(phot_tbl['flags'] & 2) == 0]
        # self.fit_results.append(deepcopy(self.psfphot))
        # if phot_tbl is None:
        #     return None

        # phot_tbl['iter_detected'] = 1
        # iter_detected = np.ones(len(phot_tbl), dtype=int)

        # iter_num = 2
        # while iter_num <= self.maxiters and phot_tbl is not None:
        #     if iter_num == 2:
        #         residual_data = self.psfphot.make_residual_image(
        #             data, self.sub_shape)
        #     else:
        #         residual_data = self.psfphot.make_residual_image(
        #             residual_data, self.sub_shape)

        #     # do not warn if no sources are found beyond the first iteration
        #     with warnings.catch_warnings():
        #         warnings.simplefilter('ignore', NoDetectionsWarning)

        #         new_sources = self.psfphot.finder(residual_data, mask=mask)
        #         if new_sources is None:  # no new sources detected
        #             break

        #     xcol = self.psfphot._init_colnames['x']
        #     ycol = self.psfphot._init_colnames['y']
        #     new_sources = new_sources['xcentroid', 'ycentroid']
        #     new_sources.rename_column('xcentroid', xcol)
        #     new_sources.rename_column('ycentroid', ycol)
        #     iter_det = np.ones(len(new_sources), dtype=int) * iter_num
        #     iter_detected = np.concatenate((iter_detected, iter_det))

        #     if self.mode == 'all':
        #         # measure initial fluxes for the new sources from the
        #         # residual data
        #         flux = self.psfphot._get_aper_fluxes(residual_data, mask,
        #                                              new_sources)
        #         unit = getattr(data, 'unit', None)
        #         if unit is not None:
        #             flux <<= unit
        #         fluxcol = self.psfphot._init_colnames['flux']
        #         new_sources[fluxcol] = flux

        #         # combine source tables and re-fit on the original data
        #         orig_sources = phot_tbl['x_fit', 'y_fit', 'flux_fit']
        #         orig_sources.rename_column('x_fit', xcol)
        #         orig_sources.rename_column('y_fit', ycol)
        #         orig_sources.rename_column('flux_fit', fluxcol)
        #         init_params = vstack([orig_sources, new_sources])

        #         residual_data = data
        #     elif self.mode == 'new':
        #         # fit new sources on the residual data
        #         init_params = new_sources

        #     # print(new_sources[new_sources['y_init'] < 0])
        #     # print(init_params[init_params['y_init'] < 0])
        #     # init_params = init_params[(init_params['flags'] & 4) == 0]
        #     new_tbl = self.psfphot(residual_data, mask=mask, error=error,
        #                            init_params=init_params)
        #     self.psfphot.finder_results = new_sources
        #     self.fit_results.append(deepcopy(self.psfphot))

        #     if self.mode == 'all':
        #         new_tbl['iter_detected'] = iter_detected
        #         phot_tbl = new_tbl

        #     elif self.mode == 'new':
        #         # combine tables
        #         new_tbl['iter_detected'] = iter_num
        #         new_tbl['id'] += np.max(phot_tbl['id'])
        #         new_tbl['group_id'] += np.max(phot_tbl['group_id'])
        #         new_tbl.meta = {}  # prevent merge conflicts
        #         phot_tbl = vstack([phot_tbl, new_tbl])
        #     # mask = (phot_tbl['flags'] & 2) == 0
        #     # phot_tbl = phot_tbl[mask]
        #     # iter_detected = iter_detected[mask]