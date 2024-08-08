from nickelpipeline.pipelines.reduction import reduce_all
from nickelpipeline.pipelines.astrometry import astrometry_all
from nickelpipeline.pipelines.photometry import photometry_all
from nickelpipeline.pipelines.final_calib import final_calib_all
from nickelpipeline.convenience.display_fits import display_many_nickel

import logging
from nickelpipeline.convenience.log import adjust_global_logger
adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)

api_key = "fknqiifdlhjliedf"

datadir = 'test_data_example'
rawdir = 'test_data_example/raw'
reddir = 'test_data_example/reduced'

# red_files = reduce_all(rawdir=rawdir, save_inters=True)

# display_many_nickel(red_files)

# astro_calib_files = astrometry_all(reddir, api_key)

src_catalog_paths = photometry_all(reddir + '/UGC_9837_V', group=True, plot_final=True,
                                   plot_inters=False, thresh=5.0)

photodir = datadir + '/photometric/unconsolidated'#src_catalog_paths[0].parent.parent
astrodir = datadir + '/astrometric/astroimg'#astro_calib_files[0].parent
astrophot_data_tables = final_calib_all(photodir, astrodir)

from nickelpipeline.convenience.graphs import plot_sources
for object, src_table_dict in astrophot_data_tables.items():
    logger.info(object)
    for file_key, src_table in src_table_dict.items():
        src_table.meta['image_path'] = reddir + '/' + object + '/' + file_key + '_red.fits'
        plot_sources(src_table, given_fwhm=8.0, flux_name='flux_psf', scale=1.5)