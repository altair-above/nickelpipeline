

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.photometry.psf_photometry import psf_analysis, consolidate_groups, format_table
from nickelpipeline.photometry.aperture_photometry import aperture_analysis

adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)

image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1040_red.fits')
phot_data_3 = psf_analysis(image, thresh=8.0, fittype='circular', show_final=True, show_inters=True,)

# image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_R/d1024_red.fits')
# phot_data_4 = psf_analysis(image, thresh=8.0, fittype='circular', show_final=True, show_inters=True,)

# image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_B/d1022_red.fits')
# phot_data_5 = psf_analysis(image, thresh=8.0, fittype='circular', show_final=True, show_inters=True,)

phot_data_3_consolidated = consolidate_groups(phot_data_3)
phot_data_3_consolidated = format_table(phot_data_3_consolidated)

phot_data_3_aper = aperture_analysis(phot_data_3_consolidated, image)
phot_data_3_aper = format_table(phot_data_3_aper)