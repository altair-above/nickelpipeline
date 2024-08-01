

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.photometry.psf_photometry import analyze_sources, consolidate_groups, format_table
from nickelpipeline.photometry.aperture_photometry import aperture_analysis

adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)

image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1040_red.fits')
phot_data_3 = analyze_sources(image, plot=True, thresh=8.0, fittype='circular')

# image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_R/d1024_red.fits')
# phot_data_4 = analyze_sources(image, plot=True, thresh=8.0, fittype='circular')

# image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_B/d1022_red.fits')
# phot_data_5 = analyze_sources(image, plot=True, thresh=8.0, fittype='circular')

phot_data_3_consolidated = consolidate_groups(phot_data_3)
phot_data_3_consolidated = format_table(phot_data_3_consolidated)

phot_data_3_aper = aperture_analysis(phot_data_3_consolidated, image)
phot_data_3_aper = format_table(phot_data_3_aper)