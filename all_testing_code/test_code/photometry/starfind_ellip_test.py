

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.photometry.starfind_ellip import analyze_sources


adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)

image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1040_red.fits')
phot_data_3 = analyze_sources(image, plot=True, thresh=8.0)

image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_R/d1024_red.fits')
phot_data_4 = analyze_sources(image, plot=True, thresh=8.0)

image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_B/d1022.fits_red.fits')
phot_data_5 = analyze_sources(image, plot=True, thresh=8.0)

