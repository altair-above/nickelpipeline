

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.photometry.starfind import analyze_sources


adjust_global_logger('DEBUG', __name__)
logger = logging.getLogger(__name__)

image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1039.fits_red.fits')
analyze_sources(image, plot=True)

# for key, val in self.psf_model.param_names:#fixed.items():