

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.convenience.dir_nav import unzip_directories
from nickelpipeline.final_calibration.combo_astro_photo import convert_coords_all, photometric_calib_all

adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)


def final_calib_all(photo_dir, astro_dir, final_calib_dir=None):
    
    if final_calib_dir is None:
        final_calib_dir = astro_dir.parent.parent / 'final_calib'
    Path.mkdir(final_calib_dir, exist_ok=True)
    
    astrophot_dir = convert_coords_all(photo_dir, astro_dir)
    
    photometric_calib_all(astrophot_dir)
    
    


photo_dir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/photometric/consolidated')
astro_dir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/astrometric/astroimg')

# convert_coords_all(photo_dir, astro_dir, final_calib_dir = astro_dir.parent.parent / 'final_calib')

astrophot_dir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/final_calib/astrophotsrcs_consol')
photometric_calib_all(astrophot_dir)



# from nickelpipeline.convenience.fits_class import Fits_Simple
# from pathlib import Path

# astro_dir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/astrometric/astroimg')

# image99 = astro_dir / 'd1099_astro.fits'
# image99 = Fits_Simple(image99)
# print(f"{image99}: {image99.airmass}")

# image22 = astro_dir / 'd1022_astro.fits'
# image22 = Fits_Simple(image22)
# print(f"{image22}: {image22.airmass}")

# image24 = astro_dir / 'd1024_astro.fits'
# image24 = Fits_Simple(image24)
# print(f"{image24}: {image24.airmass}")

# image100 = astro_dir / 'd1100_astro.fits'
# image100 = Fits_Simple(image100)
# print(f"{image100}: {image100.airmass}")



# z = 24.9286 and k = -0.180152
# z = 27.0702 and k = -0.154549


# Fit all points for one star
# Fit all points for other star
# Compare z and k and maybe do all stars together?
# keep filters separate
