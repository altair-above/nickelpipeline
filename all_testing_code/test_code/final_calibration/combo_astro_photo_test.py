

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.convenience.dir_nav import unzip_directories
from nickelpipeline.photometry.psf_photometry import psf_analysis, consolidate_groups
from nickelpipeline.photometry.aperture_photometry import aperture_analysis, format_table
from nickelpipeline.final_calibration.combo_astro_photo import convert_coords

adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)


def final_calib_all(photo_dir, astro_dir, final_calib_dir=None):
    
    
    phot_datas = {phot_data.name.split('_')[0]: phot_data for phot_data in photo_dir.iterdir()}
    astro_calibs = {astro_calib.name.split('_')[0]: astro_calib for astro_calib in astro_dir.iterdir()}
    astro_calibs = {astro_calib.name.split('.')[0].split('_')[0]: astro_calib for astro_calib in astro_dir.iterdir() if 'corr' not in str(astro_calib)}

    if final_calib_dir is None:
        final_calib_dir = photo_dir.parent / 'final_calib'
    Path.mkdir(final_calib_dir, exist_ok=True)
    astrophot_dir = final_calib_dir / 'astrophotsrcs'
    Path.mkdir(astrophot_dir, exist_ok=True)
    logger.info(f"Saving photometric source catalogs with sky coordinates (RA/Dec) to {astrophot_dir}")

    astrophot_datas = {}
    for key, phot_data in phot_datas.items():
        output_path = astrophot_dir / (key + '_astrophotsrcs.csv')
        try:
            print(astro_calibs[key])
            astrophot_datas[key] = convert_coords(phot_data, output_path, astro_calibs[key]) 
        except KeyError:
            logger.warning(f"No astrometric solution found for image {key}; skipping")
        


photo_dir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/photometric')
astro_dir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/astrometric')

final_calib_all(photo_dir, astro_dir)



# for phot_data in phot_datas:
    
    

# image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1040_red.fits')
# phot_data_3 = psf_analysis(image, thresh=8.0, fittype='circular', show_final=True, show_inters=True,)

# # image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_R/d1024_red.fits')
# # phot_data_4 = psf_analysis(image, thresh=8.0, fittype='circular', show_final=True, show_inters=True,)

# # image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/109_199_B/d1022_red.fits')
# # phot_data_5 = psf_analysis(image, thresh=8.0, fittype='circular', show_final=True, show_inters=True,)

# phot_data_3_consolidated = consolidate_groups(phot_data_3)
# phot_data_3_consolidated = format_table(phot_data_3_consolidated)

# phot_data_3_aper = aperture_analysis(phot_data_3_consolidated, image)
# phot_data_3_aper = format_table(phot_data_3_aper)


