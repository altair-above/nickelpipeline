
import logging
from pathlib import Path
from nickelpipeline.final_calibration.combo_astro_photo import convert_coords_all#, photometric_calib_all

#################################################
####  This pipeline is currently incomplete  ####
#################################################

logger = logging.getLogger(__name__)

def final_calib_all(photo_dir, astro_dir, output_dir=None):
    
    if output_dir is None:
        ouptut_dir = astro_dir.parent.parent / 'final_calib'
    Path.mkdir(ouptut_dir, exist_ok=True)
    
    astrophot_datas = convert_coords_all(photo_dir, astro_dir, ouptut_dir)
    return astrophot_datas

    # photometric_calib_all(astrophot_dir)