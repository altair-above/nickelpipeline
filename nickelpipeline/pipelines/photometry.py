from pathlib import Path
import logging

from nickelpipeline.photometry.psf_photometry import psf_analysis, consolidate_groups
from nickelpipeline.photometry.aperture_photometry import aperture_analysis


logger = logging.getLogger(__name__)
    
def photometry_all(reddir, output_dir=None, thresh=8.0, group=False, mode='all',
                   fittype='circ', plot_final=False, plot_inters=False):
    
    logger.debug(f"Extracting images from {reddir}")
    reddir = Path(reddir)
    obj_dirs = [dir for dir in reddir.iterdir() if dir.is_dir()]
    if len(obj_dirs) == 0:
        obj_dirs = [reddir]

    if output_dir is None:
        output_dir = obj_dirs[0].parent.parent / 'photometric'
    unconsol_dir = output_dir / 'unconsolidated'
    consol_dir = output_dir / 'consolidated'
    Path.mkdir(output_dir, exist_ok=True)
    Path.mkdir(unconsol_dir, exist_ok=True)
    Path.mkdir(consol_dir, exist_ok=True)
    
    source_catalog_paths = []
    for obj_dir in obj_dirs:
        if group:
            output_dir = consol_dir / obj_dir.name
        else:
            output_dir = unconsol_dir / obj_dir.name
        Path.mkdir(output_dir, exist_ok=True)
            
        for file in obj_dir.iterdir():
            psf_data = psf_analysis(file, thresh=thresh, mode=mode, 
                                    fittype=fittype, plot_final=plot_final, 
                                    plot_inters=plot_inters)
            
            filestem = file.stem.split('_')[0]
            if group:
                psf_data = consolidate_groups(psf_data)
                output_file = output_dir / f'{filestem}_photsrcs_consol.csv'
            else:
                output_file = output_dir / f'{filestem}_photsrcs.csv'
            
            all_data = aperture_analysis(psf_data, file)
            all_data.write(output_file, format='csv', overwrite=True)
            source_catalog_paths.append(output_file)
    
    return source_catalog_paths