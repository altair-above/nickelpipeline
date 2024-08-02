

import numpy as np
import logging
from pathlib import Path
from astropy.io import ascii, fits
from astropy.wcs import WCS

from nickelpipeline.convenience.log import log_astropy_table


logger = logging.getLogger(__name__)

def convert_coords(phot_data_inpath, phot_data_outpath, astrometric_img_path):
    
    phot_data_path = Path(phot_data_inpath)
    phot_data = ascii.read(phot_data_path, format='csv')
    
    # Create a WCS object from the input FITS header
    _, header = fits.getdata(astrometric_img_path, header=True)
    wcs = WCS(header)
    
    x_coords = phot_data['x_fit']
    y_coords = phot_data['y_fit']
    # Get the array of all RA coordinates and all Dec coordinates
    world_coords = wcs.all_pix2world(x_coords, y_coords, 0)  # 0 specifies no origin offset
    print(world_coords)
    
    col_index = phot_data.colnames.index('y_fit') + 1
    phot_data.add_column(world_coords[0], name='ra', index=col_index)
    phot_data.add_column(world_coords[1], name='dec', index=col_index + 1)
    phot_data['ra'].info.format = '.3f'
    phot_data['dec'].info.format = '.3f'
    
    phot_data.write(phot_data_outpath, format='csv', overwrite=True)
    
    logger.debug(f"Source Catalog w/ Sky Coordinates: \n{log_astropy_table(phot_data)}")
    logger.info(f"Saving source catalog w/ RA/Dec coords to {phot_data_outpath}")
    return phot_data_outpath
    
    
    

