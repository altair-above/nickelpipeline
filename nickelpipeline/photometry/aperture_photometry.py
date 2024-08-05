

import numpy as np
import logging
from photutils.aperture import CircularAperture
from photutils.aperture import ApertureStats

from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.log import log_astropy_table


logger = logging.getLogger(__name__)

def aperture_analysis(phot_data, image, aper_size=8.0):
    
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
        
    phot_data.sort('group_id')

    coordinates = list(zip(phot_data['x_fit'], phot_data['y_fit']))
    apertures = CircularAperture(coordinates, r=aper_size)
    aperstats = ApertureStats(image.data, apertures, local_bkg=phot_data['local_bkg'])
    
    all_data = phot_data.copy()
    col_index = phot_data.colnames.index('flux_fit') + 1
    all_data.add_column(aperstats.sum, name='flux_aper', index=col_index)
    all_data['flux_aper'].info.format = '.3f'
    all_data['flux_aper'][all_data['flags'] % 2 == 1] = np.nan
    logger.info("Aperture photometry cannot handle masked pixels--sources with masked pixels have flux_aper = nan")
    
    all_data['flux_fit'].name = 'flux_psf'
    
    all_data.add_column(all_data['flux_psf']/all_data['flux_aper'],
                        name='ratio_flux', index=col_index+1)
    all_data['ratio_flux'].info.format = '.3f'
    logger.debug(f"PSF & Aperture Photometry Results: \n{log_astropy_table(all_data)}")
    
    return format_table(all_data)


def format_table(phot_data):
    colnames = ['group_id', 'group_size', 'flags', 'x_fit', 'y_fit', 
                'flux_psf', 'flux_aper', 'ratio_flux', 'local_bkg', 
                'x_err', 'y_err', 'airmass', 'id',
                'iter_detected', 'npixfit', 'qfit', 'cfit']
    concise_data = phot_data[colnames]
    for col in colnames:
        if isinstance(concise_data[col][0], np.float64):
            concise_data[col].info.format = '.3f'
    
    return concise_data