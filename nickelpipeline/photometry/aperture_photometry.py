

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
    all_data.add_column(aperstats.sum, name='flux_aper')
    all_data['flux_aper'].info.format = '.3f'
    all_data['flux_aper'][all_data['flags'] % 2 == 1] = np.nan
    logger.info("Aperture photometry cannot handle masked pixels--sources with masked pixels have flux_aper = nan")
    
    all_data['flux_fit'].name = 'flux_psf'
    
    all_data.add_column(all_data['flux_psf']/all_data['flux_aper'],
                        name='ratio_flux')
    all_data['ratio_flux'].info.format = '.3f'
    logger.debug(f"PSF & Aperture Photometry Results: \n{log_astropy_table(all_data)}")
    
    return all_data
