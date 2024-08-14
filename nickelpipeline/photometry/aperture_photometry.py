import numpy as np
import logging
from photutils.aperture import CircularAperture
from photutils.aperture import ApertureStats

from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.log import log_astropy_table

logger = logging.getLogger(__name__)

def aperture_analysis(phot_data, image, aper_size=8.0):
    """
    Perform aperture photometry on an image, based on a PSF photometry
    source catalog.

    This function computes the flux within circular apertures centered 
    at source coordinates, and calculates the ratio of PSF flux to aperture flux.

    Parameters
    ----------
    phot_data : `astropy.table.Table`
        Table of photometric data, including source coordinates and local background.
    image : `Fits_Simple` or str
        Image from which to perform aperture photometry.
    aper_size : float, optional
        Radius of the circular aperture in pixels. Default is 8.0.

    Returns
    -------
    `astropy.table.Table`
        Table containing updated photometric data, including aperture flux and flux ratios.
    """
    # Ensure image is a Fits_Simple object
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)

    # Sort photometric data by group_id
    phot_data.sort('group_id')

    # Create list of source coordinates for aperture analysis
    coordinates = list(zip(phot_data['x_fit'], phot_data['y_fit']))
    apertures = CircularAperture(coordinates, r=aper_size)
    
    # Calculate aperture statistics including sum within apertures
    aperstats = ApertureStats(image.data, apertures, local_bkg=phot_data['local_bkg'])

    # Copy photometric data for modifications
    all_data = phot_data.copy()
    all_data['flux_fit'].name = 'flux_psf'
    col_index = all_data.colnames.index('flux_psf') + 1
    
    # Add the flux within apertures to the photometric data
    all_data.add_column(aperstats.sum, name='flux_aper', index=col_index)
    all_data['flux_aper'].info.format = '.3f'
    
    # Mark sources with masked pixels as NaN in the aperture flux
    all_data['flux_aper'][all_data['flags'] % 2 == 1] = np.nan
    logger.info("Aperture photometry cannot handle masked pixels--sources with masked pixels have flux_aper = nan")    
    
    # Calculate the ratio of PSF flux to aperture flux and add it to the table
    all_data.add_column(all_data['flux_psf'] / all_data['flux_aper'],
                        name='ratio_flux', index=col_index + 1)
    all_data['ratio_flux'].info.format = '.3f'
    
    # Log the results of the photometry
    logger.debug(f"PSF & Aperture Photometry Results: \n{log_astropy_table(all_data)}")

    return format_table(all_data)


def format_table(phot_data):
    """
    Format the photometric data table to include only relevant columns and set formatting.

    Parameters
    ----------
    phot_data : `astropy.table.Table`
        Table containing photometric data to be formatted.

    Returns
    -------
    `astropy.table.Table`
        Formatted table containing a concise selection of columns.
    """
    # Specify the desired column order and names
    colnames = ['group_id', 'group_size', 'flags', 'x_fit', 'y_fit', 
                'flux_psf', 'flux_aper', 'ratio_flux', 'local_bkg', 
                'x_err', 'y_err', 'flux_err', 'airmass', 'id',
                'iter_detected', 'npixfit', 'qfit', 'cfit']
    
    # Create a concise version of the data table
    concise_data = phot_data[colnames]
    
    # Set formatting for float columns to three decimal places
    for col in colnames:
        if isinstance(concise_data[col][0], np.float64):
            concise_data[col].info.format = '.3f'
    
    return concise_data