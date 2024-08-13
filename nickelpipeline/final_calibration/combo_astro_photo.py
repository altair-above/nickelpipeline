import logging
from pathlib import Path

from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from nickelpipeline.convenience.log import log_astropy_table

logger = logging.getLogger(__name__)

def convert_coords_all(photo_dir, astro_dir, final_calib_dir):
    """
    Convert pixel coordinates to sky coordinates (RA/Dec) for all photometric
    source catalogs.

    This function processes photometric source catalogs located in `photo_dir`, applies
    astrometric calibration using the corresponding calibration files in `astro_dir`,
    and saves the resulting catalogs with RA/Dec coordinates to `final_calib_dir`.

    Parameters
    ----------
    photo_dir : Path or str
        Directory containing photometric source catalogs to be converted.
    astro_dir : Path or str
        Directory containing astrometric calibration files.
    final_calib_dir : Path or str
        Directory where the final calibrated catalogs will be saved.

    Returns
    -------
    dict
        A dictionary w/ keys as object names and values are the corresponding
        calibrated source catalogs.
    """
    
    # Create a dictionary of astrometric calibration files indexed by object name
    astro_calibs = {astro_calib.name.split('_')[0]: astro_calib for astro_calib in astro_dir.iterdir()}
    astrophot_datas = {}

    # Iterate over all sub-directories in photo_dir
    for obj_dir in photo_dir.iterdir():
        
        # Create a dictionary of photometric source files indexed by object name
        phot_datas = {phot_data.name.split('_')[0]: phot_data for phot_data in obj_dir.iterdir()}

        # Set output directory based on whether the input directory is consolidated
        if photo_dir.name == 'consolidated':
            astrophot_dir = final_calib_dir / 'astrophotsrcs_consol'
        else:
            astrophot_dir = final_calib_dir / 'astrophotsrcs'
        output_dir = astrophot_dir / obj_dir.name
        Path.mkdir(astrophot_dir, exist_ok=True)
        Path.mkdir(output_dir, exist_ok=True)
        
        logger.info(f"Saving photometric source catalogs with sky coordinates (RA/Dec) to {output_dir}")

        result_datas = {}
        
        # Convert coordinates for each photometric data file if corresponding calibration is available
        for key, phot_data in phot_datas.items():
            output_path = output_dir / (key + '_astrophotsrcs.csv')
            if key in astro_calibs.keys():
                result_datas[key] = convert_coords(phot_data, output_path, astro_calibs[key]) 
                result_datas[key].meta['image_path'] = photo_dir.parent.parent / 'reduced' / obj_dir.name / (key + '_red.fits')
            else:
                logger.warning(f"No astrometric solution found for image {key}; skipping")
        
        astrophot_datas[obj_dir.name] = result_datas
    return astrophot_datas

def convert_coords(phot_data_inpath, phot_data_outpath, astrometric_img_path):
    """
    Convert pixel coordinates in a photometric source catalog to sky coordinates (RA/Dec) using an astrometric FITS file.

    Parameters
    ----------
    phot_data_inpath : Path or str
        Path to the input photometric source catalog (CSV format).
    phot_data_outpath : Path or str
        Path to the output catalog with RA/Dec coordinates (CSV format).
    astrometric_img_path : Path or str
        Path to the astrometric FITS file used for coordinate transformation.

    Returns
    -------
    astropy.table.Table
        The updated photometric source catalog with RA/Dec coordinates.
    """
    
    # Read the photometric source catalog
    phot_data_path = Path(phot_data_inpath)
    phot_data = ascii.read(phot_data_path, format='csv')
    
    # Create a WCS object from the FITS header for coordinate conversion
    _, header = fits.getdata(astrometric_img_path, header=True)
    wcs = WCS(header)
    
    # Extract x and y pixel coordinates from the photometric data
    x_coords = phot_data['x_fit']
    y_coords = phot_data['y_fit']
    
    # Convert pixel coordinates to world coordinates (RA/Dec)
    world_coords = wcs.all_pix2world(x_coords, y_coords, 0)  # 0 specifies no origin offset
    
    # Create SkyCoord object for RA/Dec transformation
    sky_coords = SkyCoord(ra=world_coords[0] * u.deg, 
                          dec=world_coords[1] * u.deg, 
                          frame='icrs', equinox='J2000')
    
    # Convert RA and Dec to HMS and DMS format strings
    ra = sky_coords.ra.to_string(unit=u.hourangle, sep=':', precision=2)
    dec = sky_coords.dec.to_string(unit=u.deg, sep=':', precision=2)
    
    # Insert the new RA/Dec columns into the photometric data table
    col_index = phot_data.colnames.index('y_fit') + 1
    phot_data.add_column(ra, name='ra_hms', index=col_index)
    phot_data.add_column(dec, name='dec_dms', index=col_index + 1)
    phot_data.add_column(sky_coords.ra, name='ra_deg', index=col_index + 2)
    phot_data.add_column(sky_coords.dec, name='dec_deg', index=col_index + 3)
    
    # Format the table for consistency and readability
    phot_data = format_table(phot_data)
    
    # Save the updated catalog to the specified output path
    phot_data.write(phot_data_outpath, format='csv', overwrite=True)
    
    logger.debug(f"Source Catalog with Sky Coordinates: \n{log_astropy_table(phot_data)}")
    logger.info(f"Saving source catalog with RA/Dec coordinates to {phot_data_outpath}")
    return phot_data

def format_table(phot_data):
    """
    Format a photometric data table to include only selected columns.

    This function restructures the photometric data table to include a concise 
    set of columns in most readable order for further analysis.

    Parameters
    ----------
    phot_data : astropy.table.Table
        The input photometric data table.

    Returns
    -------
    astropy.table.Table
        The formatted photometric data table with selected columns.
    """
    
    # Define the columns to retain in the formatted table & their order
    colnames = ['group_id', 'group_size', 'flags', 'ra_hms', 'dec_dms',
                'flux_psf', 'flux_aper', 'ratio_flux', 'local_bkg',
                'x_fit', 'y_fit', 'ra_deg', 'dec_deg', 'x_err', 'y_err', 'flux_err',
                'airmass', 'id', 'iter_detected', 'npixfit', 'qfit', 'cfit']
    
    # Create a new table with only the selected columns
    concise_data = phot_data[colnames]
    
    return concise_data
