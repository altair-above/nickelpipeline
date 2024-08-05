import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.spatial import KDTree
from scipy.optimize import lsq_linear

from nickelpipeline.convenience.log import log_astropy_table

logger = logging.getLogger(__name__)

def convert_coords_all(photo_dir, astro_dir, final_calib_dir):
    
    astro_calibs = {astro_calib.name.split('_')[0]: astro_calib for astro_calib in astro_dir.iterdir()}
    astrophot_datas = {}

    for obj_dir in photo_dir.iterdir():
        
        phot_datas = {phot_data.name.split('_')[0]: phot_data for phot_data in obj_dir.iterdir()}

        if photo_dir.name == 'consolidated':
            astrophot_dir = final_calib_dir / 'astrophotsrcs_consol'
        else:
            astrophot_dir = final_calib_dir / 'astrophotsrcs'
        output_dir = astrophot_dir / obj_dir.name
        Path.mkdir(astrophot_dir, exist_ok=True)
        Path.mkdir(output_dir, exist_ok=True)
        logger.info(f"Saving photometric source catalogs with sky coordinates (RA/Dec) to {output_dir}")

        result_datas = {}
        for key, phot_data in phot_datas.items():
            output_path = output_dir / (key + '_astrophotsrcs.csv')
            if key in astro_calibs.keys():
                result_datas[key] = convert_coords(phot_data, output_path, astro_calibs[key]) 
            else:
                logger.warning(f"No astrometric solution found for image {key}; skipping")
        
        astrophot_datas[obj_dir.name] = result_datas
    return astrophot_dir

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
    
    # Create SkyCoord object for RA/Dec transformation
    sky_coords = SkyCoord(ra=world_coords[0]*u.deg, 
                          dec=world_coords[1]*u.deg, 
                          frame='icrs', equinox='J2000')
    
    # Add RA and Dec in hms and dms format
    ra = sky_coords.ra.to_string(unit=u.hourangle, sep=':', precision=2)
    dec = sky_coords.dec.to_string(unit=u.deg, sep=':', precision=2)
    
    col_index = phot_data.colnames.index('y_fit') + 1
    phot_data.add_column(ra, name='ra_hms', index=col_index)
    phot_data.add_column(dec, name='dec_dms', index=col_index + 1)
    phot_data.add_column(sky_coords.ra, name='ra_deg', index=col_index + 2)
    phot_data.add_column(sky_coords.dec, name='dec_deg', index=col_index + 2)
    phot_data = format_table(phot_data)
    
    phot_data.write(phot_data_outpath, format='csv', overwrite=True)
    
    logger.debug(f"Source Catalog w/ Sky Coordinates: \n{log_astropy_table(phot_data)}")
    logger.info(f"Saving source catalog w/ RA/Dec coords to {phot_data_outpath}")
    return phot_data


def photometric_calib_all(astrophot_dir):
    
    for obj_dir in astrophot_dir.iterdir():
        logger.info(f"Analyzing {obj_dir.name}")
        phot_data_paths = list(obj_dir.iterdir())
        
        if '109_199' in obj_dir.name:
            stdrd_coords = [(266.2609583, -0.4913139)]
            stdrd_mags = [10.99,]
            stdrd_names = ['109_199']
        elif '110_232' in obj_dir.name:
            stdrd_coords = [(280.1902500, 0.0304917), (280.2145417, 0.0397944), 
                            (280.2180417, 0.0318750), (280.219625, 0.0141222)]
            stdrd_mags = [13.65, 14.28, 12.52, 12.77]
            stdrd_names = ['110_229', '110_230', '110_232', '110_233']
        zs, ks = fit_zk(stdrd_coords, stdrd_mags, phot_data_paths, flux_name='flux_psf')


def fit_zk(stdrd_coords, stdrd_mags, phot_data_paths, flux_name='flux_psf', plot=False):
    if len(phot_data_paths) == 0:
        logger.warning("No source catalogs found; skipping")
        return None, None
    
    zs = []
    ks = []
    prop_errors = []
    new_errors = []
    for coord, m_a in zip(stdrd_coords, stdrd_mags):
        logger.info(f"Fitting z, k to mag {m_a:.2f} standard star at {coord}")
        
        m_is = []
        airmasses = []
        prop_errors_part = []
        for phot_data_path in phot_data_paths:
            phot_data = ascii.read(phot_data_path, format='csv')
            world_coords = [(ra, dec) 
                            for ra, dec in zip(phot_data['ra_deg'].data,
                                               phot_data['dec_deg'].data)]
            search_tree = KDTree(world_coords)
            matched_index = match_coords(coord, search_tree, 0.001)
            if matched_index is not None:
                stdrd_src = phot_data[matched_index]
                if not np.isnan(stdrd_src[flux_name]):
                    m_is.append(stdrd_src[flux_name])
                    airmasses.append(stdrd_src['airmass'])
                    prop_errors_part.append(((-2.5/(stdrd_src[flux_name]*np.log(10)))*((stdrd_src['flux_err'])))**2)
        
        logger.info(f"Standard star found in {len(m_is)} of {len(phot_data_paths)} images")
        m_is = -2.5 * np.log10(np.array(m_is))   # Instrumental magnitude
        m_as = m_a * np.ones(len(m_is))    # Apparent magnitude from catalog
        airmasses = np.array(airmasses)
        prop_errors.append(np.nanmedian(prop_errors_part))
        
        # Compute vector b
        b = m_as - m_is
        # Construct matrix A
        A = np.column_stack((np.ones_like(airmasses), airmasses - 1))

        # Solve the system using lsq_linear
        result = lsq_linear(A, b)
        z, k = result.x
        zs.append(z)
        ks.append(k)

        # Output the results
        logger.info(f"z = {z}")
        logger.info(f"k = {k}")
        
        # Calculate m_a,calculated for each m_i
        m_a_calcs = z + k * (airmasses - 1) + m_is
        # Calculate the difference between actual m_a and calculated m_a
        differences = m_a - m_a_calcs
        new_errors.append(np.std(differences))

        if plot:
            # Plot the difference as a function of airmass
            plt.figure(figsize=(10, 6))
            plt.plot(airmasses, differences, 'o', label='Difference (m_a - m_a_calculated)')
            plt.xlabel('Airmass (a)')
            plt.ylabel('Difference')
            plt.title('Difference between Actual m_a and Calculated m_a as a Function of Airmass')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    prop_errors, new_errors = np.array(prop_errors), np.array(new_errors)
    print(prop_errors)
    print(new_errors)
    print(new_errors/prop_errors)
    
    # return z, k
    return zs, ks



# def fit_zk(stdrd_coords, stdrd_mags, phot_data_paths, flux_name='flux_psf', 
#            mode='single', plot=False):
#     if len(phot_data_paths) == 0:
#         logger.warning("No source catalogs found; skipping")
#         return None, None
    
#     if mode == 'bulk':
#         m_is, m_as, airmasses = [], [], []
#     elif mode == 'single':
#         zs, ks = [], []
    
#     def do_fit(m_is, m_as, airmasses, prop_errors_part):
#         logger.info(f"Standard star found in {len(m_is)} of {len(phot_data_paths)} images")
#         m_is = np.array(m_is)    # Instrumental magnitude
#         m_as = np.array(m_as)    # Apparent magnitude from catalog
#         airmasses = np.array(airmasses)
#         prop_errors.append(np.nanmean(prop_errors_part))
        
#         # Compute vector b, matrix A
#         b = m_as - m_is
#         A = np.column_stack((np.ones_like(airmasses), airmasses - 1))

#         # Solve the system using lsq_linear
#         result = lsq_linear(A, b)
#         z, k = result.x

#         # Output the results
#         logger.info(f"z = {z:.3f}")
#         logger.info(f"k = {k:.3f}")

#         if plot:
#             # Calculate m_a,calculated for each m_i
#             m_a_calcs = z + k * (airmasses - 1) + m_is
#             # Calculate the difference between actual m_a and calculated m_a
#             differences = m_a - m_a_calcs
#             # new_errors.append(10**np.std(differences)/(-2.5))
            
#             # Plot the difference as a function of airmass
#             plt.figure(figsize=(10, 6))
#             plt.plot(airmasses, differences, 'o', label='Difference (m_a - m_a_calculated)')
#             plt.xlabel('Airmass (a)')
#             plt.ylabel('Difference')
#             plt.title('Difference between Actual m_a and Calculated m_a as a Function of Airmass')
#             plt.legend()
#             plt.grid(True)
#             plt.show()
        
#         return z, k
        
#     for coord, m_a in zip(stdrd_coords, stdrd_mags):
#         logger.info(f"Fitting z, k to mag {m_a:.2f} standard star at {coord}")
        
#         if mode == 'single':
#             m_is = []
#             m_as = []
#             airmasses = []
#             prop_errors = []
#         prop_errors_part = []
#         for phot_data_path in phot_data_paths:
#             phot_data = ascii.read(phot_data_path, format='csv')
#             world_coords = [(ra, dec) 
#                             for ra, dec in zip(phot_data['ra_deg'].data,
#                                                phot_data['dec_deg'].data)]
#             search_tree = KDTree(world_coords)
#             matched_index = match_coords(coord, search_tree, 0.001)
#             if matched_index is not None:
#                 stdrd_src = phot_data[matched_index]
#                 if not np.isnan(stdrd_src[flux_name]):
#                     m_is.append(-2.5 * np.log10(stdrd_src[flux_name]))
#                     m_as.append(m_a)
#                     airmasses.append(stdrd_src['airmass'])
#                     prop_errors_part.append(((-2.5/stdrd_src[flux_name]/np.log(10))**2)*(stdrd_src['flux_err'])**2)
        
#         if mode == 'single':
#             z, k = do_fit(m_is, m_as, airmasses, prop_errors_part)
#             zs.append(z)
#             ks.append(k)
#     if mode == 'bulk':
#         z, k = do_fit(m_is, m_as, airmasses)
#         return z, k
    
#     prop_errors, new_errors = np.array(prop_errors), np.array(new_errors)
#     print(prop_errors)
#     print(new_errors)
#     print(new_errors/prop_errors)
    
#     # return z, k
#     return zs, ks


# def fit_zk_bulk(stdrd_coords, stdrd_mags, phot_data_paths, flux_name='flux_psf'):
#     if len(phot_data_paths) == 0:
#         return None, None
    
#     zs = []
#     ks = []
#     m_is = []
#     m_as = []
#     airmasses = []
#     for coord, m_a in zip(stdrd_coords, stdrd_mags):
#         logger.info(f"Fitting z, k to mag {m_a:.2f} standard star at {coord}")
        
#         for phot_data_path in phot_data_paths:
#             phot_data = ascii.read(phot_data_path, format='csv')
#             world_coords = [(ra, dec) 
#                             for ra, dec in zip(phot_data['ra_deg'].data,
#                                                phot_data['dec_deg'].data)]
#             search_tree = KDTree(world_coords)
#             matched_index = match_coords(coord, search_tree, 0.001)
#             if matched_index is not None:
#                 stdrd_src = phot_data[matched_index]
#                 if not np.isnan(stdrd_src[flux_name]):
#                     m_is.append(stdrd_src[flux_name])
#                     m_as.append(m_a)
#                     airmasses.append(stdrd_src['airmass'])
        
#     logger.info(f"Standard star found in {len(m_is)} of {len(phot_data_paths)*len(stdrd_coords)} images")
#     m_is = -2.5 * np.log10(np.array(m_is))   # Instrumental magnitude
#     m_as = np.array(m_as)    # Apparent magnitude from catalog
#     airmasses = np.array(airmasses)
    
#     print(airmasses)
#     print(m_is)
#     print(m_as)
    
#     # Compute vector b
#     b = m_as - m_is
#     # Construct matrix A
#     A = np.column_stack((np.ones_like(airmasses), airmasses - 1))

#     # Solve the system using lsq_linear
#     result = lsq_linear(A, b)
#     z, k = result.x
#     zs.append(z)
#     ks.append(k)

#     # Output the results
#     logger.info(f"z = {z}")
#     logger.info(f"k = {k}")
    
#     # Calculate m_a,calculated for each m_i
#     m_a_calculateds = z + k * (airmasses - 1) + m_is

#     # Calculate the difference between actual m_a and calculated m_a
#     differences = m_a - m_a_calculateds

#     # Plot the difference as a function of a
#     plt.figure(figsize=(10, 6))
#     plt.plot(airmasses, differences, 'o', label='Difference (m_a - m_a_calculated)')
#     plt.xlabel('Airmass (a)')
#     plt.ylabel('Difference')
#     plt.title('Difference between Actual m_a and Calculated m_a as a Function of Airmass')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
#     return zs, ks


def match_coords(target, search_tree, stop_dist=0.001):
    
    max_dist = 0.0002
    indices = []
    while len(indices) == 0 and max_dist < stop_dist:
        max_dist += 0.0001
        indices = search_tree.query_ball_point(target, max_dist)
    if max_dist >= 0.001:
        logger.warning(f"No source found within {stop_dist} degrees of {target}")
        return None
    logger.debug(f"Search found indices {indices} within {max_dist:.4f} deg of {target}")
    if len(indices) > 1:
        logger.info(f"Multiple nearby sources that could match this group; choosing the closer")
    
    return indices[0]

def format_table(phot_data):
    colnames = ['group_id', 'group_size', 'flags', 'ra_hms', 'dec_dms',
                'flux_psf', 'flux_aper', 'ratio_flux', 'local_bkg',
                'x_fit', 'y_fit', 'ra_deg', 'dec_deg', 'x_err', 'y_err', 'flux_err',
                'airmass', 'id', 'iter_detected', 'npixfit', 'qfit', 'cfit']
    concise_data = phot_data[colnames]
    
    return concise_data