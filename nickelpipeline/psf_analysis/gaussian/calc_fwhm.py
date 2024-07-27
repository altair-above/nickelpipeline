import numpy as np
import logging

from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from photutils import IRAFStarFinder
from photutils import CircularAperture
from photutils.psf import IterativePSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.psf import IntegratedGaussianPRF, SourceGrouper
from photutils.aperture import ApertureStats

from pathlib import Path
from astropy.table import Table
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import SigmaClip

from nickelpipeline.reduction.old_basic.reduction_split import process_single
from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.dir_nav import unzip_directories, categories_from_conditions
from nickelpipeline.convenience.nickel_data import bad_columns, bad_photometry_columns, ccd_shape
from nickelpipeline.convenience.log import log_astropy_table
from nickelpipeline.convenience.display_fits import display_nickel

# from astrometry.plate_scale import avg_plate_scale

logger = logging.getLogger(__name__)
np.set_printoptions(edgeitems=25)

def testing():
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_199_{filt}') for filt in ['B', 'V', 'R', 'I']]

    rawdir = Path("")
    rawdir = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw")  # path to directory with raw data

    image_to_analyze = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw/d1079.fits")  # path to directory with raw data

    # Initial configuration
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/PG1530+057_{filt}') for filt in ['B', 'V', 'R', 'I']]
    # Night 1 mod
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_231_{filt}') for filt in ['R']]

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if 'flat' not in str(dir)]

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]
    # graph_fwhms_by_setting(directories, condition_tuples=conditions_06_26)
    
    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]
    # graph_fwhms_by_setting(directories, condition_tuples=conditions_06_26)

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('109_199' in str(dir))]
    # graph_fwhms_by_image(directories, '06-26-24')

    # bias, flat = generate_reduction_files(rawdir)
    # fwhm = fwhm_from_raw(image_to_analyze, bias, flat)




def plot_sources(image, x, y, given_fwhm):
    logger.info(f'Image {image}')
    positions = np.transpose((x, y))
    apertures = CircularAperture(positions, r=2*given_fwhm)
    
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image.masked_array)
    cmap = plt.get_cmap()
    cmap.set_bad('r', alpha=0.5)

    plt.imshow(image.masked_array, origin='lower', vmin=vmin, vmax=vmax,
               cmap=cmap, interpolation='nearest')
    plt.colorbar()
    apertures.plot(color='r', lw=1.5, alpha=0.5)
    plt.show()


def calc_fwhm(image, mode='psf', plot=False, which_source=None, verbose=True):
    
    default_fwhm=5.0
    thresh=15
    aper_size=8
    local_bkg_range=(15,20)
    
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    logger.debug(f"calc_fwhm() called on image {image.filename}")
    
    sig2fwhm = np.sqrt(8*np.log(2))
    # aper_size = 1.5*default_fwhm/2

    img = image.masked_array
    img.data[:,bad_columns] = 0

    #----------------------------------------------------------------------
    # Do a first source detection using the default FWHM
    #----------------------------------------------------------------------
    _, median, std = sigma_clipped_stats(img, sigma=3.)
    starfind = IRAFStarFinder(fwhm=default_fwhm, threshold=thresh*std,
                              minsep_fwhm=0.1, sky=0.0, peakmax=55000,)
    sources = starfind(data=(img.data - median), mask=img.mask)
    if verbose:
        print(f'Found {len(sources)} sources in {image}.')
    # This determines the distance of each source to every other
    # source and removes sources that are too close to one another
    i, j = np.meshgrid(np.arange(len(sources)), np.arange(len(sources)),
                            indexing='ij')
    dist = np.sqrt(np.square(sources['xcentroid'][:,None]
                             - sources['xcentroid'][None,:])
                + np.square(sources['ycentroid'][:,None]
                            - sources['ycentroid'][None,:]))
    indx = (dist < default_fwhm/1.7) & (j > i)
    sources = sources[np.logical_not(np.any(indx, axis=0))]
    logger.debug(f"Sources Found (Iter 1): \n{log_astropy_table(sources)}")

    #----------------------------------------------------------------------
    # Attempt to improve the source detection by improving the FWHM estimate
    #----------------------------------------------------------------------
    win = int(np.ceil(2*default_fwhm))
    if win % 2 == 0:
        win += 1
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(img)
    # Source finder
    iraffind = IRAFStarFinder(fwhm=default_fwhm, threshold=thresh*std, 
                              minsep_fwhm=0.1, peakmax=55000)
    grouper = SourceGrouper(min_separation=2*default_fwhm)  # Grouping algorithm
    mmm_bkg = MMMBackground()   # Background-determining function
    local_bkg = LocalBackground(*local_bkg_range, mmm_bkg)
    fitter = LevMarLSQFitter()  # This is the optimization algorithm
    # This is the model of the PSF
    gaussian_prf = IntegratedGaussianPRF(sigma=default_fwhm/sig2fwhm)
    gaussian_prf.sigma.fixed = False    # Turn off fixing the sigma to be a constant
    # This is the object that performs the photometry
    phot = IterativePSFPhotometry(finder=iraffind,
                                  grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=gaussian_prf,
                                  fitter=fitter, fit_shape=win, maxiters=4,
                                  fitter_maxiters=4, aperture_radius = aper_size)
    # This is actually when the fitting is done
    phot_data = phot(data=img.data, mask=img.mask,
                     init_params=Table(sources['xcentroid', 'ycentroid', 'flux'],
                                       names=('x_0', 'y_0', 'flux_0')))

    logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")

    phot_data = filter_phot_data(phot_data)
    fwhm = np.median(np.abs(phot_data['sigma_fit']))*sig2fwhm
    
    #----------------------------------------------------------------------
    # Refit using the "improved" FWHM
    #----------------------------------------------------------------------
    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=thresh*std,
                              minsep_fwhm=0.1, peakmax=55000)
    grouper = SourceGrouper(min_separation=2*fwhm)
    gaussian_prf = IntegratedGaussianPRF(sigma=fwhm/sig2fwhm)
    gaussian_prf.sigma.fixed = False
    phot = IterativePSFPhotometry(finder=iraffind,
                                  grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=gaussian_prf,
                                  fitter=fitter, fit_shape=win, maxiters=2,
                                  fitter_maxiters=2, aperture_radius = aper_size)
    phot_data = phot(data=img.data, mask=img.mask,
                     init_params=Table([phot_data['x_fit'],
                                        phot_data['y_fit'],
                                        phot_data['flux_fit']],
                                        names=('x_0', 'y_0', 'flux_0')))
    
    logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")
    #----------------------------------------------------------------------
    # Extract the source which_source, & calculate fwhm
    #----------------------------------------------------------------------

    # Extracts only the which_source brightest src from phot_data to analyze
    if which_source is not None:
        phot_data = phot_data[phot_data['iter_detected']]
        indices_by_peak = phot_data.argsort('flux_init')
        chosen_star = indices_by_peak[which_source]
        print(f"flux of star chosen = {phot_data['flux_init'][chosen_star]}")
        phot_data = phot_data[indices_by_peak[chosen_star]]
    
    phot_data = filter_phot_data(phot_data)
    psf_fwhm_median = np.median(phot_data['sigma_fit'])*sig2fwhm
    psf_fwhm_std = np.std(phot_data['sigma_fit']*sig2fwhm)

    if plot:
        plot_sources(image, phot_data['x_fit'], phot_data['y_fit'],fwhm)
    # plot_sources(image, phot_data['x_fit'], phot_data['y_fit'],fwhm)
    
    #----------------------------------------------------------------------
    # AperatureStats Analysis
    #----------------------------------------------------------------------
    if mode == 'aper':
        coordinates = list(zip(phot_data['x_fit'], phot_data['y_fit']))
        local_bkg_values = local_bkg(img, phot_data['x_fit'], phot_data['y_fit'])
        apertures = CircularAperture(coordinates, r=aper_size)
        aperstats = ApertureStats(img, apertures, local_bkg=local_bkg_values)
        
        aper_fwhm = aperstats.fwhm
        aper_fwhm_median = np.nanmedian(aper_fwhm)
        aper_fwhm_std = np.nanstd(aper_fwhm)

    #----------------------------------------------------------------------
    # Sigma Clip PSF FWHMs
    #----------------------------------------------------------------------
    # (psf_fwhm_median, aper_fwhm_median, psf_fwhm_std, aper_fwhm_std)
    all_fwhms = np.array(phot_data['sigma_fit'])*sig2fwhm
    all_x = np.array(phot_data['x_fit'])
    all_y = np.array(phot_data['y_fit'])
    # Create a SigmaClip object and apply it to get a mask
    sigma_clip = SigmaClip(sigma=3, maxiters=5)
    masked_fwhms = sigma_clip(all_fwhms)

    # Apply the mask to the original data
    clipped_x = np.array(all_x)[~masked_fwhms.mask]
    clipped_y = np.array(all_y)[~masked_fwhms.mask]
    clipped_fwhms = np.array(all_fwhms)[~masked_fwhms.mask]
    
    if verbose:
        print("Number of sources removed =", len(all_x) - len(clipped_x))
        print(clipped_fwhms)
    
    #----------------------------------------------------------------------
    # Return
    #----------------------------------------------------------------------
    if mode == 'psf':
        return psf_fwhm_median, psf_fwhm_std
    elif mode == 'aper':
        return aper_fwhm_median
    elif mode == 'fwhms, std':
        return all_fwhms, psf_fwhm_std
    elif mode == 'fwhm':
        return clipped_x, clipped_y, clipped_fwhms
    elif mode == 'fwhm residuals':
        # Calculate the residuals wrt the minimum FWHM
        # Approximating "removing" the atmospheric FWHM, leaving (mostly) the telescopic
        min_fwhm = np.min(clipped_fwhms)
        fwhm_residuals = np.sqrt(clipped_fwhms**2 - min_fwhm**2)
        if verbose:
            print("min_fwhm = ", min_fwhm)
            print(fwhm_residuals)
        return (clipped_x, clipped_y, fwhm_residuals)
    else:
        raise ValueError("mode must = 'psf', 'aper', 'fwhm', 'fwhms, std', or 'fwhm coords'")


def filter_phot_data(table):
    """
    Removes all rows from the table with coordinates outside ccd_shape
    and with 'iter_detected' == 1.
    
    Parameters:
    table (astropy.table.Table): The input table.
    ccd_shape (tuple): A tuple (width, height) representing the shape of the CCD.
    
    Returns:
    astropy.table.Table: The filtered table.
    """
    # Create boolean masks for each condition
    indx1 = table['iter_detected'] == 1 #np.max(table['iter_detected'])
    indx2 = table['x_fit'] > 0
    indx3 = table['y_fit'] > 0
    indx4 = table['x_fit'] < ccd_shape[0]
    indx5 = table['y_fit'] < ccd_shape[1]
    
    # Combine all masks using logical AND
    combined_mask = indx2 & indx3 & indx4 & indx5 & indx1
    
    return table[combined_mask]


def fwhm_from_raw(image, bias, flat):
    image = Path(image)
    reduced_image = process_single(image, bias, flat)
    fwhm = calc_fwhm(reduced_image)
    print(f"FWHM for image {image.name} = {fwhm}")
    return fwhm


def batch_fwhm(path_list, plot=False, max_std=0.5, mode='psf'):
    images = unzip_directories(path_list, output_format='Fits_Simple')
    
    fwhms = []
    stds = []
    data = []
    for image in images:
        try:
            fwhm, std = calc_fwhm(image, plot=plot, mode=mode)
        except TypeError as e:
            print(f"No sources detected in {image} -- {e}")
        print(f"FWHM for image {image} = {fwhm} pix")
        print(f"STD for image {image} = {std} pix")
        if std < max_std:
            data.append((int(image.image_num), fwhm, std, image.object))#f"{image.object} - {image.filtnam}"))
    
    image_numbers, fwhms, stds, objects = zip(*data)

    avg = np.mean(np.hstack(fwhms))
    print(f'Mean FWHM for all images = {avg}')
    avg_std = np.mean(stds)
    print(f'Mean of individual images STDs = {avg_std}')
    std = np.std(np.hstack(fwhms))
    print(f'STD for average FWHM measurements = {std}')
    
    return avg, data


def avg_fwhm(directories, files=None, plot=False, max_std=0.5):
    avg, _ = batch_fwhm(directories, files=files, plot=plot, max_std=max_std)
    return avg
