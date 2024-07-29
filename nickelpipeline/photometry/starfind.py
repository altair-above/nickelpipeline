import numpy as np
import logging

from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Moffat2D
from photutils.detection import IRAFStarFinder
from photutils.aperture import CircularAperture
from photutils.psf import IterativePSFPhotometry, make_psf_model
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

from nickelpipeline.psf_analysis.moffat.moffat_fitting import get_source_pars
from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single


logger = logging.getLogger(__name__)
np.set_printoptions(edgeitems=25)


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


def analyze_sources(image, mode='psf', plot=False, which_source=None, verbose=True):
    
    default_fwhm=5.0
    thresh=10.0
    aper_size=8
    local_bkg_range=(15,20)
    
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    logger.debug(f"analyze_sources() called on image {image.filename}")
    
    sig2fwhm = np.sqrt(8*np.log(2))
    # aper_size = 1.5*default_fwhm/2

    img = image.masked_array
    img.data[:,bad_columns] = 0
    
    #----------------------------------------------------------------------
    # Use a Moffat fit to find & fit initial sources
    #----------------------------------------------------------------------
    
    # Create output directories
    img_name = image.filename.split('.')[0]
    proc_dir = Path('.').resolve() / "proc_files"
    Path.mkdir(proc_dir, exist_ok=True)
    proc_subdir = proc_dir / 'circular'
    Path.mkdir(proc_subdir, exist_ok=True)
    base_parent = proc_subdir / img_name
    Path.mkdir(base_parent, exist_ok=True)
    base = proc_subdir / img_name / img_name
    
    # Generate stamps (image of sources) for image data
    source_data = generate_stamps([image], output_base=base, thresh=thresh)
    
    # Fit PSF models and get source coordinates and parameters
    _, source_pars, _ = fit_psf_single(base, 1, fittype='circular')
    
    avg_par = np.mean(source_pars, axis=0)
    logger.debug(f"Averaged-out Moffat fit parameters: {avg_par}")

    # #----------------------------------------------------------------------
    # # Do a first source detection using the default FWHM
    # #----------------------------------------------------------------------
    # _, median, std = sigma_clipped_stats(img, sigma=3.)
    # starfind = IRAFStarFinder(fwhm=default_fwhm, threshold=thresh*std,
    #                           minsep_fwhm=0.1, sky=0.0, peakmax=55000,)
    # sources = starfind(data=(img.data - median), mask=img.mask)
    # # if sources is None:
        
    # logger.info(f'Found {len(sources)} sources in {image}.')
    # # This determines the distance of each source to every other
    # # source and removes sources that are too close to one another
    # i, j = np.meshgrid(np.arange(len(sources)), np.arange(len(sources)),
    #                         indexing='ij')
    # dist = np.sqrt(np.square(sources['xcentroid'][:,None]
    #                          - sources['xcentroid'][None,:])
    #             + np.square(sources['ycentroid'][:,None]
    #                         - sources['ycentroid'][None,:]))
    # indx = (dist < default_fwhm/1.7) & (j > i)
    # logger.debug(f"{sum(indx)} sources removed for being too close")
    # sources = sources[np.logical_not(np.any(indx, axis=0))]
    # Define the column names
    column_names = ['CHIP', 'ID', 'X', 'Y', 'BKG', 'R', 'RAW_FLUX', 'FLUX', '?']
    # Create the Astropy table
    sources = Table(source_data, names=column_names)
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
    
    # # # This is the model of the PSF
    # gaussian_prf = IntegratedGaussianPRF(sigma=default_fwhm/sig2fwhm)
    # gaussian_prf.sigma.fixed = False    # Turn off fixing the sigma to be a constant
    
    # This is the model of the PSF
    # avg_fwhm = np.mean(sources['fwhm'])
    # avg_gamma = fwhm_to_gamma(avg_fwhm, alpha=3.5)
    moffat_psf = Moffat2D(gamma=avg_par[3], alpha=avg_par[4])
                        #   fixed={'amplitude':False, 'x_0':False, 
                        #          'y_0':False, 'gamma':False, 'alpha':False})
    moffat_psf = make_psf_model(moffat_psf)
    # moffat_psf.fixed['gamma_2'] = False
    # moffat_psf.fixed['alpha_2'] = False
    # moffat_psf.fixed['amplitude_4'] = False
    
    # This is the object that performs the photometry
    phot = IterativePSFPhotometry(finder=iraffind, grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=moffat_psf,
                                  fitter=fitter, fit_shape=win, maxiters=4,
                                  fitter_maxiters=4, aperture_radius = aper_size)
    # This is actually when the fitting is done
    phot_data = phot(data=img.data, mask=img.mask,
                     init_params=Table(sources['X', 'Y', 'FLUX'],
                                       names=('x_0', 'y_0', 'flux_0')))
    plot_sources(image, phot_data['x_fit'], phot_data['y_fit'], gamma_to_fwhm(avg_par[3], avg_par[4]))
    logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")
    return phot_data

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
    combined_mask = indx2 & indx3 & indx4 & indx5 #& indx1
    
    return table[combined_mask]



def fwhm_to_gamma(fwhm, alpha):
        """
        Convert full-width half-maximum (FWHM) to gamma.
        """
        return fwhm / 2 / np.sqrt(2**(1/alpha)-1)

def gamma_to_fwhm(gamma, alpha):
    """
    Convert gamma to full-width half-maximum (FWHM).
    """
    return 2 * gamma * np.sqrt(2**(1/alpha)-1)

