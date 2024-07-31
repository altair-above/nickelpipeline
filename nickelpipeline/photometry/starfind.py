import numpy as np
import logging

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Moffat2D
from photutils.detection import IRAFStarFinder
from photutils.aperture import CircularAperture
from photutils.psf import IterativePSFPhotometry, make_psf_model
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.psf import IntegratedGaussianPRF, SourceGrouper

from pathlib import Path
from astropy.table import Table
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval

from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.nickel_data import bad_columns, ccd_shape
from nickelpipeline.convenience.log import log_astropy_table

from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single


logger = logging.getLogger(__name__)
np.set_printoptions(edgeitems=100)


def analyze_sources(image, plot=False):
    
    thresh=10.0
    local_bkg_range=(15,25)
    
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    logger.debug(f"analyze_sources() called on image {image.filename}")

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
    
    # Convert source data into Astropy table
    column_names = ['chip', 'id', 'xcentroid', 'ycentroid', 'bkg', 'kron_radius', 'raw_flux', 'flux', '?']
    sources = Table(source_data, names=column_names)
    logger.debug(f"Sources Found (Iter 1): \n{log_astropy_table(sources)}")
    # sources = filter_off_ccd(sources, xname='xcentroid', yname='ycentroid')
    
    # Fit PSF models and get source coordinates and parameters
    source_coords, source_pars, _ = fit_psf_single(base, 1, fittype='circular')
    
    avg_par = np.mean(source_pars, axis=0)
    avg_fwhm = gamma_to_fwhm(avg_par[3], avg_par[4])
    logger.debug(f"Averaged-out Moffat fit parameters: \namplitude = {avg_par[2]}, gamma = {avg_par[3]}, alpha = {avg_par[4]}, background = {avg_par[5]}")
    logger.info(f"Averaged-out FWHM = {avg_fwhm}")
    
    init_phot_data = Table()
    init_phot_data.add_column(source_coords[:,0], name='x_fit')
    init_phot_data.add_column(source_coords[:,1], name='y_fit')
    init_phot_data.add_column(moffat_integral(source_pars[:,2], source_pars[:,3], source_pars[:,4]), name='flux_fit')
    init_phot_data.add_column([i for i in range(len(source_pars))], name='group_id')
    init_phot_data.add_column([1 for _ in range(len(source_pars))], name='group_size')
    # logger.debug(f"Sources Found (Iter 1): \n{log_astropy_table(phot_data)}")
    # init_phot_data = filter_off_ccd(init_phot_data)
    logger.debug(log_astropy_table(init_phot_data))
    
    if plot:
        plot_sources(image, init_phot_data, avg_fwhm)

    # #----------------------------------------------------------------------
    # # Do a first source detection using the default FWHM
    # #----------------------------------------------------------------------
    # _, median, std = sigma_clipped_stats(img, sigma=3.)
    # starfind = IRAFStarFinder(fwhm=avg_fwhm, threshold=thresh*std,
    #                           minsep_fwhm=0.1, sky=0.0, peakmax=55000,)
    # sources = starfind(data=(img.data - median), mask=img.mask)
    # if sources is None:
    #     logger.info(f'Found {len(sources)} sources in {image}.')

    #----------------------------------------------------------------------
    # Attempt to improve the source detection by improving the FWHM estimate
    #----------------------------------------------------------------------
    thresh=7.0
    aper_size=avg_fwhm*1.8
    local_bkg_range=(3*avg_fwhm,6*avg_fwhm)
    win = int(np.ceil(2*avg_fwhm))
    if win % 2 == 0:
        win += 1
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(img)

    # Source finder
    iraffind = IRAFStarFinder(fwhm=avg_fwhm, threshold=thresh*std, 
                              minsep_fwhm=0.1, peakmax=55000)
    grouper = SourceGrouper(min_separation=2*avg_fwhm)  # Grouping algorithm
    mmm_bkg = MMMBackground()   # Background-determining function
    local_bkg = LocalBackground(*local_bkg_range, mmm_bkg)
    fitter = LevMarLSQFitter()  # This is the optimization algorithm
    
    # This is the model of the PSF
    moffat_psf = Moffat2D(gamma=avg_par[3], alpha=avg_par[4])
    moffat_psf = make_psf_model(moffat_psf)
    
    # This is the object that performs the photometry
    phot = IterativePSFPhotometry(finder=iraffind, grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=moffat_psf,
                                  fitter=fitter, fit_shape=win,
                                  aperture_radius=aper_size, mode='all')
    # This is actually when the fitting is done
    phot_data = phot(data=img.data, mask=img.mask,
                     init_params=Table(sources['xcentroid', 'ycentroid', 'flux'],
                                       names=('x_0', 'y_0', 'flux_0')))
    
    logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")
    
    # phot_data = filter_off_ccd(phot_data)
    phot_data = filter_phot_data(phot_data, avg_fwhm)
    
    if plot:
        plot_sources(image, phot_data, avg_fwhm)
    
    return phot_data


def filter_off_ccd(table, xname='x_fit', yname='y_fit'):
    """
    Removes all rows from the table with coordinates outside ccd_shape
    
    Parameters:
    table (astropy.table.Table): The input table.
    
    Returns:
    astropy.table.Table: The filtered table.
    """

    indx2 = table[xname] > 0
    indx3 = table[yname] > 0
    indx4 = table[xname] < ccd_shape[0]
    indx5 = table[yname] < ccd_shape[1]
    
    # Combine all masks
    combined_mask = indx2 & indx3 & indx4 & indx5 #& indx6 #& indx1
    return table[combined_mask]


def filter_phot_data(table, fwhm):
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
    # indx2 = table['x_fit'] > 0
    # indx3 = table['y_fit'] > 0
    # indx4 = table['x_fit'] < ccd_shape[0]
    # indx5 = table['y_fit'] < ccd_shape[1]
    
    i, j = np.meshgrid(np.arange(len(table)), np.arange(len(table)),
                            indexing='ij')
    dist = np.sqrt(np.square(table['x_fit'][:,None]
                             - table['x_fit'][None,:])
                + np.square(table['y_fit'][:,None]
                            - table['y_fit'][None,:]))
    indx6 = (dist < fwhm*1.7) & (i != j) & (j > i)
    indx6 = np.logical_not(np.any(indx6, axis=0))
    logger.debug(f"{len(indx6)-sum(indx6)} sources removed for being too close")
    
    # Combine all masks
    # combined_mask = indx2 & indx3 & indx4 & indx5 #& indx6 #& indx1
    
    indx_flag = table['flags'] <= 1
    indx_high_err = np.array(table['flux_err']) / np.array(table['flux_fit']) < 0.2
    combined_mask = indx_flag & indx_high_err
    
    # table.remove_columns(['x_0_2_init', 'y_0_2_init', 'amplitude_2_init', 'x_0_2_fit', 'y_0_2_fit', 'gamma_2_init', 'alpha_2_init', 'amplitude_4_init'])
    
    # return table[combined_mask]
    return table



def plot_sources(image, phot_data, given_fwhm):
    # bad_sources = phot_data['flags'] > 1
    good_phot_data = phot_data[phot_data['group_size'] <= 1]
    bad_phot_data = phot_data[phot_data['group_size'] > 1]
    
    logger.info(f'Image {image}')
    
    x_good = good_phot_data['x_fit']
    y_good = good_phot_data['y_fit']
    good_positions = np.transpose((x_good, y_good))
    good_apertures = CircularAperture(good_positions, r=2*given_fwhm)
    
    x_bad = bad_phot_data['x_fit']
    y_bad = bad_phot_data['y_fit']
    bad_positions = np.transpose((x_bad, y_bad))
    bad_apertures = CircularAperture(bad_positions, r=2*given_fwhm)
    
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image.masked_array)
    cmap = plt.get_cmap()
    cmap.set_bad('r', alpha=0.5)
    plt.figure(figsize=(12,10))
    plt.imshow(image.masked_array, origin='lower', vmin=vmin, vmax=vmax,
               cmap=cmap, interpolation='nearest')
    plt.colorbar()
    good_apertures.plot(color='m', lw=1.5, alpha=0.5)
    bad_apertures.plot(color='r', lw=1.5, alpha=0.5)
    
    # Annotate good sources with flux_fit values
    for i in range(len(good_phot_data)):
        plt.text(x_good[i]-10, y_good[i]+17, f'{good_phot_data["flux_fit"][i]:.0f}', color='white', fontsize=8, ha='center', va='center')
    
    group_ids = set(bad_phot_data['group_id'])
    for id in group_ids:
        jitter_x = 45
        group = bad_phot_data[bad_phot_data['group_id'] == id]
        for i in range(len(group)):
            plt.text(group['x_fit'][i]+jitter_x, group['y_fit'][i]+(i-2)*20, f'{group["flux_fit"][i]:.0f}', color='red', fontsize=8, ha='center', va='center')
    # for i in range(len(bad_phot_data)):
    #     jitter_x = 45
    #     jitter_y = np.random.normal(scale=35)
    #     plt.text(x_bad[i]+jitter_x, y_bad[i]+jitter_y, f'{bad_phot_data["flux_fit"][i]:.0f}', color='red', fontsize=8, ha='center', va='center')
    plt.gcf().set_dpi(300)
    plt.show()


def check_integrals(phot_data):
    integ = moffat_integral((phot_data['amplitude_2_fit']), phot_data['gamma_2_fit'], phot_data['alpha_2_fit'])
    print(integ)
    phot_data.add_column(np.array(integ), name='integral of moffat psf')
    phot_data.add_column(np.array(integ*phot_data['amplitude_4_fit']), name='integral * amp_4')
    return


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

def moffat_integral(amplitude, gamma, alpha):
    return amplitude * np.pi * gamma**2 / (alpha - 1)

def discrete_moffat_integral(amplitude, gamma, alpha, step_size=1.0):
    # Define the grid size and step size
    grid_size = 10

    # Calculate the start and end points
    half_size = grid_size // 2
    x_start, x_end = -half_size + step_size / 2, half_size - step_size / 2
    y_start, y_end = half_size - step_size / 2, -half_size + step_size / 2
    x_coords = np.arange(x_start, x_end + step_size, step_size)
    y_coords = np.arange(y_start, y_end - step_size, -step_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    # print(grid_x)

    pixel_fluxes = Moffat2D.evaluate(grid_x, grid_y, amplitude, 0, 0, gamma, alpha)
    pixel_fluxes *= step_size**2
    # print(pixel_fluxes)
    print(f"total flux = {np.sum(pixel_fluxes)}")
    return np.sum(pixel_fluxes)

# result = discrete_moffat_integral(0.0381, 4.776, 3.728)


