import numpy as np
import logging

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Moffat2D
from photutils.detection import IRAFStarFinder
from photutils.aperture import CircularAperture
from photutils.psf import IterativePSFPhotometry, make_psf_model
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.psf import SourceGrouper

from pathlib import Path
from astropy.table import Table
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval

from scipy.spatial import KDTree

from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.nickel_data import bad_columns
from nickelpipeline.convenience.log import log_astropy_table
from nickelpipeline.convenience.graphs import plot_sources

from nickelpipeline.photometry.moffat_model_photutils import MoffatElliptical2D
from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single, fit_psf_stack, psf_plot


logger = logging.getLogger(__name__)
np.set_printoptions(edgeitems=100)


def psf_analysis(image, proc_dir, thresh=10.0, mode='all', fittype='circ',
                 plot_final=True, plot_inters=False):
    
    local_bkg_range=(15,25)
    
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    logger.debug(f"analyze_sources() called on image {image.filename}")

    new_mask = image.mask.copy()
    new_mask[:5, :] = True
    new_mask[-5:, :] = True
    new_mask[:, :5] = True
    new_mask[:, -5:] = True
    image.mask = new_mask
    new_data = image.data.copy()
    new_data[:,bad_columns] = 0
    image.data = new_data
    img = image.masked_array
    
    #----------------------------------------------------------------------
    # Use a Moffat fit to find & fit initial sources
    #----------------------------------------------------------------------
    
    # Create output directories
    img_name = image.path.stem.split('_')[0]
    proc_subdir = proc_dir / fittype
    Path.mkdir(proc_subdir, exist_ok=True)
    base_parent = proc_subdir / img_name
    Path.mkdir(base_parent, exist_ok=True)
    base = proc_subdir / img_name / img_name
    
    # Generate stamps (image of sources) for image data
    source_data = generate_stamps([image], output_base=base, thresh=thresh)
    
    # Convert source data into Astropy table
    column_names = ['chip', 'id', 'xcentroid', 'ycentroid', 'bkg', 'kron_radius', 'raw_flux', 'flux', '?']
    sources = Table(source_data, names=column_names)
    table_str = log_astropy_table(sources)
    logger.debug(f"Sources Found (Iter 1): \n{table_str}")
    
    # Fit PSF models and get source coordinates and parameters
    source_coords, source_fits, _ = fit_psf_single(base, 1, fittype=fittype, sigma_clip=False)
    source_pars = np.array([fit.par for fit in source_fits])

    try:
        psf_file = Path(f'{str(base)}.psf.fits').resolve()  # PSF info stored here
        stack_par = fit_psf_stack(base, 1, fittype=fittype, ofile=psf_file).par
        stack_fwhm = process_par(stack_par, 'Stack', fittype=fittype)
        fit_par = stack_par
        fit_fwhm = stack_fwhm
    except:
        brightest = np.array(sorted(source_pars, key=lambda coord: coord[2])[:min(7, len(source_pars))])
        clip_avg_par = np.mean(brightest, axis=0)
        clip_avg_fwhm = process_par(clip_avg_par, 'Clipped Avg', fittype=fittype)
        fit_par = clip_avg_par
        fit_fwhm = clip_avg_fwhm
    
    init_phot_data = Table()
    init_phot_data.add_column(source_coords[:,0], name='x_fit')
    init_phot_data.add_column(source_coords[:,1], name='y_fit')
    flux_integrals = [discrete_moffat_integral(par, fittype=fittype, step_size=0.5) for par in source_pars]
    init_phot_data.add_column(flux_integrals, name='flux_fit')
    init_phot_data.add_column(list(range(len(source_pars))), name='group_id')
    init_phot_data.add_column([1] * len(source_pars), name='group_size')
    init_phot_data.meta['image_path'] = image.path
    
    if plot_inters:
        plot_sources(init_phot_data, fit_fwhm)

    #----------------------------------------------------------------------
    # Attempt to improve the source detection by improving the FWHM estimate
    #----------------------------------------------------------------------
    aper_size=fit_fwhm*1.8
    local_bkg_range=(3*fit_fwhm,6*fit_fwhm)
    win = int(np.ceil(2*fit_fwhm))
    if win % 2 == 0:
        win += 1
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(img)

    # Source finder
    iraffind = IRAFStarFinder(fwhm=fit_fwhm, threshold=thresh*std, 
                              minsep_fwhm=0.1)
    grouper = SourceGrouper(min_separation=2*fit_fwhm)  # Grouping algorithm
    mmm_bkg = MMMBackground()   # Background-determining function
    local_bkg = LocalBackground(*local_bkg_range, mmm_bkg)
    fitter = LevMarLSQFitter()  # This is the optimization algorithm
    
    # Set bad columns to the background value
    bkgd = mmm_bkg.calc_background(img)
    new_data = img.data.copy()
    new_data[:,bad_columns] = bkgd
    img = np.ma.masked_array(new_data, img.mask)
    image.data = new_data
    
    # This is the model of the PSF
    moffat_psf = Moffat2D(gamma=fit_par[3], alpha=fit_par[4])
    moffat_psf = make_psf_model(moffat_psf)
    
    # This is the object that performs the photometry
    phot = IterativePSFPhotometry(finder=iraffind, grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=moffat_psf,
                                  fitter=fitter, fit_shape=win,
                                  aperture_radius=aper_size, mode=mode,)
                                  #fitter_maxiters=250)
    # This is actually when the fitting is done
    phot_data = phot(data=img.data, mask=img.mask,
                     init_params=Table(sources['xcentroid', 'ycentroid', 'flux'],
                                       names=('x_0', 'y_0', 'flux_0')))
    phot_data.add_column(image.airmass * np.ones(len(phot_data)), name='airmass')
    phot_data.meta['image_path'] = image.path
    
    table_str = log_astropy_table(phot_data)
    logger.debug(f"Sources Found (Iter 2): \n{table_str}")
    
    if plot_inters:
        plot_groups(phot_data, source_coords, source_fits, base)
    if plot_final:
        plot_sources(phot_data, fit_fwhm)
    
    return phot_data


def plot_groups(phot_data, source_coords, source_fits, base):
    group_data = phot_data[phot_data['group_size'] > 1]
    # group_data = phot_data[phot_data['group_id'] == 18]
    group_ids = list(sorted(set(group_data['group_id'])))
    for id in group_ids:
        logger.warning(f"Group {id} has multiple fitted PSF's: displaying original source")
        group = phot_data[phot_data['group_id'] == id]
        group_x = np.median(group['x_fit'])
        group_y = np.median(group['y_fit'])
        
        matching_indices = match_coords((group_x, group_y), source_coords, 2.0)
        if len(matching_indices) == 0:
            matching_indices = match_coords((group_x, group_y), source_coords, 4.0)
            if len(matching_indices) == 0:
                logger.warning("No nearby source found to display")
        if len(matching_indices) > 1:
            logger.info(f"Multiple nearby sources that could match this group; displaying all")
        for index in matching_indices:
            matching_fit = source_fits[index]
            plot_file = Path(f'{str(base)}_src{index+1}.psf.pdf').resolve()
            psf_plot(plot_file, matching_fit, show=True, plot_fit=True)
        

def match_coords(target, search_space, max_dist=2.0):
    search_tree = KDTree(search_space)
    indices = search_tree.query_ball_point(target, max_dist)
    logger.debug(f"Search found indices {indices} within {max_dist} of {target}")
    return indices


def gamma_to_fwhm(gamma, alpha):
    """ Convert gamma to full-width half-maximum (FWHM). """
    return 2 * gamma * np.sqrt(2**(1/alpha)-1)

def discrete_moffat_integral(par, fittype, step_size=1.0):
    
    # Calculate the start and end points
    grid_size = 10
    half_size = grid_size // 2
    x_start, x_end = -half_size + step_size / 2, half_size - step_size / 2
    y_start, y_end = half_size - step_size / 2, -half_size + step_size / 2
    x_coords = np.arange(x_start, x_end + step_size, step_size)
    y_coords = np.arange(y_start, y_end - step_size, -step_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Calculate flux in each pixel
    if fittype == 'circ':
        pixel_fluxes = Moffat2D.evaluate(grid_x, grid_y, par[2], 0, 0, par[3], par[4])
    elif fittype == 'ellip':
        pixel_fluxes = MoffatElliptical2D.evaluate(grid_x, grid_y, par[2], 0, 0, par[3], par[4], par[5], par[6])
    pixel_fluxes *= step_size**2
    return np.sum(pixel_fluxes)

def process_par(par, label, fittype):
    if fittype == 'circ':
        fwhm = gamma_to_fwhm(par[3], par[4])
        logger.debug(f"{label} Moffat fit parameters: \namplitude = {par[2]}, gamma = {par[3]}, alpha = {par[4]}, bkgd = {par[5]}")
    elif fittype == 'ellip':
        fwhm1 = gamma_to_fwhm(par[3], par[6])
        fwhm2 = gamma_to_fwhm(par[4], par[6])
        fwhm = (fwhm1 + fwhm2)/2
        logger.debug(f"{label} Moffat fit parameters: \namplitude = {par[2]}, gamma1 = {par[3]}, gamma2 = {par[4]}, phi = {par[5]}, alpha = {par[6]}, bkgd = {par[7]}")
    logger.info(f"{label} FWHM = {fwhm}")
    return fwhm



def check_integrals(phot_data):
    integ = moffat_integral((phot_data['amplitude_2_fit']), phot_data['gamma_2_fit'], phot_data['alpha_2_fit'])
    phot_data.add_column(np.array(integ), name='integral of moffat psf')
    phot_data.add_column(np.array(integ*phot_data['amplitude_4_fit']), name='integral * amp_4')
    return

def moffat_integral(amplitude, gamma, alpha):
    return amplitude * np.pi * gamma**2 / (alpha - 1)


def consolidate_groups(phot_data, preserve=[]):
    new_data = phot_data.copy()
    group_data = new_data[new_data['group_size'] > 1]
    group_data = group_data.copy()[~np.isin(group_data['group_id'], preserve)]
    
    group_ids = list(sorted(set(group_data['group_id'])))
    logger.info(f"Consolidating groups {group_ids}")
    logger.info(f"Preserving groups {preserve}")
    
    new_data = phot_data.copy()[~np.isin(phot_data['group_id'], group_ids)]
    
    for id in group_ids:
        group = group_data[group_data['group_id'] == id]
        new_row = {
            'id': id,
            'group_id': id,
            'group_size': group['group_size'][0],
            'iter_detected': group['iter_detected'][-1],
            'local_bkg': np.average(group['local_bkg'], weights=abs(group['flux_fit'])),
            'x_init': np.average(group['x_init'], weights=abs(group['flux_fit'])),
            'y_init': np.average(group['y_init'], weights=abs(group['flux_fit'])),
            'flux_init': np.average(group['flux_init'], weights=abs(group['flux_fit'])),
            'x_fit': np.average(group['x_fit'], weights=abs(group['flux_fit'])),
            'y_fit': np.average(group['y_fit'], weights=abs(group['flux_fit'])),
            'flux_fit': np.sum(group['flux_fit']),
            'x_err': np.sum(group['x_err']),
            'y_err': np.sum(group['y_err']),
            'flux_err': np.sum(group['flux_err']),
            'airmass': group['airmass'][0],
            'npixfit': np.sum(group['npixfit']),
            'qfit': np.average(group['qfit'], weights=abs(group['flux_fit'])),
            'cfit': np.average(group['cfit'], weights=abs(group['flux_fit'])),
            'flags': np.bitwise_or.reduce(group['flags'])
        }
        new_data.add_row(new_row)
    
    new_data.sort('group_id')
    table_str = log_astropy_table(new_data)
    logger.debug(f"Consolidated sources: \n{table_str}")
    return new_data


    