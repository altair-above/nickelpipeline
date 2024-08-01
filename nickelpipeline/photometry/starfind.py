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

from nickelpipeline.photometry.moffat_model_photutils import MoffatElliptical2D

from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single, fit_psf_stack, psf_plot


logger = logging.getLogger(__name__)
np.set_printoptions(edgeitems=100)


def analyze_sources(image, plot=False, thresh=10.0, mode='all', fittype='circular'):
    
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
    img = image.masked_array
    img.data[:,bad_columns] = 0
    
    #----------------------------------------------------------------------
    # Use a Moffat fit to find & fit initial sources
    #----------------------------------------------------------------------
    
    # Create output directories
    img_name = image.path.stem.split('_')[0]
    proc_dir = Path('.').resolve() / "proc_files"
    Path.mkdir(proc_dir, exist_ok=True)
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
    logger.debug(f"Sources Found (Iter 1): \n{log_astropy_table(sources)}")
    # sources = filter_off_ccd(sources, xname='xcentroid', yname='ycentroid')
    
    # Fit PSF models and get source coordinates and parameters
    source_coords, source_fits, _ = fit_psf_single(base, 1, fittype=fittype, sigma_clip=False)
    source_pars = np.array([fit.par for fit in source_fits])
    
    avg_par = np.mean(source_pars, axis=0)
    avg_fwhm = process_par(avg_par, 'Averaged-out', fittype=fittype)
    
    brightest = np.array(sorted(source_pars, key=lambda coord: coord[2])[:5])
    clip_avg_par = np.mean(brightest, axis=0)
    clip_avg_fwhm = process_par(clip_avg_par, 'Clipped Avg', fittype=fittype)
    
    psf_file = Path(f'{str(base)}.psf.fits').resolve()  # PSF info stored here
    stack_par = fit_psf_stack(base, 1, fittype=fittype, ofile=psf_file).par
    stack_fwhm = process_par(stack_par, 'Stack', fittype=fittype)

    fit_par = stack_par
    fit_fwhm = stack_fwhm
    # fit_par = clip_avg_par
    # fit_fwhm = clip_avg_fwhm
    # fit_par = avg_par
    # fit_fwhm = avg_fwhm
    
    init_phot_data = Table()
    init_phot_data.add_column(source_coords[:,0], name='x_fit')
    init_phot_data.add_column(source_coords[:,1], name='y_fit')
    flux_integrals = [discrete_moffat_integral(par, fittype=fittype, step_size=0.5) for par in source_pars]
    init_phot_data.add_column(flux_integrals, name='flux_fit')
    init_phot_data.add_column(list(range(len(source_pars))), name='group_id')
    init_phot_data.add_column([1] * len(source_pars), name='group_size')
    
    if plot:
        plot_sources(image, init_phot_data, fit_fwhm)
    
    img.data[:,bad_columns] = fit_par[5]

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
    
    logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")
        
    if plot:
        plot_groups(phot_data, source_coords, source_fits, base)
        
        plot_sources(image, phot_data, fit_fwhm)
    
    return phot_data


def plot_groups(phot_data, source_coords, source_fits, base):
    bad_data = phot_data[phot_data['group_size'] > 1]
    group_ids = list(sorted(set(bad_data['group_id'])))
    for id in group_ids:
        logger.info(f"Group {id} has multiple fitted PSF's: displaying original source")
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
            plot_file = Path(f'{str(base)}_src{index}.psf.pdf').resolve()
            psf_plot(plot_file, matching_fit, show=True, plot_fit=False)
        

def match_coords(target, search_space, max_dist=2.0):
    search_tree = KDTree(search_space)
    indices = search_tree.query_ball_point(target, max_dist)
    logger.debug(f"Search found indices {indices} within {max_dist} of {target}")
    return indices


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
    good_apertures.plot(color='m', lw=1.5, alpha=0.9)
    bad_apertures.plot(color='r', lw=1.5, alpha=0.5)
    
    # Annotate good sources with flux_fit values
    for i in range(len(good_phot_data)):
        plt.text(x_good[i], y_good[i]+17, f'{good_phot_data["flux_fit"][i]:.0f}', color='white', fontsize=8, ha='center', va='center')
    
    group_ids = set(bad_phot_data['group_id'])
    for id in group_ids:
        group = bad_phot_data[bad_phot_data['group_id'] == id]
        group_x = np.mean(group['x_fit']) + 15
        group_y = np.mean(group['y_fit'])
        for i in range(len(group)):
            plt.text(group_x, group_y+(i-1)*20, f'{group["group_id"][i]:.0f}:{group["iter_detected"][i]:.0f}: {group["flux_fit"][i]:.0f}', color='red', fontsize=8, ha='left', va='center')
    
    plt.gcf().set_dpi(300)
    plt.show()


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
    if fittype == 'circular':
        pixel_fluxes = Moffat2D.evaluate(grid_x, grid_y, par[2], 0, 0, par[3], par[4])
    elif fittype == 'elliptical':
        pixel_fluxes = MoffatElliptical2D.evaluate(grid_x, grid_y, par[2], 0, 0, par[3], par[4], par[5], par[6])
    pixel_fluxes *= step_size**2
    return np.sum(pixel_fluxes)

def process_par(par, label, fittype):
    if fittype == 'circular':
        fwhm = gamma_to_fwhm(par[3], par[4])
        logger.debug(f"{label} Moffat fit parameters: \namplitude = {par[2]}, gamma = {par[3]}, alpha = {par[4]}, bkgd = {par[5]}")
    elif fittype == 'elliptical':
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





# result = discrete_moffat_integral(0.0381, 4.776, 3.728)







# import numpy as np
# import logging

# from astropy.modeling.fitting import LevMarLSQFitter
# from astropy.modeling.functional_models import Moffat2D
# from photutils.detection import IRAFStarFinder
# from photutils.aperture import CircularAperture
# from photutils.psf import IterativePSFPhotometry, make_psf_model
# from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
# from photutils.psf import IntegratedGaussianPRF, SourceGrouper

# from pathlib import Path
# from astropy.table import Table
# from matplotlib import pyplot as plt
# from astropy.visualization import ZScaleInterval

# from scipy.spatial import KDTree

# from nickelpipeline.convenience.fits_class import Fits_Simple
# from nickelpipeline.convenience.nickel_data import bad_columns, ccd_shape
# from nickelpipeline.convenience.log import log_astropy_table

# from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
# from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single, fit_psf_stack, psf_plot


# logger = logging.getLogger(__name__)
# np.set_printoptions(edgeitems=100)


# def analyze_sources(image, plot=False, thresh=10.0, mode='all', fittype='elliptical'):
    
#     # thresh=10.0
#     local_bkg_range=(15,25)
    
#     if not isinstance(image, Fits_Simple):
#         image = Fits_Simple(image)
#     logger.debug(f"analyze_sources() called on image {image.filename}")

#     new_mask = image.mask.copy()
#     new_mask[:5, :] = True
#     new_mask[-5:, :] = True
#     new_mask[:, :5] = True
#     new_mask[:, -5:] = True
#     image.mask = new_mask
#     img = image.masked_array
#     img.data[:,bad_columns] = 0
    
#     #----------------------------------------------------------------------
#     # Use a Moffat fit to find & fit initial sources
#     #----------------------------------------------------------------------
    
#     # Create output directories
#     img_name = image.path.stem.split('_')[0]
#     proc_dir = Path('.').resolve() / "proc_files"
#     Path.mkdir(proc_dir, exist_ok=True)
#     proc_subdir = proc_dir / 'circular'
#     Path.mkdir(proc_subdir, exist_ok=True)
#     base_parent = proc_subdir / img_name
#     Path.mkdir(base_parent, exist_ok=True)
#     base = proc_subdir / img_name / img_name
    
#     # Generate stamps (image of sources) for image data
#     source_data = generate_stamps([image], output_base=base, thresh=thresh)
    
#     # Convert source data into Astropy table
#     column_names = ['chip', 'id', 'xcentroid', 'ycentroid', 'bkg', 'kron_radius', 'raw_flux', 'flux', '?']
#     sources = Table(source_data, names=column_names)
#     logger.debug(f"Sources Found (Iter 1): \n{log_astropy_table(sources)}")
#     # sources = filter_off_ccd(sources, xname='xcentroid', yname='ycentroid')
    
#     # Fit PSF models and get source coordinates and parameters
#     source_coords, source_fits, _ = fit_psf_single(base, 1, fittype='circular')
#     source_pars = np.array([fit.par for fit in source_fits])
    
#     avg_par = np.mean(source_pars, axis=0)
#     avg_fwhm = gamma_to_fwhm(avg_par[3], avg_par[4])
#     logger.debug(f"Averaged-out Moffat fit parameters: \namplitude = {avg_par[2]}, gamma = {avg_par[3]}, alpha = {avg_par[4]}, background = {avg_par[5]}")
#     logger.info(f"Averaged-out FWHM = {avg_fwhm}")
    
#     brightest = np.array(sorted(source_pars, key=lambda coord: coord[2])[:3])
#     logger.debug(brightest)
#     lim_avg_par = np.mean(brightest, axis=0)
#     lim_avg_fwhm = gamma_to_fwhm(lim_avg_par[3], lim_avg_par[4])
#     logger.debug(f"lim_avg Moffat fit parameters: \namplitude = {lim_avg_par[2]}, gamma = {lim_avg_par[3]}, alpha = {lim_avg_par[4]}, background = {lim_avg_par[5]}")
#     logger.info(f"lim_avg FWHM = {lim_avg_fwhm}")
    
#     psf_file = Path(f'{str(base)}.psf.fits').resolve()  # PSF info stored here
#     stack_fit = fit_psf_stack(base, 1, fittype='circular', ofile=psf_file)
#     stack_par = stack_fit.par
#     stack_fwhm = gamma_to_fwhm(stack_par[3], stack_par[4])
#     logger.debug(f"Stack Moffat fit parameters: \namplitude = {stack_par[2]}, gamma = {stack_par[3]}, alpha = {stack_par[4]}, background = {stack_par[5]}")
#     logger.info(f"Stack FWHM = {stack_fwhm}")

#     fit_par = stack_par
#     fit_fwhm = stack_fwhm
    
#     # fit_par = lim_avg_par
#     # fit_fwhm = lim_avg_fwhm
    
#     # fit_par = avg_par
#     # fit_fwhm = avg_fwhm
    
#     init_phot_data = Table()
#     init_phot_data.add_column(source_coords[:,0], name='x_fit')
#     init_phot_data.add_column(source_coords[:,1], name='y_fit')
#     init_phot_data.add_column(moffat_integral(source_pars[:,2], source_pars[:,3], source_pars[:,4]), name='flux_fit')
#     init_phot_data.add_column([i for i in range(len(source_pars))], name='group_id')
#     init_phot_data.add_column([1 for _ in range(len(source_pars))], name='group_size')
    
#     if plot:
#         plot_sources(image, init_phot_data, fit_fwhm)
    
#     img.data[:,bad_columns] = fit_par[5]

#     #----------------------------------------------------------------------
#     # Attempt to improve the source detection by improving the FWHM estimate
#     #----------------------------------------------------------------------
#     aper_size=fit_fwhm*1.8
#     local_bkg_range=(3*fit_fwhm,6*fit_fwhm)
#     win = int(np.ceil(2*fit_fwhm))
#     if win % 2 == 0:
#         win += 1
#     bkgrms = MADStdBackgroundRMS()
#     std = bkgrms(img)

#     # Source finder
#     iraffind = IRAFStarFinder(fwhm=fit_fwhm, threshold=thresh*std, 
#                               minsep_fwhm=0.1, peakmax=55000)
#     grouper = SourceGrouper(min_separation=2*fit_fwhm)  # Grouping algorithm
#     mmm_bkg = MMMBackground()   # Background-determining function
#     local_bkg = LocalBackground(*local_bkg_range, mmm_bkg)
#     fitter = LevMarLSQFitter()  # This is the optimization algorithm
    
#     # This is the model of the PSF
#     moffat_psf = Moffat2D(gamma=fit_par[3], alpha=fit_par[4])
#     moffat_psf = make_psf_model(moffat_psf)
    
    
#     # This is the object that performs the photometry
#     phot = IterativePSFPhotometry(finder=iraffind, grouper=grouper,
#                                   localbkg_estimator=local_bkg, psf_model=moffat_psf,
#                                   fitter=fitter, fit_shape=win,
#                                   aperture_radius=aper_size, mode=mode,)
#                                   #fitter_maxiters=250)
#     # This is actually when the fitting is done
#     phot_data = phot(data=img.data, mask=img.mask,
#                      init_params=Table(sources['xcentroid', 'ycentroid', 'flux'],
#                                        names=('x_0', 'y_0', 'flux_0')))
    
#     logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")
        
#     if plot:
#         plot_groups(phot_data, source_coords, source_fits, base)
        
#         plot_sources(image, phot_data, fit_fwhm)
    
#     return phot_data


# def plot_groups(phot_data, source_coords, source_fits, base):
#     bad_data = phot_data[phot_data['group_size'] > 1]
#     group_ids = list(sorted(set(bad_data['group_id'])))
#     for id in group_ids:
#         logger.info(f"Group {id} has multiple fitted PSF's: displaying original source")
#         group = phot_data[phot_data['group_id'] == id]
#         group_x = np.median(group['x_fit'])
#         group_y = np.median(group['y_fit'])
        
#         matching_indices = match_coords((group_x, group_y), source_coords, 2.0)
#         if len(matching_indices) == 0:
#             matching_indices = match_coords((group_x, group_y), source_coords, 4.0)
#             if len(matching_indices) == 0:
#                 logger.warning("No nearby source found to display")
#         if len(matching_indices) > 1:
#             logger.info(f"Multiple nearby sources that could match this group; displaying all")
#         for index in matching_indices:
#             matching_fit = source_fits[index]
#             plot_file = Path(f'{str(base)}_src{index}.psf.pdf').resolve()
#             psf_plot(plot_file, matching_fit, show=True, plot_fit=False)
        

# def match_coords(target, search_space, max_dist=2.0):
#     search_tree = KDTree(search_space)
#     indices = search_tree.query_ball_point(target, max_dist)
#     logger.debug(f"Search found indices {indices} within {max_dist} of {target}")
#     return indices


# def plot_sources(image, phot_data, given_fwhm):
#     # bad_sources = phot_data['flags'] > 1
#     good_phot_data = phot_data[phot_data['group_size'] <= 1]
#     bad_phot_data = phot_data[phot_data['group_size'] > 1]
    
#     logger.info(f'Image {image}')
    
#     x_good = good_phot_data['x_fit']
#     y_good = good_phot_data['y_fit']
#     good_positions = np.transpose((x_good, y_good))
#     good_apertures = CircularAperture(good_positions, r=2*given_fwhm)
    
#     x_bad = bad_phot_data['x_fit']
#     y_bad = bad_phot_data['y_fit']
#     bad_positions = np.transpose((x_bad, y_bad))
#     bad_apertures = CircularAperture(bad_positions, r=2*given_fwhm)
    
#     interval = ZScaleInterval()
#     vmin, vmax = interval.get_limits(image.masked_array)
#     cmap = plt.get_cmap()
#     cmap.set_bad('r', alpha=0.5)
#     plt.figure(figsize=(12,10))
#     plt.imshow(image.masked_array, origin='lower', vmin=vmin, vmax=vmax,
#                cmap=cmap, interpolation='nearest')
#     plt.colorbar()
#     good_apertures.plot(color='m', lw=1.5, alpha=0.5)
#     bad_apertures.plot(color='r', lw=1.5, alpha=0.5)
    
#     # Annotate good sources with flux_fit values
#     for i in range(len(good_phot_data)):
#         plt.text(x_good[i], y_good[i]+17, f'{good_phot_data["flux_fit"][i]:.0f}', color='white', fontsize=8, ha='center', va='center')
    
#     group_ids = set(bad_phot_data['group_id'])
#     for id in group_ids:
#         group = bad_phot_data[bad_phot_data['group_id'] == id]
#         group_x = np.mean(group['x_fit']) + 15
#         group_y = np.mean(group['y_fit'])
#         for i in range(len(group)):
#             plt.text(group_x, group_y+(i-1)*20, f'{group["group_id"][i]:.0f}:{group["iter_detected"][i]:.0f}: {group["flux_fit"][i]:.0f}', color='red', fontsize=8, ha='left', va='center')
#     # for i in range(len(bad_phot_data)):
#     #     jitter_x = 45
#     #     jitter_y = np.random.normal(scale=35)
#     #     plt.text(x_bad[i]+jitter_x, y_bad[i]+jitter_y, f'{bad_phot_data["flux_fit"][i]:.0f}', color='red', fontsize=8, ha='center', va='center')
#     plt.gcf().set_dpi(300)
#     plt.show()


# def check_integrals(phot_data):
#     integ = moffat_integral((phot_data['amplitude_2_fit']), phot_data['gamma_2_fit'], phot_data['alpha_2_fit'])
#     print(integ)
#     phot_data.add_column(np.array(integ), name='integral of moffat psf')
#     phot_data.add_column(np.array(integ*phot_data['amplitude_4_fit']), name='integral * amp_4')
#     return


# def gamma_to_fwhm(gamma, alpha):
#     """
#     Convert gamma to full-width half-maximum (FWHM).
#     """
#     return 2 * gamma * np.sqrt(2**(1/alpha)-1)

# def moffat_integral(amplitude, gamma, alpha):
#     return amplitude * np.pi * gamma**2 / (alpha - 1)

# def discrete_moffat_integral(amplitude, gamma, alpha, step_size=1.0):
#     # Define the grid size and step size
#     grid_size = 10

#     # Calculate the start and end points
#     half_size = grid_size // 2
#     x_start, x_end = -half_size + step_size / 2, half_size - step_size / 2
#     y_start, y_end = half_size - step_size / 2, -half_size + step_size / 2
#     x_coords = np.arange(x_start, x_end + step_size, step_size)
#     y_coords = np.arange(y_start, y_end - step_size, -step_size)
#     grid_x, grid_y = np.meshgrid(x_coords, y_coords)

#     pixel_fluxes = Moffat2D.evaluate(grid_x, grid_y, amplitude, 0, 0, gamma, alpha)
#     pixel_fluxes *= step_size**2
#     print(f"total flux = {np.sum(pixel_fluxes)}")
#     return np.sum(pixel_fluxes)

# # result = discrete_moffat_integral(0.0381, 4.776, 3.728)


