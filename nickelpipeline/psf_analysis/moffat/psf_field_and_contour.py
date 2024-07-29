
######################################################################################
########  Create field/contour plots of PSF fit parameters for spacer widths  ########
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import logging

from loess.loess_2d import loess_2d

from nickelpipeline.psf_analysis.moffat.model_psf import FitEllipticalMoffat2D, FitMoffat2D, make_ellipse
from nickelpipeline.psf_analysis.moffat.moffat_fitting import get_source_pars

from nickelpipeline.convenience.dir_nav import unzip_directories, categories_from_conditions
from nickelpipeline.convenience.graphs import smooth_contour, scatter_sources
from nickelpipeline.convenience.nickel_data import plate_scale_approx    # For the Nickel Telescope original camera

logger = logging.getLogger(__name__)

def fit_field_by_category(path_list, condition_tuples, frac=0.5, 
                          subplot_size=70, include_smooth=True, include_srcs=False):
    """
    Display plots of PSF Moffat fit models as they vary across a CCD field, categorized
    by specific conditions. By default, shows a smoothed estimate of the fit at a grid 
    of points. Can also display the actual source fits.
    
    Args:
        path_list (list): List of paths (directories or files) to unzip.
        condition_tuples (list of tuples): Conditions for categorizing images.
        frac (float): Fraction parameter for Loess smoothing. [0.2, 0.8] is typical.
        subplot_size (int): Size of the area sampled for one PSF fit.
        include_smooth (bool): Whether to include smoothed fit field map.
        include_srcs (bool): Whether to include source fit field map.
    """
    # Gather files from all directories, sort into dict = {spacer_width: file_list}
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        
        # Convert float category of form 1.375 to a string 1_375
        category_str = str(category).replace('.', '_')
        logger.info(f"Working on fit map for category {category}")
        
        # Create a figure for plotting
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        background_img = np.zeros((1024,1024))
        ax.imshow(background_img, origin='lower')
        plt.suptitle(f"Moffat Fits for Spacer Width {category}")
        
        # Get source coordinates and parameters
        source_coords, source_pars, _ = get_source_pars(file_list, category_str)
        
        #------------------------------------------------------------
        # Plot the smoothed fit field map
        #------------------------------------------------------------
        if include_smooth:
            logger.info("Starting smoothed fit field map plotting")
            # Get smoothed parameters
            smooth_pars, _, _ = get_smoothed_pars(source_coords, source_pars, frac=frac,
                                                  subplot_size=subplot_size)
            
            # Plots ellipses representing the estimated fit at points on a grid
            for smooth_par in smooth_pars:
                # Calculate FWHM and plot ellipse
                fwhm1 = FitMoffat2D.to_fwhm(smooth_par[3], smooth_par[6])
                fwhm2 = FitMoffat2D.to_fwhm(smooth_par[4], smooth_par[6])
                scale = subplot_size // 15
                ell_x, ell_y = make_ellipse(fwhm1 * scale, fwhm2 * scale, smooth_par[5])
                ell_x += smooth_par[0]
                ell_y += smooth_par[1]
                
                ax.plot(ell_x, ell_y, color='g', lw=2, label='Smoothed Fit')
        
        #------------------------------------------------------------
        # Plot the sources' fit field map
        #------------------------------------------------------------
        if include_srcs:
            logger.info(f"Working on sources plot for category {category}")
            
            # Plots the actual fit for every source detected in images
            for coord, source_par in zip(source_coords, source_pars):
                # Calculate FWHM and plot ellipse for sources
                fwhm1 = FitMoffat2D.to_fwhm(source_par[3], source_par[6])
                fwhm2 = FitMoffat2D.to_fwhm(source_par[4], source_par[6])
                scale = subplot_size // 15
                ell_x, ell_y = make_ellipse(fwhm1 * scale, fwhm2 * scale, source_par[5])
                
                ax.plot(ell_x + coord[0], ell_y + coord[1], color='r', 
                        lw=0.7, label='Sources Fit')

        plt.show()


def param_graph_by_category(param_type, path_list, condition_tuples, frac=0.5,
                            include_smooth=True, include_srcs=False):
    """
    Plot contour maps of a Moffat fit parameter (FWHM, eccentricity, rotation angle phi), 
    categorized by certain conditions.
    
    Args:
        param_type (str): Type of parameter to plot ('fwhm', 'phi', 'ecc').
        path_list (list): List of paths (directories or files) to unzip.
        condition_tuples (list of tuples): Conditions for categorizing images.
        frac (float): Fraction parameter for Loess smoothing.
        include_smooth (bool): Whether to include smoothed parameter contour graph.
        include_srcs (bool): Whether to include source parameter contour graph.
    """
    # Gather files from all directories, sort into dict = {spacer_width: file_list}
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        single_param_graph(param_type, file_list, category, frac,
                           include_smooth, include_srcs)


def param_graph_individuals(param_type, path_list, condition_tuples, frac=0.5, 
                            include_smooth=True, include_srcs=False):
    
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        logger.info(f"Category: {category}")
        for image in file_list:
            logger.info(f"Image {image}:")
            single_param_graph(param_type, [image], str(category), frac, 
                               include_smooth, include_srcs)
        logger.info("------------------")


def single_param_graph(param_type, file_list, category, frac=0.5,
                       include_smooth=True, include_srcs=False):
    """
    Plot contour maps of a Moffat fit parameter (FWHM, eccentricity, rotation angle phi), 
    categorized by certain conditions.
    
    Args:
        param_type (str): Type of parameter to plot ('fwhm', 'phi', 'ecc').
        file_list (list): List of Fits_Simple images to graph.
        category (any): Category of images.
        frac (float): Fraction parameter for Loess smoothing.
        include_smooth (bool): Whether to include smoothed parameter contour graph.
        include_srcs (bool): Whether to include source parameter contour graph.
    """
    logger.info(f"Working on {param_type} map for category {category}")
    
    # Convert float category of form 1.375 to a string 1_375
    category_str = str(category).replace('.', '_')
    
    # Create a figure for plotting
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Get source coordinates and parameters
    source_coords, source_pars, img_nums = get_source_pars(file_list, category_str)
    
    # Extract source coordinates and parameters
    x_list, y_list = zip(*source_coords)
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    source_param_list, color_range, title = get_param_list(param_type, source_pars, x_list.shape, img_nums)

    # Plot the smoothed parameter contour graph
    if include_smooth:
        logger.info(f"Working on smoothed contour plot for category {category}")
        ax, cp = smooth_contour(x_list, y_list, source_param_list, 
                                color_range, ax, frac, title, category_str)
    
    # Plot the sources' parameters
    if include_srcs:
        logger.info(f"Working on sources plot for category {category}")
        ax, cmap = scatter_sources(x_list, y_list, source_param_list, 
                                color_range, ax, title, category_str)
    
    try:
        plt.colorbar(cp, ax=ax)
    except (UnboundLocalError, RuntimeError):
        plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.show()


def get_param_list(param_type, pars, shape, img_nums=None):
    """
    Generate the parameter list, color range, and title for contour plotting.
    
    Args:
        param_type (str): Type of parameter ('fwhm', 'phi', 'ecc').
        pars (ndarray): Fit parameters (list of par)
        shape (ndarray): Shape to output param_list
        img_nums (ndarray): Image number for each source
    
    Returns:
        param_list (ndarray): List of parameter values.
        color_range (list): Range of colors for plotting.
        title (str): Title for the plot.
    """
    if param_type == 'fwhm':
        # Calculate average FWHM (between semi-major and minor axes)
        param_list = (FitMoffat2D.to_fwhm(pars[:,3], pars[:,6]) + 
                      FitMoffat2D.to_fwhm(pars[:,4], pars[:,6]))/2 * plate_scale_approx
        color_range = [1.5, 2.7]    # Optimized for Nickel 06-26-24 data
        title = "FWHM (arcsec)"
    elif param_type == 'fwhm residuals':
        fwhm_list = (FitMoffat2D.to_fwhm(pars[:,3], pars[:,6]) + 
                     FitMoffat2D.to_fwhm(pars[:,4], pars[:,6]))/2
        mins = {img_num: np.min(fwhm_list[img_nums==img_num]) 
                for img_num in list(set(img_nums))}
        param_list = np.array([fwhm_list[i]-mins[img_num] 
                               for i, img_num in enumerate(img_nums)]) * plate_scale_approx
        color_range = [0.0, 0.36]
        title = "FWHM Residuals (arcsec)"
    elif param_type == 'phi':
        # Convert phi rotation angle from messy original phi
        param_list = np.array([FitEllipticalMoffat2D.get_nice_phi(smooth_par) 
                 for smooth_par in pars])
        color_range = [-45., 45.]
        title = "Phi Rotation Angle (deg)"
    elif param_type == 'ecc':
        # Calculate eccentricity
        param_list = []
        for smooth_par in pars:
            fwhm1 = FitMoffat2D.to_fwhm(smooth_par[3], smooth_par[6])
            fwhm2 = FitMoffat2D.to_fwhm(smooth_par[4], smooth_par[6])
            param_list.append(np.sqrt(np.abs(fwhm1**2 - fwhm2**2)) / max(fwhm1, fwhm2))
        param_list = np.array(param_list)
        color_range = [0.29, 0.65]   # Optimized for Nickel 06-26-24 data
        title = "Eccentricity"
    else:
        raise ValueError("Input param_type must be 'fwhm' or 'phi'")
    
    param_list = param_list.reshape(shape)
    return param_list, color_range, title


def get_smoothed_pars(source_coords, source_pars, frac=0.5,
                      subplot_size=70):
    """
    Apply Loess smoothing to the source parameters and return a grid sampling.
    
    Args:
        source_coords (list of tuples): List of source coordinates.
        source_pars (ndarray): Array of source parameters.
        frac (float): Fraction parameter for Loess smoothing.
        subplot_size (int): Size of the subplot.
    
    Returns:
        smooth_pars (ndarray): Smoothed parameters. (x0, y0, amplitude, gamma1, gamma2, phi, alpha, background)
        grid_x (ndarray): Grid x-coordinates.
        grid_y (ndarray): Grid y-coordinates.
    """
    # Extract source coordinates and parameters
    centroid_xs, centroid_ys = zip(*source_coords)
    xs, ys, amplitudes, gamma1s, gamma2s, phis, alphas, backgrounds = zip(*source_pars)
    # Convert phis to nice phi (angle between semi-major axis & +x-axis, degrees)
    phis = [FitEllipticalMoffat2D.get_nice_phi(par) for par in source_pars]
    
    # Take average for unimportant parameters instead of smoothing
    avg_amplitude = np.mean(amplitudes)
    avg_alpha = np.mean(alphas)
    avg_background = np.mean(backgrounds)

    # Convert lists to NumPy arrays
    (centroid_xs, centroid_ys, 
     gamma1s, gamma2s, phis) = [np.array(lst) for lst in [centroid_xs, centroid_ys, 
                                                          gamma1s, gamma2s, phis]]
    
    # Create a grid for at which to sample the smoothed parameters
    border = int(subplot_size / 2)
    grid_x, grid_y = np.mgrid[border:1024-border:subplot_size, 
                              border:1024-border:subplot_size]
    
    logger.info(f"{len(source_pars)} stars being used for Loess")
    
    # Apply Loess smoothing
    smooth_gamma1s, _ = loess_2d(centroid_xs, centroid_ys, gamma1s, xnew=grid_x.flatten(),
                                  ynew=grid_y.flatten(), frac=frac)
    smooth_gamma2s, _ = loess_2d(centroid_xs, centroid_ys, gamma2s, xnew=grid_x.flatten(),
                                  ynew=grid_y.flatten(), frac=frac)
    smooth_phis, _ = loess_2d(centroid_xs, centroid_ys, phis, xnew=grid_x.flatten(),
                                  ynew=grid_y.flatten(), frac=frac)
    smooth_phis = [FitEllipticalMoffat2D.get_orig_phi(gamma1, gamma2, phi) 
                   for gamma1, gamma2, phi in zip(smooth_gamma1s, smooth_gamma2s, smooth_phis)]
    
    # Combine smoothed parameters into FitEllipticalMoffat2D's self.par format
    smooth_pars = np.array([
        [x, y, avg_amplitude, g1, g2, phi, avg_alpha, avg_background]
        for x, y, g1, g2, phi in zip(grid_x.flatten(), grid_y.flatten(), 
                                     smooth_gamma1s, smooth_gamma2s, smooth_phis)
    ])
    
    return smooth_pars, grid_x, grid_y

