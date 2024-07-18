
######################################################################################
########  Create field/contour plots of PSF fit parameters for spacer widths  ########
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from loess.loess_2d import loess_2d

from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps_bulk
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single
from nickelpipeline.psf_analysis.moffat.model_psf import FitEllipticalMoffat2D, FitMoffat2D, make_ellipse

from nickelpipeline.convenience.dir_nav import unzip_directories, categories_from_conditions
from nickelpipeline.astrometry.nickel_data import plate_scale_approx    # For the Nickel Telescope original camera


def fit_field_by_category(path_list, condition_tuples, frac=0.5, verbose=False, 
                          subplot_size=70, include_smooth=True, include_srcs=False):
    """
    Display plots of PSF Moffat fit models as they vary across a CCD field, categorized
    by specific conditions. By default, shows a smoothed estimate of the fit at a grid 
    of points. Can also display the actual source fits.
    
    Args:
        path_list (list): List of paths (directories or files) to unzip.
        condition_tuples (list of tuples): Conditions for categorizing images.
        frac (float): Fraction parameter for Loess smoothing. [0.2, 0.8] is typical.
        verbose (bool): If True, print detailed output during processing.
        subplot_size (int): Size of the area sampled for one PSF fit.
        include_smooth (bool): Whether to include smoothed fit field map.
        include_srcs (bool): Whether to include source fit field map.
    """
    # Gather files from all directories, sort into dict = {spacer_width: file_list}
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        
        # Convert float category of form 1.375 to a string 1_375
        category_str = f"{str(category)[0]}_{str(category)[2:]}"
        if verbose: 
            print(f"Working on fit map for category {category}")
        
        # Create a figure for plotting
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        background_img = np.zeros((1024,1024))
        ax.imshow(background_img, origin='lower')
        plt.suptitle(f"Moffat Fits for Spacer Width {category}")
        
        # Get source coordinates and parameters
        source_coords, source_pars, img_nums = get_source_pars(file_list, 
                                                     category_str, verbose)
        
        #################################
        # Plot the smoothed fit field map
        #################################
        if include_smooth:
            # Get smoothed parameters
            smooth_pars, _, _ = get_smoothed_pars(source_coords, source_pars, frac=frac,
                                                  subplot_size=subplot_size, verbose=verbose)
            if verbose: 
                print("Starting smoothed fit field map plotting")

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
        
        #################################
        # Plot the sources' fit field map
        #################################
        
        if include_srcs:
            if verbose: 
                print(f"Working on sources plot for category {category}")
            
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

def param_contour_by_category(param_type, path_list, condition_tuples,
                              frac=0.5, verbose=False, include_smooth=True, include_srcs=False):
    """
    Plot contour maps of a Moffat fit parameter (FWHM, eccentricity, rotation angle phi), 
    categorized by certain conditions.
    
    Args:
        param_type (str): Type of parameter to plot ('fwhm', 'phi', 'ecc').
        path_list (list): List of paths (directories or files) to unzip.
        condition_tuples (list of tuples): Conditions for categorizing images.
        frac (float): Fraction parameter for Loess smoothing.
        verbose (bool): If True, print detailed output during processing.
        include_smooth (bool): Whether to include smoothed parameter contour graph.
        include_srcs (bool): Whether to include source parameter contour graph.
    """
    # Gather files from all directories, sort into dict = {spacer_width: file_list}
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        
        # Convert float category of form 1.375 to a string 1_375
        category_str = f"{str(category)[0]}_{str(category)[2:]}"
        if verbose: 
            print(f"Working on {param_type} map for category {category}")
        
        # Create a figure for plotting
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Get source coordinates and parameters
        source_coords, source_pars, img_nums = get_source_pars(file_list, category_str, verbose)

        ###########################################
        # Plot the smoothed parameter contour graph
        ###########################################
        if include_smooth:
            # Extract source coordinates and parameters
            subplot_size=15
            centroid_xs, centroid_ys = zip(*source_coords)
            centroid_xs = np.array(centroid_xs)
            centroid_ys = np.array(centroid_ys)
            
            # Create a grid for at which to sample the smoothed parameters
            border = int(subplot_size / 2)
            grid_x, grid_y = np.mgrid[border:1024-border:subplot_size, 
                                      border:1024-border:subplot_size]
            
            source_param_list, color_range, title = get_param_list(param_type, source_pars, centroid_xs.shape, img_nums)
            
            param_list, _ = loess_2d(centroid_xs, centroid_ys, source_param_list, xnew=grid_x.flatten(),
                                        ynew=grid_y.flatten(), frac=frac)
            param_list = param_list.reshape(grid_x.shape)
            
            # # Get smoothed parameters
            # smooth_pars, grid_x, grid_y = get_smoothed_pars(source_coords, source_pars, frac=frac, 
            #                                         subplot_size=15, verbose=verbose)
    
            # # Get parameter list, range of colors, and title for plotting
            # param_list, color_range, title = get_param_list(param_type, smooth_pars, grid_x.shape, img_nums)
            
            # Define the colors & range of contour levels
            colors = ["#cd0000", "#cb4000", "#c97f00", "#c7bc00", "#91c400", "#52c200", 
                    "#00bc62", "#00ba9c", "#009cb8", "#0061b6", "#1100b1", "#4800af", "#7e00ad"]
            levels = np.linspace(color_range[0], color_range[1], len(colors))

            # Create contour plot
            cp = ax.contourf(grid_x, grid_y, param_list, levels=levels, colors=colors)
            ax.set_title(f'{title} Graph - {category_str}')
        
        ##############################
        # Plot the sources' parameters
        ##############################
        if include_srcs:
            if verbose: 
                print(f"Working on sources plot for category {category}")
            
            # Extract coordinates and parameter list
            x, y = zip(*source_coords)
            x = np.array(x)
            y = np.array(y)
            param_list, color_range, title = get_param_list(param_type, source_pars, x.shape, img_nums)

            # Define the colors & range of contour levels
            colors = ["#cd0000", "#cb4000", "#c97f00", "#c7bc00", "#91c400", "#52c200", 
                      "#00bc62", "#00ba9c", "#009cb8", "#0061b6", "#1100b1", "#4800af"]
            levels = np.linspace(color_range[0], color_range[1], len(colors))
            cmap_custom = ListedColormap(colors)

            # Create a scatter plot, color coded by param_list value
            ax.set_title(f'{title} Graph - {category_str}')
            jitter_x = np.random.normal(scale=6, size=len(x))
            jitter_y = np.random.normal(scale=6, size=len(y))
            ax.scatter(x+jitter_x, y+jitter_y, s=15, c=param_list, cmap=cmap_custom, 
                                vmin=levels[0], vmax=levels[-1], alpha=1.0,
                                linewidths=0.7, edgecolors='k')
        
        plt.colorbar(cp, ax=ax)
        plt.show()

def get_param_list(param_type, smooth_pars, shape, img_nums=None):
    """
    Generate the parameter list, color range, and title for contour plotting.
    
    Args:
        param_type (str): Type of parameter ('fwhm', 'phi', 'ecc').
        smooth_pars (ndarray): Smoothed parameters.
        shape (ndarray): Shape to output param_list
        img_nums (ndarray): Image number for each source
    
    Returns:
        param_list (ndarray): List of parameter values.
        color_range (list): Range of colors for plotting.
        title (str): Title for the plot.
    """
    if param_type == 'fwhm':
        # Calculate average FWHM (between semi-major and minor axes)
        param_list = (FitMoffat2D.to_fwhm(smooth_pars[:,3], smooth_pars[:,6]) + 
                      FitMoffat2D.to_fwhm(smooth_pars[:,4], smooth_pars[:,6]))/2 * plate_scale_approx
        color_range = [1.4, 3.1]    # Optimized for Nickel 06-26-24 data
        title = "FWHM (arcsec)"
    elif param_type == 'fwhm residuals':
        fwhm_list = (FitMoffat2D.to_fwhm(smooth_pars[:,3], smooth_pars[:,6]) + 
                     FitMoffat2D.to_fwhm(smooth_pars[:,4], smooth_pars[:,6]))/2
        print(len(fwhm_list))
        num_imgs = img_nums[-1]
        mins = [np.min(fwhm_list[img_nums==i]) for i in range(num_imgs)]
        param_list = np.array([mins[i] for i in img_nums]) * plate_scale_approx
        color_range = [0.0, 1.5]
        title = "FWHM Residuals (arcsec)"
    elif param_type == 'phi':
        # Convert phi rotation angle from messy original phi
        param_list = np.array([FitEllipticalMoffat2D.get_nice_phi(smooth_par) 
                 for smooth_par in smooth_pars])
        color_range = [-45., 45.]
        title = "Phi Rotation Angle (deg)"
    elif param_type == 'ecc':
        # Calculate eccentricity
        param_list = []
        for smooth_par in smooth_pars:
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

def get_source_pars(path_list, category_str=None, verbose=False):
    """
    Extract source coordinates and fit parameters from image data.
    
    Args:
        path_list (list): List of paths (directories or files) to unzip.
        category_str (str): Category string for identifying the path to data
        verbose (bool): If True, print detailed output during processing.
    
    Returns:
        source_coords (ndarray): Array of source coordinates.
        source_pars (ndarray): Array of source parameters.
                (x0, y0, amplitude, gamma1, gamma2, phi, alpha, background)
    """
    # Unzip directories to get image files
    images = unzip_directories(path_list, output_format='Path')
    
    # Generate stamps (image of sources) for image data
    generate_stamps_bulk(images, category_str, verbose=verbose)
        
    # Fit PSF models and get source coordinates and parameters
    source_coords, source_pars, img_nums = fit_psf_single(category_str, len(images))
    return source_coords, source_pars, img_nums


def get_smoothed_pars(source_coords, source_pars, frac=0.5,
                      verbose=False, subplot_size=70):
    """
    Apply Loess smoothing to the source parameters and return a grid sampling.
    
    Args:
        source_coords (list of tuples): List of source coordinates.
        source_pars (ndarray): Array of source parameters.
        frac (float): Fraction parameter for Loess smoothing.
        verbose (bool): If True, print detailed output during processing.
        subplot_size (int): Size of the subplot.
    
    Returns:
        smooth_pars (ndarray): Smoothed parameters.
                (x0, y0, amplitude, gamma1, gamma2, phi, alpha, background)
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
    
    if verbose: 
        print(f"{len(source_pars)} stars being used for Loess")
    
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



# Suppress warnings
#warnings.filterwarnings('ignore')