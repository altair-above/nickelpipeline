import numpy as np
from pathlib import Path
import warnings
from matplotlib import pyplot as plt
from matplotlib import cm
from loess.loess_2d import loess_2d

from nickelpipeline.psf_analysis.gaussian.calc_fwhm import calc_fwhm

from nickelpipeline.convenience.dir_nav import categories_from_conditions, unzip_directories
from nickelpipeline.convenience.graphs import smooth_contour, scatter_sources
from nickelpipeline.convenience.nickel_data import plate_scale_approx
from nickelpipeline.astrometry.plate_scale import avg_plate_scale



# def single_fwhm_contour(directories, files=None, title="", frac=0.3, verbose=False):
    
#     images = unzip_directories(directories, files, output_format='Fits_Simple')
    
#     # Collect all coordinates and FWHMs in numpy arrays
#     all_x, all_y, all_residuals = zip(*(calc_fwhm(image, mode='fwhm residuals', verbose=verbose) 
#                                         for image in images))
#     all_x = np.concatenate(all_x)
#     all_y = np.concatenate(all_y)
#     all_residuals = np.concatenate(all_residuals)
    
#     plate_scale = avg_plate_scale(directories, files=files, verbose=verbose, fast=True)
#     all_residuals = all_residuals * plate_scale
    
#     # Create grid for interpolation
#     grid_x, grid_y = np.mgrid[0:1024:3, 0:1024:3]
    
#     # Interpolate data using Loess smoothing - computationally difficult
#     if verbose:
#         print('loess_2d beginning')
#     flat_z, wout = loess_2d(all_x, all_y, all_residuals, xnew=grid_x.flatten(),
#                       ynew=grid_y.flatten(), frac=frac)
#     if verbose:
#         print('loess_2d done')
#     grid_z = flat_z.reshape(grid_x.shape)
    
#     # Define the colors & range of contour levels
#     colors = ["#cc0018", "#cd0000", "#cb4000", "#c97f00", "#c7bc00", "#91c400", "#52c200", 
#               "#00bc62", "#00ba9c", "#009cb8", "#0061b6", "#1100b1", "#4800af", "#7e00ad"]
#     levels = np.linspace(0.0, 1.3, len(colors))

#     # Plot contour map
#     plt.figure()
#     cp = plt.contourf(grid_x, grid_y, grid_z, levels=levels, colors=colors)
#     plt.colorbar(cp)
#     plt.title(f'FWHM Residuals (arcsec) Contour Map - {title}')
#     plt.xlabel('X (pixels)')
#     plt.ylabel('Y (pixels)')
#     plt.show()
    

def param_graph_by_category(param_type, path_list, condition_tuples, frac=0.3, 
                            verbose=False, include_smooth=True, include_srcs=False):
    
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        single_param_graph(param_type, file_list, str(category), frac, 
                           verbose, include_smooth, include_srcs)

def param_graph_individuals(param_type, path_list, condition_tuples, frac=0.5, 
                            verbose=False, include_smooth=True, include_srcs=False):
    
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        print(f"Category: {category}")
        for image in file_list:
            print(f"Image {image}:")
            single_param_graph(param_type, [image], str(category), frac, 
                                 verbose, include_smooth, include_srcs)
        print("------------------")


def single_param_graph(param_type, path_list, category_str="", frac=0.3,
                        verbose=False, include_smooth=True, include_srcs=False):
    
    images = unzip_directories(path_list, output_format='Fits_Simple')
    
    if param_type == 'fwhm':
        color_range = [1.5, 2.8]    # Optimized for Nickel 06-26-24 data
        title = "FWHM (arcsec)"
    elif param_type == 'fwhm residuals':
        color_range = [0.0, 1.2]    # Optimized for Nickel 06-26-24 data
        title = "FWHM Residuals (arcsec)"
    
    # Collect all coordinates and FWHMs in numpy arrays
    x_list, y_list, param_list = zip(*(calc_fwhm(image, mode=param_type, verbose=verbose) 
                                        for image in images))
    x_list = np.concatenate(x_list)
    y_list = np.concatenate(y_list)
    param_list = np.concatenate(param_list)
    param_list = param_list * plate_scale_approx
    
    # Create a figure for plotting
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    if include_smooth:
        if verbose: 
            print(x_list)
            print(f"Working on smoothed contour plot for category {category_str}")
        ax, cp = smooth_contour(x_list, y_list, param_list, color_range, 
                                ax, frac, title, category_str)
    if include_srcs:
        if verbose: 
            print(f"Working on sources plot for category {category_str}")
        ax, cmap = scatter_sources(x_list, y_list, param_list, 
                             color_range, ax, title, category_str)
    
    try:
        plt.colorbar(cp, ax=ax)
    except (UnboundLocalError, RuntimeError):
        plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.show()

