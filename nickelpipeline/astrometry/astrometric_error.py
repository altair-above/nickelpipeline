import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import cm

from nickelpipeline.astrometry.plate_scale import avg_plate_scale
from nickelpipeline.astrometry.astrometry_api import run_astrometry, get_astrometric_solves
from nickelpipeline.convenience.nickel_data import plate_scale_approx
from nickelpipeline.convenience.dir_nav import unzip_directories, categories_from_conditions
from nickelpipeline.convenience.graphs import smooth_contour, scatter_sources


def graph_topographic(path_list, condition_tuples, error_type='error', 
                      frac=0.3, fast=False, verbose=False,
                      include_smooth=True, include_srcs=False):
    
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        single_graph_topographic(file_list, str(category), error_type, frac, 
                                 fast, verbose, include_smooth, include_srcs)


def graph_topographic_individuals(path_list, condition_tuples, error_type='error', 
                                  frac=0.3, fast=False, verbose=False,
                                  include_smooth=True, include_srcs=False):
    
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        print(f"Category: {category}")
        for image in file_list:
            print(f"Image {image.name}:")
            single_graph_topographic([image], str(category), error_type, frac, 
                                     fast, verbose, include_smooth, include_srcs)
        print("------------------")


def single_graph_topographic(path_list, category_str="", error_type='error', 
                             frac=0.3, fast=False, verbose=False, 
                             include_smooth=True, include_srcs=False):
    
    images = unzip_directories(path_list, output_format='Path')
    output_dir = str(images[0].parent.parent.parent / 'astrometric')
    if fast:
        astro_calib_images = get_astrometric_solves(images, output_dir, mode='corr')
    else:
        astro_calib_images = run_astrometry(images, output_dir, mode='corr')
    
    if len(astro_calib_images) == 0:
        print(f"Astrometry.net failed to calibrate image(s). Cannot graph.")
        return

    # Collect all coordinates and FWHMs in numpy arrays
    x_list, y_list, errors_list = zip(*(get_errors(image, error_type) 
                                        for image in astro_calib_images))
    x_list = np.concatenate(x_list)
    y_list = np.concatenate(y_list)
    errors_list = np.concatenate(errors_list)
    
    if fast:
        errors_list = errors_list * plate_scale_approx
    else:
        errors_list = errors_list * avg_plate_scale(path_list, verbose=False, fast=True)
    
    if error_type == 'error':
        color_range = [0.0, 0.3]
        title = "Astrometric Residuals (arcsec)"
    else:
        color_range = [-0.15, 0.15]
        title = f"Astrometric Residuals ({error_type} only) (arcsec)"
    
    # Create a figure for plotting
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    if include_smooth:
        if verbose: 
            print(f"Working on smoothed contour plot for category {category_str}")
        ax, cp = smooth_contour(x_list, y_list, errors_list, color_range, 
                                ax, frac, title, category_str)
    if include_srcs:
        if verbose: 
            print(f"Working on sources plot for category {category_str}")
        ax, cmap = scatter_sources(x_list, y_list, errors_list, 
                             color_range, ax, title, category_str)
    
    try:
        plt.colorbar(cp, ax=ax)
    except (UnboundLocalError, RuntimeError):
        plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.show()
    
    

def get_errors(image, error_type='error'):
    try:
        with fits.open(image) as hdul:
            # Assuming the table is in the first extension
            data = hdul[1].data
            data = data[data['match_weight'] > 0.99]
            
            x_error = data['field_x'] - data['index_x']
            y_error = data['field_y'] - data['index_y']
            error = np.sqrt(x_error**2 + y_error**2)
        
        if error_type == 'error':
            return np.array(data['field_x']), np.array(data['field_y']), np.array(error)
        elif error_type == 'y':
            return np.array(data['field_x']), np.array(data['field_y']), np.array(y_error)
        elif error_type == 'x':
            return np.array(data['field_x']), np.array(data['field_y']), np.array(x_error)
    except OSError:
        print("Astrometry.net could not solve this image")
        return np.array([]), np.array([]), np.array([])
    


