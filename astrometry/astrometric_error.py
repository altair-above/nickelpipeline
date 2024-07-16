import numpy as np
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from astrometry_api import run_astrometry
from loess.loess_2d import loess_2d

from plate_scale import avg_plate_scale

# from ..convenience_funcs.fits_convenience_class import Fits_Simple

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# from convenience_funcs.fits_convenience_class import Fits_Simple
from convenience_funcs.all_funcs import (unzip_directories, categories_from_conditions,
                                         conditions_06_24, conditions_06_26, conditions)


def testing():
    reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]
    graph_topographic(directories, condition_tuples=conditions_06_26)

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 
    #                                                 'NGC' not in str(dir))]
    # graph_topographic_individuals(directories, condition_tuples=conditions_06_26, 
    #                 frac=0.3, verbose=False)


def graph_topographic(directories, condition_tuples, files=None, error_type='error',
                      fit_type='loess', frac=0.3, fast=False):
    
    images = unzip_directories(directories, files, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        single_graph_topographic(None, files=file_list, title=category, 
                                 error_type=error_type, fit_type=fit_type,
                                 frac=frac, fast=fast)


def graph_topographic_individuals(directories, condition_tuples, files=None, error_type='error',
                      fit_type='loess', frac=0.3, fast=False):
    
    images = unzip_directories(directories, files, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        print(f"Category: {category}")
        for image in file_list:
            print(f"Image {image}:")
            single_graph_topographic(None, files=[image], title=category, 
                                    error_type=error_type, fit_type=fit_type,
                                    frac=frac, fast=fast)
        print("------------------")


def single_graph_topographic(directories, files=None, title="", error_type='error',
                      fit_type='loess', frac=0.3, fast=False):
    
    images = unzip_directories(directories, files, output_format='Path')
    
    if files is not None:
        output_dir = str(Path(files[0]).parent.parent.parent / 'astrometric')
    else:
        output_dir = str(Path(directories[0]).parent.parent / 'astrometric')
    
    astro_calib_images = run_astrometry(images, output_dir, mode='corr', fast=fast)
    print(astro_calib_images)
    
    # Collect all coordinates and FWHMs in numpy arrays
    all_x, all_y, all_errors = zip(*(get_errors(image, error_type) 
                                        for image in astro_calib_images))
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_errors = np.concatenate(all_errors)
    
    plate_scale = avg_plate_scale(directories, files=files, verbose=False, fast=fast)
    all_errors = all_errors * plate_scale
    
    # Create grid for interpolation
    grid_x, grid_y = np.mgrid[0:1024:3, 0:1024:3]
    
    if fit_type == 'loess':
        # Interpolate data using Loess smoothing - computationally difficult
        print('loess_2d beginning')
        flat_z, wout = loess_2d(all_x, all_y, all_errors, xnew=grid_x.flatten(),
                        ynew=grid_y.flatten(), frac=frac)
        print('loess_2d done')
        grid_z = flat_z.reshape(grid_x.shape)
    elif fit_type == 'griddata':
        # Interpolate data - griddata
        grid_z = griddata((all_x, all_y), all_errors, (grid_x, grid_y), method=fit_type)
    else:
        print("fit_type must be 'loess' or 'griddata'")
        return "fit_type must be 'loess' or 'griddata"
    
    # cmap = plt.get_cmap('terrain')
    # Define the color & range of contour levels
    colors = ["#cc0018", "#cd0000", "#cb4000", "#c97f00", "#c7bc00", "#91c400", "#52c200", 
              "#00bc62", "#00ba9c", "#009cb8", "#0061b6", "#1100b1", "#4800af", "#7e00ad"]
    if fit_type == 'loess':
        if error_type == 'error':
            levels = np.linspace(0, 0.3, len(colors))
        else:
            levels = np.linspace(-0.15, 0.15, len(colors))
    elif fit_type == 'griddata':
        if error_type == 'error':
            levels = np.linspace(0, 4.0, len(colors))
        else:
            levels = np.linspace(-2.0, 2.0, len(colors))
    else:
        print("fit_type must be 'loess' or 'griddata', and error_type must be 'error', 'x', or 'y'")
        return "fit_type must be 'loess' or 'griddata', and error_type must be 'error', 'x', or 'y'"


    # colors = ["#b70000", "#b64d00", "#b59a00", "#82b300", "#00b117", 
    #           "#00afab", "#0067ae", "#001dac", "#2c00ab", "#7400aa"]

    # Plot contour map
    plt.figure()
    cp = plt.contourf(grid_x, grid_y, grid_z, levels=levels, colors=colors,)
    plt.colorbar(cp)
    plt.title(f'Astrometric Residuals (arcsec) Contour Map ({error_type}) - {title}')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
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



# conditions_06_26 = [(1.375, (65, 74)),
#                     (1.625, (22, 31)),
#                     (1.625, (88, 105)),
#                     (1.875, (33, 42)),
#                     (2.625, (43, 53)),
#                     (3.375, (54, 64)),
#                     ]

# conditions_06_24 = [(1.375, (53, 60)),
#                     (1.625, (1, 52)),
#                     (1.625, (88, 105))
#                     ]

# conditions = {'06-26': conditions_06_26, '06-24': conditions_06_24}    




