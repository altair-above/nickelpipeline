import numpy as np
from pathlib import Path
import warnings
from matplotlib import pyplot as plt
from collections import OrderedDict
from scipy import stats
from loess.loess_2d import loess_2d

from calc_fwhm import *

# from symlink_conv_funcs.dir_nav import categories_from_conditions, unzip_directories
# from symlink_astrometry.plate_scale import avg_plate_scale

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Allows for computing 
astrometry_dir = os.path.abspath(os.path.join(parent_dir, 'astrometry'))
sys.path.append(parent_dir)
sys.path.append(astrometry_dir)
from convenience_funcs.all_funcs import categories_from_conditions, unzip_directories, conditions_06_26, conditions_06_24, conditions
from astrometry.plate_scale import avg_plate_scale



def testing():
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_199_{filt}') for filt in ['B', 'V', 'R', 'I']]

    # rawdir = Path("")
    # rawdir = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw")  # path to directory with raw data

    # image_to_analyze = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw/d1079.fits")  # path to directory with raw data

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
    
    reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    directories = [dir for dir in reddir.iterdir() if ('NGC' not in str(dir) and 'Focus' not in str(dir))]
    fwhm_contour_by_category(directories, conditions_06_26)


default_fwhm_default=5.0
thresh_default=15
aper_size_default=8
local_bkg_range_default=(15,20)

plate_scale_approx = 0.37   # For the Nickel Telescope


def single_fwhm_contour(directories, files=None, title="", frac=0.3, verbose=False):
    
    images = unzip_directories(directories, files, output_format='Fits_Simple')
    
    # Collect all coordinates and FWHMs in numpy arrays
    all_x, all_y, all_residuals = zip(*(calc_fwhm(image, mode='fwhm residuals', verbose=verbose) 
                                        for image in images))
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_residuals = np.concatenate(all_residuals)
    # if verbose:
    # print("all_residuals:")
    # print(all_residuals)
    
    plate_scale = avg_plate_scale(directories, files=files, verbose=verbose, fast=True)
    all_residuals = all_residuals * plate_scale
    
    # Create grid for interpolation
    grid_x, grid_y = np.mgrid[0:1024:3, 0:1024:3]
    
    # Interpolate data using griddate
    # grid_z = griddata((clipped_x, clipped_y), clipped_fwhm, (grid_x, grid_y), method='linear')
    
    # Interpolate data using Loess smoothing - computationally difficult
    if verbose:
        print('loess_2d beginning')
    flat_z, wout = loess_2d(all_x, all_y, all_residuals, xnew=grid_x.flatten(),
                      ynew=grid_y.flatten(), frac=frac)
    if verbose:
        print('loess_2d done')
    grid_z = flat_z.reshape(grid_x.shape)
    
    # Define the colors & range of contour levels
    colors = ["#cc0018", "#cd0000", "#cb4000", "#c97f00", "#c7bc00", "#91c400", "#52c200", 
              "#00bc62", "#00ba9c", "#009cb8", "#0061b6", "#1100b1", "#4800af", "#7e00ad"]
    levels = np.linspace(0.0, 1.3, len(colors))
    # levels = np.linspace(4.5, 7.5, 10)
    # colors = ["#b70000", "#b64d00", "#b59a00", "#82b300", "#00b117", 
    #           "#00afab", "#0067ae", "#001dac", "#2c00ab", "#7400aa"]
    
    # Plot contour map
    plt.figure()
    cp = plt.contourf(grid_x, grid_y, grid_z, levels=levels, colors=colors)
    plt.colorbar(cp)
    plt.title(f'FWHM Residuals (arcsec) Contour Map - {title}')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.show()
    

def fwhm_contour_by_category(directories, condition_tuples, files=None, frac=0.3, verbose=False):
    
    images = unzip_directories(directories, files, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        single_fwhm_contour(None, files=file_list, title=category, frac=frac, verbose=verbose)

def fwhm_contour_individuals(directories, condition_tuples, files=None, frac=0.5, verbose=False):
    
    images = unzip_directories(directories, files, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    for category, file_list in categories.items():
        print(f"Category: {category}")
        for image in file_list:
            print(f"Image {image}:")
            single_fwhm_contour(None, files=[image], title=category,
                                     frac=frac, verbose=verbose)
        print("------------------")



    
warnings.filterwarnings('ignore')





