from pathlib import Path
import numpy as np
import re
from matplotlib import pyplot as plt

# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)
# # from convenience_funcs.fits_convenience_class import Fits_Simple
from nickelpipeline.astrometry.astrometry_api import run_astrometry
from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.dir_nav import unzip_directories, categories_from_conditions


def avg_plate_scale(path_list, verbose=True, fast=False):
    
    images = unzip_directories(path_list, output_format='Path')
    
    output_dir = str(images[0].parent.parent.parent / 'astrometric')
    
    astro_calib_images = run_astrometry(images, output_dir, fast=fast)
    if verbose:
        print(astro_calib_images)
    
    plate_scales = []
    for image in astro_calib_images:
        try:
            image = Fits_Simple(image)
        except (KeyError, OSError):
            continue
        for row in image.header['COMMENT']:
            if 'scale: ' in row:
                plate_scale = float(re.findall(r'\d+\.\d+|\d+', row)[0])
        if verbose:
            print(f"Scale for image {image.filename} = {plate_scale} arcsec/pix")
        plate_scales.append(plate_scale)
    
    avg = np.mean(plate_scales)
    std = np.std(plate_scales)
    if verbose:
        print(f'Mean scale for all images = {avg}')
        print(f'STD for all scale measurements = {std}')
    return avg

def graph_plate_scale_by_setting(path_list, condition_tuples, verbose=True, fast=False):
    
    images = unzip_directories(path_list, 
                               output_format='Fits_Simple', allow_exceptions=True)

    categories = categories_from_conditions(condition_tuples, images)

    data = []
    # Print out the categories and their corresponding file lists
    for category, file_list in categories.items():
        plate_scale = avg_plate_scale(file_list, fast=fast, verbose=verbose)
        data.append((category, plate_scale))
    
    data.sort()
    widths, plate_scales = zip(*data)
    
    # Plot the plate_scales relative to widths, with different colors for different objects
    plt.figure(figsize=(8, 5))
    plt.plot(widths, plate_scales, marker='o', linestyle='-', label='Plate Scale')
    
    # Calculate the line of best fit
    coefficients = np.polyfit(widths, plate_scales, 1)  # 1 for linear fit

    # Generate the line of best fit
    x_fit = np.linspace(min(widths), max(widths), 100)
    y_fit = coefficients[0] * x_fit + coefficients[1]

    # Plot the line of best fit
    plt.plot(x_fit, y_fit, color='red', label='Line of Best Fit')
    # Display the equation on the graph
    equation_text = f'Y = {coefficients[0]:.2e}X + {coefficients[1]:.4f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.xlabel('Spacer Width (in)')
    plt.ylabel('Plate Scale (arcsec/pixel)')
    plt.title(f'Plate Scale vs. Spacer Width')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return data
    
