from pathlib import Path
import numpy as np
from astrometry_api import run_astrometry
import re
from matplotlib import pyplot as plt

# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)
# # from convenience_funcs.fits_convenience_class import Fits_Simple
from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.dir_nav import unzip_directories, categories_from_conditions


def testing():
    #
    # calib_images = [f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_199_{filt}' for filt in ['B', 'V', 'R', 'I']]

    # directories = []
    # output_dir = 'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-/astrometric'

    # raw_red = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced')
    # dirs = [dir for dir in raw_red.iterdir()]
    # display_many_nickel(dirs)

    # # Initial configuration
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/PG1530+057_{filt}') for filt in ['B', 'V', 'R', 'I']]

    # # Night 1 mod
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_231_{filt}') for filt in ['R']]

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]
    # graph_plate_scale_by_setting(directories, condition_tuples=conditions_06_26, fast=True)
    return


def avg_plate_scale(directories, files=None, verbose=True, fast=False):
    
    images = unzip_directories(directories, files, output_format='Path')
    
    if files is not None:
        output_dir = str(Path(files[0]).parent.parent.parent / 'astrometric')
    else:
        output_dir = str(Path(directories[0]).parent.parent / 'astrometric')
    
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

def graph_plate_scale_by_setting(directories, condition_tuples, files=None, fast=False):
    
    # if fast:
    #     directories = [Path(directories[0]).parent.parent / 'astrometric']
    
    if files is not None:
        images = [Fits_Simple(file) for file in files]
    else:
        if not isinstance(directories, list):
            directories = [directories,]
        images = []
        for dir in directories:
            for file in Path(dir).iterdir():
                try:
                    images.append(Fits_Simple(file))
                except (KeyError, OSError):
                    continue
    
    # # Initialize an empty dictionary
    # conditions = {}
    # # Process the input list
    # for value, (start, end) in condition_tuples:
    #     if value in conditions:
    #         conditions[value].append((start, end))
    #     else:
    #         conditions[value] = [(start, end)]

    # # Convert ranges to lambda functions
    # conditions = {width: (lambda ranges: lambda img_num: any(start <= img_num <= end for start, end in ranges))(ranges) for width, ranges in conditions.items()}
    # # Use dictionary comprehension to create the categories
    # categories = {width: [file.path for file in images if condition(file.image_num)] for width, condition in conditions.items()}
    
    categories = categories_from_conditions(condition_tuples, images)

    data = []
    # Print out the categories and their corresponding file lists
    for category, file_list in categories.items():
        plate_scale = avg_plate_scale(None, files=file_list, fast=fast)
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
    
