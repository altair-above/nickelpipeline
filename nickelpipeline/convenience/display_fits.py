import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.dir_nav import unzip_directories
from nickelpipeline.convenience.nickel_data import bad_columns


def print_fits_info(image_path: str):
    """
    Prints HDU List info, HDU header, and displays the data.

    Args:
        image_path (str): Path to the FITS image (greyscale only).
    """
    with fits.open(image_path) as hdul:
        print("\nHDU Header")
        print(repr(hdul[0].header))
        
        plt.figure(figsize=(8, 6))
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(hdul[0].data)
        plt.imshow(hdul[0].data, origin='lower', vmin=vmin, vmax=vmax)
        plt.gcf().set_dpi(300)
        plt.colorbar()
        plt.show()


def display_nickel(image: Union[str, Fits_Simple]):
    """
    Displays the data of a fits image (in path or Fits_Simple format) after
    removing columns corresponding to the old Nickel science camera's bad columns.

    Args:
        image (Union[str, Fits_Simple]): The Fits_Simple object or path to the FITS image.
    """
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    print(image)
    print(f'Filter = {image.filtnam}')

    data = image.data
    data = np.delete(data, bad_columns, axis=1)
    plt.figure(figsize=(8, 6))

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    plt.gcf().set_dpi(300)
    plt.colorbar()
    plt.show()
   
def display_many_nickel(path_list):
    """
    Displays the data of all images in a list of directories or files.

    Args:
        image (Union[str, Fits_Simple]): The Fits_Simple object or path to the FITS image.
    """
    images = unzip_directories(path_list, output_format='Fits_Simple')
    for image in images:
        display_nickel(image)