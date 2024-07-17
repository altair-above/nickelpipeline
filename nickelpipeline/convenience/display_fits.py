import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
from astropy.io import fits
from nickelpipeline.convenience.fits_class import Fits_Simple
from astropy.visualization import ZScaleInterval


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
        plt.imshow(hdul[0].data, origin='lower')
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
    data = np.delete(data, [255, 256, 783, 784, 1002], axis=1)
    plt.figure(figsize=(8, 6))

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    plt.gcf().set_dpi(300)
    plt.colorbar()
   
def display_many_nickel(directories, files=None):
    """
    Displays the data of all images in a list of directories or files.

    Args:
        image (Union[str, Fits_Simple]): The Fits_Simple object or path to the FITS image.
    """
    if files is not None:
        images = [Fits_Simple(file) for file in files]
    else:
        if not isinstance(directories, list):
            directories = [directories,]
        images = []
        for dir in directories:
            dir = Path(dir)
            images += [Fits_Simple(file) for file in dir.iterdir()]
    for image in images:
        display_nickel(image)