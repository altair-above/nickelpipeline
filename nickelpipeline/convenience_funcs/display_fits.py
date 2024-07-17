from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from fits_convenience_class import Fits_Simple
from astropy.visualization import ZScaleInterval


def print_fits_info(image_path):
    """Prints HDU List info, HDU header, and data

    Args:
        image_path (str): path to fits image (greyscale only)
    """

    with fits.open(image_path) as hdul:
        # print("HDU List Info:")
        # print(hdul.info())
        print("\nHDU Header")
        print(repr(hdul[0].header))
        
        # print(hdul[0].data)
        
        plt.figure(figsize=(8, 6))  # Set the figure size if needed
        plt.imshow(hdul[0].data, origin='lower')
        # Set the DPI (dots per inch) for the figure
        plt.gcf().set_dpi(300)  # Adjust the DPI value as needed

        # Display the plot
        plt.colorbar()
        plt.show()
        
def display_nickel(image):
    """Displays data

    Args:
        image (Fits_Simple):
    """
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    print(image)
    print(f'filt = {image.filtnam}')

    data = image.data
    data = np.delete(data, [255, 256, 783, 784, 1002], axis=1)
    plt.figure(figsize=(8, 6))  # Set the figure size if needed

    # Apply ZScaleInterval
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    # Set the DPI (dots per inch) for the figure
    plt.gcf().set_dpi(300)  # Adjust the DPI value as needed

    # Display the plot
    plt.colorbar()
    plt.show()

def display_many_nickel(directories, files=None):
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