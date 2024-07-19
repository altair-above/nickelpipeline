import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.path import Path as matPath
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from pathlib import Path
from typing import Union

from nickelpipeline.convenience.nickel_data import bad_columns, bad_triangles, bad_rectangles

        
class Fits_Simple:
    """
    A simple class to handle FITS files.

    Attributes:
        path (Path): The file path to the FITS image.
        filename (str): The name of the FITS file.
        header (Header): The header information of the FITS file.
        data (ndarray): The data contained in the FITS file.
        object (str): The object name from the FITS header.
        filtnam (str): The filter name from the FITS header.
        exptime (float): The exposure time from the FITS header.
    """
    
    def __new__(cls, *args, **kwargs):
        """
        Returns an existing instance if the first argument is a Fits_Simple instance,
        otherwise creates a new instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Fits_Simple: An instance of the Fits_Simple class.
        """
        if args and isinstance(args[0], cls):
            return args[0]
        return super(Fits_Simple, cls).__new__(cls)
    
    def __init__(self, image_path: Union[str, Path]):
        """
        Initializes the Fits_Simple object with the given image path.

        Args:
            image_path (Union[str, Path]): The file path to the FITS image. Can be a string or a Path object.
        """
        if isinstance(image_path, Fits_Simple):
            return
        else:
            image_path = Path(image_path)
            self.path = image_path
            self.filename = image_path.stem
        
            with fits.open(image_path) as hdu:
                self.header = hdu[0].header
                self.data = hdu[0].data
            
            self.image_num = int(self.filename[2:5])
            self.object = self.header["OBJECT"]
            self.filtnam = self.header["FILTNAM"]
            self.exptime = self.header["EXPTIME"]
            
            self.mask = mask
            self.masked_array = ma.masked_array(self.data, mask)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Fits_Simple object.

        Returns:
            str: A string containing the filename and object name.
        """
        return f"{self.filename} ({self.object} - {self.filtnam})"

    def display(self, header: bool = False, data: bool = False):
        """
        Displays the FITS image and optionally prints the header and data.

        Args:
            header (bool): Whether to print the header information. Defaults to False.
            data (bool): Whether to print the data. Defaults to False.
        """
        with fits.open(self.path) as hdul:
            if header:
                print("HDU List Info:")
                print(hdul.info())
                print("\nHDU Header")
                print(repr(hdul[0].header))
            if data:
                print(hdul[0].data)
            
            plt.figure(figsize=(8, 6))
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(hdul[0].data)
            plt.imshow(hdul[0].data, origin='lower', vmin=vmin, vmax=vmax)
            plt.gcf().set_dpi(300)
            plt.colorbar()
            plt.show()


def add_mask(data: np.ndarray, cols_to_mask: list, tris_to_mask: list, rects_to_mask: list) -> ma.MaskedArray:
    """
    Masks the triangles from the image by setting the pixel values within those triangles to zero.

    Args:
        data (ndarray): 2D numpy array representing the image to mask.
        cols_to_mask (list): List of column indices to mask.
        tris_to_mask (list): List of tuples of 3 coordinates representing triangles to mask.
        rects_to_mask (list): List of tuples of 4 coordinates representing rectangles to mask.

    
    Returns:
    numpy.ndarray: The image with triangles masked.
    """
    rows, cols = data.shape
    mask = np.zeros((rows, cols), dtype=bool)
    
    # Mask the rectangles
    for rectangle in rects_to_mask:
        mask[rectangle[0][0]:rectangle[1][0],
             rectangle[0][1]:rectangle[1][1]] = True
    # Transpose mask so that correct areas are masked (FITS indexing is odd)
    mask = mask.T

    for triangle in tris_to_mask:
        # Create a path object for the triangle
        path = matPath(triangle)
        
        # Create a grid of points
        y, x = np.mgrid[:rows, :cols]
        points = np.vstack((x.flatten(), y.flatten())).T
        
        # Determine which points are inside the triangle
        mask = np.logical_or(mask, path.contains_points(points).reshape(rows, cols))

    # Mask the specified columns
    for col in cols_to_mask:
        mask[:, col] = True
    
    # Create the masked array
    masked_data = ma.masked_array(data, mask)
    return masked_data


# Mask for Nickel images (masking bad columns and blind corners)
img_shape = (1024, 1024)
columns_to_mask = bad_columns
triangles_to_mask = bad_triangles
rectangles_to_mask = bad_rectangles
mask_cols_only = add_mask(np.zeros(img_shape), columns_to_mask, [], []).mask
mask = add_mask(np.zeros(img_shape), columns_to_mask, triangles_to_mask,
                rectangles_to_mask).mask
