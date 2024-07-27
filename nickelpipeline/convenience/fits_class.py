import numpy as np
import numpy.ma as ma
import re
import matplotlib.pyplot as plt
from matplotlib.path import Path as matPath
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from pathlib import Path
from typing import Union
import logging

from nickelpipeline.convenience.nickel_data import (ccd_shape, fov_shape, bad_columns, 
                                                    bad_triangles, bad_rectangles)

        
logger = logging.getLogger(__name__)

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
        
            # with fits.open(image_path) as hdul:
            #     self.header = hdul[0].header
            #     self.data = hdul[0].data
            #     try:
            #         self.mask = hdul['MASK'].data
            #     except KeyError:
            #         if all(self.data.shape == ccd_shape):
            #             self.mask = nickel_mask
            #         elif all(self.data.shape == fov_shape):
            #             self.mask = nickel_fov_mask
            # self.masked_array = ma.masked_array(self.data, self.mask)
            
            # try:
            #     self.image_num = extract_number(self.filename)
            #     self.object = self.header["OBJECT"]
            #     self.filtnam = self.header["FILTNAM"]
            #     self.exptime = self.header["EXPTIME"]
            # except:
            #     # For FITS images with limited header information
            #     self.image_num = None
            #     self.object = None
            #     self.filtnam = None
            #     self.exptime = None
    
    @property
    def header(self):
        with fits.open(self.path) as hdul:
            return hdul[0].header
    
    @property
    def data(self):
        with fits.open(self.path) as hdul:
            return hdul[0].data
        
    @property
    def mask(self):
        with fits.open(self.path) as hdul:
            try:
                return hdul['MASK'].data
            except KeyError:
                logger.debug('No mask in fits file; returning default mask')
                if all(self.data.shape == ccd_shape):
                    return nickel_mask
                elif all(self.data.shape == fov_shape):
                    return nickel_fov_mask
    
    @property
    def masked_array(self):
        return ma.masked_array(self.data, self.mask)

    @property
    def image_num(self):
        # Use regular expression to find all digits in the string
        numbers = re.findall(r'\d+', self.filename)
        return int(''.join(numbers))
    
    @property
    def object(self):
        return self.header["OBJECT"]
    @property
    def filtnam(self):
        return self.header["FILTNAM"]
    @property
    def exptime(self):
        return self.header["EXPTIME"]
    
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
        ndarray: The image with triangles masked.
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
        
        # Determine which points are inside the triangle
        y, x = np.mgrid[:rows, :cols]
        points = np.vstack((x.flatten(), y.flatten())).T
        mask = np.logical_or(mask, path.contains_points(points).reshape(rows, cols))

    # Mask the specified columns
    for col in cols_to_mask:
        mask[:, col] = True
    
    # Create the masked array
    masked_data = ma.masked_array(data, mask)
    return masked_data


# Mask for Nickel images (masking bad columns and blind corners)
nickel_mask_cols_only = add_mask(np.zeros(ccd_shape), bad_columns, [], []).mask
nickel_mask = add_mask(np.zeros(ccd_shape), bad_columns, bad_triangles,
                       bad_rectangles).mask

# Calculate the padding needed
pad_height = fov_shape[0] - ccd_shape[0]
pad_width = fov_shape[1] - ccd_shape[1]
# Apply padding
nickel_fov_mask_cols_only = np.pad(nickel_mask_cols_only, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
nickel_fov_mask = np.pad(nickel_mask, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

