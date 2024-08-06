import numpy.ma as ma
import re
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from pathlib import Path
from typing import Union
import logging

from nickelpipeline.convenience.nickel_data import (ccd_shape, fov_shape)
from nickelpipeline.convenience.nickel_masks import get_masks_from_file
        
logger = logging.getLogger(__name__)

class Fits_Simple:
    """
    A simple class to handle FITS files.

    Attributes:
        path (Path): The file path to the FITS image.
        filename (str): The name of the FITS file.
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
            self.filename = image_path.name
            self.filenamebase = image_path.name.split('_')[0]
            self._data = None
            self._mask = None
    
    @property
    def header(self):
        """The header information of the FITS file."""
        with fits.open(self.path) as hdul:
            return hdul[0].header
    
    @property
    def data(self):
        """The data contained in the FITS file."""
        if self._data is None:
            with fits.open(self.path) as hdul:
                self._data = hdul[0].data
        return self._data
    @data.setter
    def data(self, new_data):
        """The data contained in the FITS file."""
        self._data = new_data
        
    @property
    def mask(self):
        """Mask for the fits data."""
        if self._mask is None:
            with fits.open(self.path) as hdul:
                try:
                    return hdul['MASK'].data
                except KeyError:
                    logger.debug(f'No mask in FITS file {self.path.name}; returning default mask')
                    if all(self.shape == ccd_shape):
                        return get_masks_from_file('mask')
                    elif all(self.shape == fov_shape):
                        return get_masks_from_file('fov_mask')
        else:
            return self._mask
    @mask.setter
    def mask(self, new_mask):
        if new_mask.shape != self.shape:
            raise ValueError(f"new_mask must have same shape as data {self.data.shape}. new_mask.shape = {new_mask.shape}")
        self._mask = new_mask
    
    @property
    def masked_array(self):
        return ma.masked_array(self.data, self.mask)

    @property
    def shape(self):
        header = self.header
        shape = (int(header['NAXIS1']), int(header['NAXIS2']))
        return shape

    @property
    def image_num(self):
        # Use regular expression to find all digits in the string
        numbers = re.findall(r'\d+', self.filename)
        return int(''.join(numbers))
    
    @property
    def object(self):
        """The object name from the FITS header."""
        try:
            return self.header["OBJECT"]
        except KeyError:
            return None
    @property
    def filtnam(self):
        """The filter name from the FITS header."""
        try:
            return self.header["FILTNAM"]
        except KeyError:
            return None
    @property
    def exptime(self):
        """The exposure time from the FITS header."""
        try:
            return self.header["EXPTIME"]
        except KeyError:
            return None
    @property
    def airmass(self):
        try:
            return self.header["AIRMASS"]
        except KeyError:
            return None
    
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




