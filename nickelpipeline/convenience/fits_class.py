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

    Attributes
    ----------
    path : Path
        The file path to the FITS image.
    filename : str
        The name of the FITS file.
    filenamebase : str
        The base name of the FITS file (before any underscores).
    _data : ndarray
        Cached image data from the FITS file.
    _mask : ndarray
        Cached mask data from the FITS file.
    """

    def __new__(cls, *args, **kwargs):
        """
        Returns an existing instance if the first argument is a Fits_Simple instance,
        otherwise creates a new instance.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        Fits_Simple
            An instance of the Fits_Simple class.
        """
        if args and isinstance(args[0], cls):
            return args[0]
        return super(Fits_Simple, cls).__new__(cls)

    def __init__(self, image_path: Union[str, Path]):
        """
        Initializes the Fits_Simple object with the given image path.

        Parameters
        ----------
        image_path : Union[str, Path]
            The file path to the FITS image. Can be a string or a Path object.
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
        """
        The header information of the FITS file.

        Returns
        -------
        header : Header
            The header information from the FITS file.
        """
        with fits.open(self.path) as hdul:
            return hdul[0].header

    @property
    def data(self):
        """
        The data contained in the FITS file.

        Returns
        -------
        data : ndarray
            The image data from the FITS file.
        """
        if self._data is None:
            with fits.open(self.path) as hdul:
                self._data = hdul[0].data
        return self._data

    @data.setter
    def data(self, new_data):
        """
        Setter for the data attribute.

        Parameters
        ----------
        new_data : ndarray
            The new data to replace the existing data in the FITS file.
        """
        self._data = new_data

    @property
    def mask(self):
        """
        Mask for the FITS data. If no mask is present in the FITS file,
        returns a default mask based on the image shape.

        Returns
        -------
        mask : ndarray
            The mask data from the FITS file or a default mask.
        """
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
        """
        Setter for the mask attribute.

        Parameters
        ----------
        new_mask : ndarray
            The new mask to replace the existing mask in the FITS file.

        Raises
        ------
        ValueError
            If the shape of the new mask does not match the shape of the data.
        """
        if new_mask.shape != self.shape:
            raise ValueError(f"new_mask must have same shape as data {self.data.shape}. new_mask.shape = {new_mask.shape}")
        self._mask = new_mask

    @property
    def masked_array(self):
        """
        Returns a masked array combining data and mask.

        Returns
        -------
        masked_array : MaskedArray
            The masked array combining the FITS data and mask.
        """
        return ma.masked_array(self.data, self.mask)

    @property
    def shape(self):
        """
        The shape (dimensions) of the FITS image.

        Returns
        -------
        shape : tuple
            The shape of the FITS image as a tuple (NAXIS1, NAXIS2).
        """
        header = self.header
        shape = (int(header['NAXIS1']), int(header['NAXIS2']))
        return shape

    @property
    def image_num(self):
        """
        Extracts and returns the image number from the filename.

        Returns
        -------
        image_num : int
            The image number parsed from the filename.
        """
        numbers = re.findall(r'\d+', self.filename)
        return int(''.join(numbers))

    @property
    def object(self):
        """
        The object name from the FITS header.

        Returns
        -------
        object : str or None
            The object name from the FITS header, or None if not found.
        """
        try:
            return self.header["OBJECT"]
        except KeyError:
            return None

    @property
    def filtnam(self):
        """
        The filter name from the FITS header.

        Returns
        -------
        filtnam : str or None
            The filter name from the FITS header, or None if not found.
        """
        try:
            return self.header["FILTNAM"]
        except KeyError:
            return None

    @property
    def exptime(self):
        """
        The exposure time from the FITS header.

        Returns
        -------
        exptime : float or None
            The exposure time from the FITS header, or None if not found.
        """
        try:
            return self.header["EXPTIME"]
        except KeyError:
            return None

    @property
    def airmass(self):
        """
        The airmass value from the FITS header.

        Returns
        -------
        airmass : float or None
            The airmass value from the FITS header, or None if not found.
        """
        try:
            return self.header["AIRMASS"]
        except KeyError:
            return None

    def __str__(self) -> str:
        """
        Returns a string representation of the Fits_Simple object.

        Returns
        -------
        str
            A string containing the filename, object name, and filter.
        """
        return f"{self.filename} ({self.object} - {self.filtnam})"

    def display(self, header: bool = False, data: bool = False):
        """
        Displays the FITS image and optionally prints the header and data.

        Parameters
        ----------
        header : bool, optional
            Whether to print the header information (default is False).
        data : bool, optional
            Whether to print the data (default is False).
        """
        with fits.open(self.path) as hdul:
            if header:
                print("HDU List Info:")
                print(hdul.info())
                print("\nHDU Header")
                print(repr(hdul[0].header))
            if data:
                print(hdul[0].data)

            # Display the image with ZScale interval normalization
            plt.figure(figsize=(8, 6))
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(hdul[0].data)
            plt.imshow(hdul[0].data, origin='lower', vmin=vmin, vmax=vmax)
            plt.gcf().set_dpi(300)
            plt.colorbar()
            plt.show()
