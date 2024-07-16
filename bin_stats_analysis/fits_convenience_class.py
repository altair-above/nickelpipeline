from astropy.io import fits
from pathlib import Path

# class Fits_Simple:
#     """Class to simplify the handling of FITS files

#     Attributes:
#         path (str): The file path to the FITS image.
#         filename (str): The name of the FITS file.
#         header: The header information of the FITS file.
#         data: The data contained in the FITS file.
#         object (str): The object name from the FITS header.
#         filtnam (str): The filter name from the FITS header.
#         exptime (float): The exposure time from the FITS header.
#     """

#     def __init__(self, image_path):
#         self.path = image_path
#         try:
#             self.filename = image_path.name
#         except:
#             print("image_path is not a Path instance: filename may be inaccurate")
#             self.filename = image_path.split('/')[-1]
#         with fits.open(image_path) as hdu:
#             self.header = hdu[0].header
#             self.data = hdu[0].data
#         self.object = self.header["OBJECT"]
#         self.filtnam = self.header["FILTNAM"]
#         self.exptime = self.header["EXPTIME"]
        
#     def __str__(self):
#         return f"{self.filename} ({self.object})"
    

from astropy.io import fits
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

class Fits_Simple:
    """
    A simple class to handle FITS files.

    Attributes:
        path (str): The file path to the FITS image.
        filename (str): The name of the FITS file.
        header: The header information of the FITS file.
        data: The data contained in the FITS file.
        object (str): The object name from the FITS header.
        filtnam (str): The filter name from the FITS header.
        exptime (float): The exposure time from the FITS header.
    """
    
    def __init__(self, image_path: Union[str, Path]):
        """
        Initializes the Fits_Simple object with the given image path.

        Args:
            image_path (Union[str, Path]): The file path to the FITS image. Can be a string or a Path object.
        """
        if isinstance(image_path, Fits_Simple):
            self.filename = self.filename
            self.path = self.path
        else:
            self.path: Union[str, Path] = image_path
            try:
                self.filename: Path = image_path.stem
            except AttributeError:
                print("image_path is not a Path instance: filename may be inaccurate")
                self.filename = image_path.split('/')[-1]
        
        with fits.open(image_path) as hdu:
            self.header = hdu[0].header
            self.data = hdu[0].data
        
        self.object: str = self.header["OBJECT"]
        self.filtnam: str = self.header["FILTNAM"]
        self.exptime: float = self.header["EXPTIME"]
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Fits_Simple object.

        Returns:
            str: A string containing the filename and object name.
        """
        return f"{self.filename} ({self.object} - {self.filtnam})"
    
    
    def display(self, header=False, data=False):
        """Prints HDU List info, HDU header, and data

        Args:
            image_path (str): path to fits image (greyscale only)
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
            plt.imshow(hdul[0].data, origin='lower')
            # Set the DPI (dots per inch) for the figure
            plt.gcf().set_dpi(300)

            # Display the plot
            plt.colorbar()
            plt.show()