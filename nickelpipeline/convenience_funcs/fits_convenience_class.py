from astropy.io import fits
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


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
    
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], cls):
            # If the first argument is an instance of Fits_Simple, return it directly
            return args[0]
        # Otherwise, create a new instance
        return super(Fits_Simple, cls).__new__(cls)
    
    
    def __init__(self, image_path: Union[str, Path]):
        """
        Initializes the Fits_Simple object with the given image path.

        Args:
            image_path (Union[str, Path]): The file path to the FITS image. Can be a string or a Path object.
        """
        if isinstance(image_path, Fits_Simple):
            return
            # self = Fits_Simple(image_path.path)
        else:
            image_path = Path(image_path)
            self.path = image_path
            self.filename: Path = image_path.stem
        
            with fits.open(image_path) as hdu:
                self.header = hdu[0].header
                self.data = hdu[0].data
            
            self.image_num = int(self.filename[2:5])
            self.object: str = self.header["OBJECT"]
            self.filtnam: str = self.header["FILTNAM"]
            self.exptime: float = self.header["EXPTIME"]
            
            # Mask for Nickel images (masking bad columns and blind corners)
            columns_to_mask = [255, 256, 783, 784, 1002]
            triangles_to_mask = []#[((0, 960), (64, 1024), (0, 1024)), ((0, 33), (34, 0), (0,0))]

            self.masked_array = add_mask(self.data, columns_to_mask, triangles_to_mask)
            self.mask = self.masked_array.mask
            
    
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



def add_mask(data, columns_to_mask, triangles_to_mask):
    """
    Create a masked array with specified columns and triangular regions masked.
    
    Parameters:
    - shape: Tuple representing the shape of the array (rows, columns).
    - columns_to_mask: List of column indices to mask.
    - triangles_to_mask: List of tuples representing triangles to mask.
      Each triangle is defined by a tuple of two points (each point is a tuple of (row, column)).
      
    Returns:
    - masked_array: A numpy masked array with specified regions masked.
    """
    rows, cols = data.shape
    mask = np.zeros((rows, cols), dtype=bool)

    # Mask the specified columns
    for col in columns_to_mask:
        mask[:, col] = True
    
    # Mask the specified triangular regions
    for triangle in triangles_to_mask:
        (r0, c0), (r1, c1), (r2, c2) = triangle
        
        for r in range(rows):
            for c in range(cols):
                # Use line equation to determine if point (r, c) is inside the triangle
                # Equation of a line in parametric form: (x - x0) / (x1 - x0) = (y - y0) / (y1 - y0)
                if (r0 <= r <= r1 and c0 <= c <= c1) or (r1 <= r <= r0 and c1 <= c <= c0):
                    mask[r, c] = True
    
    # Create the masked array
    masked_array = ma.masked_array(data, mask)
    
    return masked_array


