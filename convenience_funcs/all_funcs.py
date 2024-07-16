
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

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
            
            # Mask for Nickel images (masking bad columns and blind corners)
            columns_to_mask = [255, 256, 783, 784, 1002]
            triangles_to_mask = [] #[((0, 960), (64, 1024)), ((0, 33), (34, 0))]
            rectangles_to_mask = [((0,960), (64, 1024)), ((0,0), (34, 33))]

            self.masked_array_cols_only = add_mask(self.data, columns_to_mask, 
                                                   triangles_to_mask, [])
            self.masked_array = add_mask(self.data, columns_to_mask, triangles_to_mask,
                                         rectangles_to_mask)
            self.mask = self.masked_array.mask
    
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
            plt.imshow(hdul[0].data, origin='lower')
            plt.gcf().set_dpi(300)
            plt.colorbar()
            plt.show()


def add_mask(data: np.ndarray, columns_to_mask: list, triangles_to_mask: list, rectangles_to_mask: list) -> ma.MaskedArray:
    """
    Create a masked array with specified columns and regions masked.

    Args:
        data (np.ndarray): The data to be masked.
        columns_to_mask (list): List of column indices to mask.
        triangles_to_mask (list): List of tuples representing triangles to mask.
        rectangles_to_mask (list): List of tuples representing rectangles to mask.

    Returns:
        ma.MaskedArray: A masked array with specified regions masked.
    """
    rows, cols = data.shape
    mask = np.zeros((rows, cols), dtype=bool)
    # mask = mask.T
    
    for rectangle in rectangles_to_mask:
        mask[rectangle[0][0]:rectangle[1][0],
             rectangle[0][1]:rectangle[1][1]] = True
    mask = mask.T

    # Mask the specified columns
    for col in columns_to_mask:
        mask[:, col] = True
        
    # Mask the specified triangular regions
    for triangle in triangles_to_mask:
        (r0, c0), (r1, c1) = triangle
        for r in range(rows):
            for c in range(cols):
                # Use line equation to determine if point (r, c) is inside the triangle
                # Equation of a line in parametric form: (x - x0) / (x1 - x0) = (y - y0) / (y1 - y0)
                if (r0 <= r <= r1 and c0 <= c <= c1) or (r1 <= r <= r0 and c1 <= c <= c0):
                    mask[r, c] = True
    
    # Create the masked array
    masked_array = ma.masked_array(data, mask)
    return masked_array


def categories_from_conditions(condition_tuples: list, images: list) -> dict:
    """
    Categorizes images based on conditions.

    Args:
        condition_tuples (list): List of tuples where each tuple contains a value and a range (start, end).
        images (list): List of Fits_Simple objects.

    Returns:
        dict: Dictionary categorizing images based on conditions {condition: [image1, image2,...]}
    """
    conditions = {}
    for value, (start, end) in condition_tuples:
        if value in conditions:
            conditions[value].append((start, end))
        else:
            conditions[value] = [(start, end)]

    conditions = {width: (lambda ranges: lambda img_num: any(start <= img_num <= end for start, end in ranges))(ranges) for width, ranges in conditions.items()}
    categories = {width: [file.path for file in images if condition(file.image_num)] for width, condition in conditions.items()}
    return categories


def unzip_directories(directories: Union[str, list], files: list = None, output_format: str = 'Fits_Simple') -> list:
    """
    Unzips directories and returns a list of image objects.

    Args:
        directories (Union[str, list]): Directories to unzip.
        files (list, optional): List of files to unzip. Defaults to None.
        output_format (str, optional): The format of the output objects. Defaults to 'Fits_Simple'.

    Returns:
        list: List of image objects.
    """
    if output_format == 'Path':
        output = Path
    elif output_format == 'Fits_Simple':
        output = Fits_Simple
    
    if files is not None:
        images = [output(file) for file in files]
    else:
        if not isinstance(directories, list):
            directories = [directories]
        images = []
        for dir in directories:
            images += [output(file) for file in Path(dir).iterdir()]
    return images


def unzip_directories(directories: list, files: list = None, output_format: str = 'Fits_Simple') -> list:
    """
    Unzips directories and returns a list of image objects.

    Args:
        directories (Union[str, list]): Directories to unzip.
        files (list, optional): List of files to unzip. Defaults to None.
        output_format (str, optional): The format of the output objects. Defaults to 'Fits_Simple'.

    Returns:
        list: List of image objects.
    """
    if output_format == 'Path':
        output = Path
    elif output_format == 'Fits_Simple':
        output = Fits_Simple
    
    if files is not None:
        images = [output(file) for file in files]
        # print("'files' parameter is not ideal. 'directories' handles lists that contain both directories & files")
    else:
        images = []
        for elem_path in directories:
            elem_path = Path(elem_path)
            if elem_path.is_dir():
                images += [output(file) for file in elem_path.iterdir()]
            else:
                images.append(output(elem_path))
    return images



conditions_06_26 = [
    (1.375, (65, 74)),
    (1.624, (22, 31)),
    (1.625, (88, 105)),
    (1.875, (33, 42)),
    (2.625, (43, 53)),
    (3.375, (54, 64)),
]

conditions_06_24 = [
    (1.375, (53, 60)),
    (1.625, (1, 52)),
    (1.625, (88, 105))
]

conditions = {'06-26': conditions_06_26, '06-24': conditions_06_24}


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