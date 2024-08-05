import numpy as np
import numpy.ma as ma
from pathlib import Path
from matplotlib.path import Path as matPath
import logging

from nickelpipeline.convenience.nickel_data import * 
        
logger = logging.getLogger(__name__)

mask_file = 'nickel_masks.npz'
parent_dir = Path(__file__).parent.resolve()
mask_file = parent_dir / mask_file


def get_masks_from_file(mode):
    """
    Load and retrieve specific masks from a .npz file based on the given mode.

    Args:
        mask_file (str): Path to the .npz file containing the masks.
        mode (str): Specifies which mask to retrieve. Possible values are:
            - 'mask': Returns the standard nickel mask
            - 'fov_mask': Returns standard nickel mask w/ overscan columns
            - 'mask_cols_only': Returns nickel mask w/ only bad columns masked
            - 'fov_mask_cols_only': Returns nickel mask w/ overscan columns w/ only bad columns masked

    Returns:
        np.ndarray: The requested mask array corresponding to the specified mode.

    Raises:
        ValueError: If the mode is not one of the specified options.
    """
    loaded_masks = np.load(mask_file)
    
    if mode == 'mask':
        return loaded_masks['nickel_mask']
    elif mode == 'fov_mask':
        return loaded_masks['nickel_fov_mask']
    elif mode == 'mask_cols_only':
        return loaded_masks['nickel_mask_cols_only']
    elif mode == 'fov_mask_cols_only':
        return loaded_masks['nickel_fov_mask_cols_only']
    else:
        raise ValueError("Invalid mode. Expected one of: 'mask', 'fov_mask', 'mask_cols_only', 'fov_mask_cols_only'.")


def generate_masks():
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
    
    np.savez(mask_file, nickel_mask_cols_only=nickel_mask_cols_only,
                        nickel_mask=nickel_mask,
                        nickel_fov_mask_cols_only=nickel_fov_mask_cols_only,
                        nickel_fov_mask=nickel_fov_mask)
    
    return nickel_mask_cols_only, nickel_mask, nickel_fov_mask_cols_only, nickel_fov_mask


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
