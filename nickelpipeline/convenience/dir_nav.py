from pathlib import Path
from nickelpipeline.convenience.fits_class import Fits_Simple

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


# def unzip_directories(path_list: list, files: list = None, output_format: str = 'Fits_Simple', allow_exceptions: bool = False) -> list:
#     """
#     Unzips a list of directories/files to access all files inside 
#     and returns a list of image objects.

#     Args:
#         path_list (list): List of paths to unzip.
#         files (list, optional): List of files to unzip. Defaults to None.
#         output_format (str, optional): The format of the output objects. Defaults to 'Fits_Simple'.

#     Returns:
#         list: List of image objects.
#     """
#     if output_format == 'Path':
#         output = Path
#     elif output_format == 'Fits_Simple':
#         output = Fits_Simple
    
#     if files is not None:
#         images = [output(file) for file in files]
#         print("'files' parameter is not ideal. 'directories' handles lists that contain both directories & files")
#     else:
#         images = []
#         for elem_path in path_list:
#             elem_path = Path(elem_path)
#             if elem_path.is_dir():
#                 if allow_exceptions:
#                     for file in Path(elem_path).iterdir():
#                         try:
#                             images.append(Fits_Simple(file))
#                         except (KeyError, OSError):
#                             continue
#                 else:
#                     images += [output(file) for file in elem_path.iterdir()]
#             elif elem_path.is_file():
#                 images.append(output(elem_path))
#     return images


def unzip_directories(path_list: list, output_format: str = 'Fits_Simple', 
                      allow_exceptions: bool = False) -> list:
    """
    Unzips a list of directories/files to access all files inside 
    and returns a list of image objects.

    Args:
        path_list (list): List of paths to unzip.
        output_format (str, optional): The format of the output objects. Defaults to 'Fits_Simple'.
        allow_exceptions (bool): Whether to allow for file not found errors

    Returns:
        list: List of image objects.
    """
    if output_format == 'Path':
        output = Path
    elif output_format == 'Fits_Simple':
        output = Fits_Simple
    
    images = []
    for elem_path in path_list:
        elem_path = Path(elem_path)
        if elem_path.is_dir():
            if allow_exceptions:
                for file in Path(elem_path).iterdir():
                    try:
                        images.append(Fits_Simple(file))
                    except (KeyError, OSError):
                        pass
            else:
                images += [output(file) for file in elem_path.iterdir()]
        elif elem_path.is_file():
            images.append(output(elem_path))
    return images

