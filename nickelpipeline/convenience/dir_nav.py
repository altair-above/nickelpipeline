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


# def unzip_directories(directories: Union[str, list], files: list = None, output_format: str = 'Fits_Simple') -> list:
#     """
#     Unzips directories and returns a list of image objects.

#     Args:
#         directories (Union[str, list]): Directories to unzip.
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
#     else:
#         if not isinstance(directories, list):
#             directories = [directories]
#         images = []
#         for dir in directories:
#             images += [output(file) for file in Path(dir).iterdir()]
#     return images



conditions_06_26 = [(1.375, (65, 74)),
                    (1.624, (22, 31)),
                    (1.625, (88, 105)),
                    (1.875, (33, 42)),
                    (2.625, (43, 53)),
                    (3.375, (54, 64)),
                    ]

conditions_06_24 = [(1.375, (53, 60)),
                    (1.625, (1, 52)),
                    (1.625, (88, 105))
                    ]

conditions = {'06-26': conditions_06_26, '06-24': conditions_06_24}