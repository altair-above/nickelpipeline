from pathlib import Path
from fits_convenience_class import Fits_Simple

def categories_from_conditions(condition_tuples, images):
    # Initialize an empty dictionary
    conditions = {}
    # Process the input list
    for value, (start, end) in condition_tuples:
        if value in conditions:
            conditions[value].append((start, end))
        else:
            conditions[value] = [(start, end)]

    # Convert ranges to lambda functions
    conditions = {width: (lambda ranges: lambda img_num: any(start <= img_num <= end for start, end in ranges))(ranges) for width, ranges in conditions.items()}
    # Use dictionary comprehension to create the categories
    categories = {width: [file.path for file in images if condition(file.image_num)] for width, condition in conditions.items()}
    return categories


def unzip_directories(directories, files=None, output_format='Fits_Simple'):
    if output_format == 'Path':
        output = Path
    elif output_format == 'Fits_Simple':
        output = Fits_Simple
    
    if files is not None:
        images = [output(file) for file in files]
    else:
        if not isinstance(directories, list):
            directories = [directories,]
        images = []
        for dir in directories:
            images += [output(file) for file in Path(dir).iterdir()]
    return images

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