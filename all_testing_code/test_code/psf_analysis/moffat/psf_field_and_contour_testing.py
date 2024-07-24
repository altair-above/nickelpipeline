
from pathlib import Path
from nickelpipeline.psf_analysis.moffat.psf_field_and_contour import param_graph_by_category, fit_field_by_category
from nickelpipeline.convenience.conditions import conditions_06_26, conditions_06_24


verbose = False

# Define the path to the directory containing raw reduced data
reddir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/')

# Filter directories to exclude those containing 'Focus' and 'NGC' in their names
directories = [dir for dir in reddir.iterdir() if ('FOCUS' not in str(dir) and 'NGC' not in str(dir))]
# directories = [dir for dir in reddir.iterdir() if ('NGC' not in str(dir) and 
#                                                    'FOCUS' not in str(dir) and
#                                                    '_B' not in str(dir))]
# Fit field by category and plot parameter contours for different parameters
fit_field_by_category(directories, conditions_06_26, verbose=verbose, include_srcs=True)
param_graph_by_category('fwhm residuals', directories, conditions_06_26, verbose=verbose, include_srcs=True)
param_graph_by_category('fwhm', directories, conditions_06_26, verbose=verbose, include_srcs=True)
param_graph_by_category('ecc', directories, conditions_06_26, verbose=verbose, include_srcs=True)
param_graph_by_category('phi', directories, conditions_06_26, verbose=verbose, include_srcs=True)


# reddir_05 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-05-12/reduced/')
# directories = [dir for dir in reddir_05.iterdir() if 'flat' not in str(dir)]

# reddir_06_24 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-24/reduced/')
# reddir_06_26 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26/reduced/')
# directories_06_24 = [dir for dir in reddir_06_26.iterdir() if ('Focus' not in str(dir) and 'Po' not in str(dir))]
# directories_06_26 = [dir for dir in reddir_06_26.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]

# param_graph_by_category('fwhm', directories_06_24, conditions_06_24, verbose=verbose, include_srcs=True)
# param_graph_by_category('ecc', directories_06_24, conditions_06_24, verbose=verbose, include_srcs=True)
# param_graph_by_category('phi', directories_06_24, conditions_06_24, verbose=verbose, include_srcs=True)

