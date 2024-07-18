
from pathlib import Path
import warnings

from nickelpipeline.convenience.conditions import conditions_06_26
from nickelpipeline.psf_analysis.gaussian.fwhm_graphs_contour import param_graph_by_category, param_graph_individuals

# Suppress warnings
warnings.filterwarnings('ignore')
    
reddir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26/reduced/')
# directories = [dir for dir in reddir.iterdir() if ('NGC' not in str(dir) and 
#                                                    'Focus' not in str(dir) and
#                                                    '_B' not in str(dir))]
directories = [dir for dir in reddir.iterdir() if ('NGC' not in str(dir) and 
                                                   'Focus' not in str(dir))]
# fwhm_contour_by_category(directories, conditions_06_26)
# param_graph_by_category('fwhm residuals', directories, conditions_06_26, 
#                           include_smooth=True, include_srcs=False)
param_graph_by_category('fwhm residuals', directories, conditions_06_26, 
                          include_smooth=True, include_srcs=True)
param_graph_by_category('fwhm', directories, conditions_06_26, 
                          include_smooth=True, include_srcs=True)
# param_graph_individuals('fwhm residuals', directories, conditions_06_26, 
#                         include_smooth=True, include_srcs=True)
# param_graph_individuals('fwhm residuals', directories, conditions_06_26, 
#                         include_smooth=False, include_srcs=True)