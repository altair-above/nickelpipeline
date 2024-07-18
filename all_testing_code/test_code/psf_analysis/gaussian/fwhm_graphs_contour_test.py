
from pathlib import Path

from nickelpipeline.convenience.conditions import conditions_06_26
from nickelpipeline.psf_analysis.gaussian.fwhm_graphs_contour import param_graph_by_category, param_graph_individuals

import warnings
# Suppress warnings
warnings.filterwarnings('ignore')

def testing():
    directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_199_{filt}') for filt in ['B', 'V', 'R', 'I']]

    # rawdir = Path("")
    # rawdir = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw")  # path to directory with raw data

    # image_to_analyze = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw/d1079.fits")  # path to directory with raw data

    # Initial configuration
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/PG1530+057_{filt}') for filt in ['B', 'V', 'R', 'I']]
    # Night 1 mod
    # directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_231_{filt}') for filt in ['R']]

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if 'flat' not in str(dir)]

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]
    # graph_fwhms_by_setting(directories, condition_tuples=conditions_06_26)
    
    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]
    # graph_fwhms_by_setting(directories, condition_tuples=conditions_06_26)

    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('109_199' in str(dir))]
    # graph_fwhms_by_image(directories, '06-26-24')

    # bias, flat = generate_reduction_files(rawdir)
    # fwhm = fwhm_from_raw(image_to_analyze, bias, flat)
    
    # reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/raw-reduced/')
    # directories = [dir for dir in reddir.iterdir() if ('NGC' not in str(dir) and 'Focus' not in str(dir))]
    # fwhm_contour_by_category(directories, conditions_06_26)
    
reddir = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26/reduced/')
directories = [dir for dir in reddir.iterdir() if ('NGC' not in str(dir) and 'Focus' not in str(dir))]
# fwhm_contour_by_category(directories, conditions_06_26)
param_graph_by_category('fwhm residuals', directories, conditions_06_26, 
                          include_smooth=True, include_srcs=False)
param_graph_by_category('fwhm residuals', directories, conditions_06_26, 
                          include_smooth=True, include_srcs=True)
param_graph_by_category('fwhm', directories, conditions_06_26, 
                          include_smooth=True, include_srcs=True)
param_graph_individuals('fwhm residuals', directories, conditions_06_26, 
                        include_smooth=True, include_srcs=True)