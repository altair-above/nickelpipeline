import warnings
from pathlib import Path
from nickelpipeline.psf_analysis.gaussian.fwhm_graphs import graph_fwhms_by_image, graph_fwhms_by_setting, multi_date_graph_fwhms_by_setting
from nickelpipeline.convenience.conditions import conditions_06_26, conditions

from nickelpipeline.reduction.reduction_mod_basic import generate_reduction_files, process_single


warnings.filterwarnings('ignore')


# directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_199_{filt}') for filt in ['B', 'V', 'R', 'I']]

# rawdir = Path("")
rawdir = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-24/raw")  # path to directory with raw data

# image_to_analyze = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw/d1079.fits")  # path to directory with raw data

# # Initial configuration
# directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/PG1530+057_{filt}') for filt in ['B', 'V', 'R', 'I']]
# # Night 1 mod
# directories = [Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/109_231_{filt}') for filt in ['R']]

reddir_05 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-05-12/reduced/')
directories = [dir for dir in reddir_05.iterdir() if 'flat' not in str(dir)]

reddir_06_24 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-24/reduced/')
reddir_06_26 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26/reduced/')
directories_06_24 = [dir for dir in reddir_06_26.iterdir() if ('Focus' not in str(dir) and 'Po' not in str(dir))]
directories_06_26 = [dir for dir in reddir_06_26.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]

graph_fwhms_by_setting(directories_06_26, condition_tuples=conditions_06_26)
graph_fwhms_by_image(directories_06_26, '06-26-24')

dir_dict = {'06-24': directories_06_24, '06-26': directories_06_26}
multi_date_graph_fwhms_by_setting(dir_dict, conditions,)

bias, flat = generate_reduction_files(rawdir)
image_to_analyze = process_single(rawdir/'d1051.fits', bias, flat)
fwhm = graph_fwhms_by_image([image_to_analyze])

