
from pathlib import Path
import logging

from nickelpipeline.psf_analysis.moffat.psf_param_graphs import graph_psf_pars_bulk, graph_psf_pars_many, graph_psf_pars_individuals, multi_date_graph_fwhms_by_setting
from nickelpipeline.convenience.conditions import conditions_06_26, conditions_06_24, conditions
from nickelpipeline.convenience.log import adjust_global_logger

adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)

verbose = False

reddir1 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-24/reduced/')
reddir2 = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/')

# Filter directories to exclude those containing 'Focus' and 'NGC' in their names
directories1 = [dir for dir in reddir1.iterdir() 
                if ('NGC' not in str(dir) and 'Focus' not in str(dir))]
directories2 = [dir for dir in reddir2.iterdir() 
                if ('NGC' not in str(dir) and 'Focus' not in str(dir))]

# Analyze PSF parameters in bulk, many, and individual categories
# graph_psf_pars_bulk(directories, conditions_06_26, verbose=verbose)
# graph_psf_pars_many(directories, conditions_06_26, stack=True verbose=verbose)
# graph_psf_pars_many(directories, conditions_06_24, stack=False, verbose=verbose)
# graph_psf_pars_individuals(directories, verbose=verbose)

multi_date_graph_fwhms_by_setting({'06-26': directories2, '06-24': directories1}, conditions, stack=False, verbose=verbose)
