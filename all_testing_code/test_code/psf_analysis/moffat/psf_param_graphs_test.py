
from pathlib import Path
import logging

from nickelpipeline.psf_analysis.moffat.psf_param_graphs import graph_psf_pars_bulk, graph_psf_pars_many, graph_psf_pars_individuals
from nickelpipeline.convenience.conditions import conditions_06_26
from nickelpipeline.convenience.log import adjust_global_logger

adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)

verbose = False

reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/')

# Filter directories to exclude those containing 'Focus' and 'NGC' in their names
directories = [dir for dir in reddir.iterdir() 
                if ('NGC' not in str(dir) and 'Focus' not in str(dir))]

# Analyze PSF parameters in bulk, many, and individual categories
graph_psf_pars_bulk(directories, conditions_06_26, verbose=verbose)
graph_psf_pars_many(directories, conditions_06_26, verbose=verbose)
graph_psf_pars_individuals(directories, verbose=verbose)