from pathlib import Path
from nickelpipeline.convenience.conditions import conditions_06_26
from nickelpipeline.astrometry.astrometric_error import graph_topographic, graph_topographic_individuals

reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26/reduced/')
directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]

verbose = False

graph_topographic(directories, condition_tuples=conditions_06_26, fast=True, 
                  verbose=verbose, include_smooth=True, include_srcs=True)
# graph_topographic(directories, condition_tuples=conditions_06_26, fast=False, 
#                   verbose=verbose, include_smooth=True, include_srcs=True)

graph_topographic_individuals(directories, condition_tuples=conditions_06_26, 
                              frac=0.5, fast=True, include_smooth=True, 
                              include_srcs=True)