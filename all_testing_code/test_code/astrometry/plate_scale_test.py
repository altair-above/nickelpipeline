from pathlib import Path
from nickelpipeline.convenience.conditions import conditions_06_26
from nickelpipeline.astrometry.plate_scale import graph_plate_scale_by_setting, avg_plate_scale


def testing():
    reddir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26/reduced/')
    directories = [dir for dir in reddir.iterdir() if ('Focus' not in str(dir) and 'NGC' not in str(dir))]
    graph_plate_scale_by_setting(directories, condition_tuples=conditions_06_26, fast=True)
    return