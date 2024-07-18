from pathlib import Path
from nickelpipeline.reduction.reduction_pipeline import reduce_all

rawdir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/raw/')

redfiles = reduce_all(rawdir)
