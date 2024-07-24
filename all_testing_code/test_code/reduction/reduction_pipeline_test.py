from pathlib import Path
from nickelpipeline.reduction.old_basic.reduction_pipeline import reduce_all
from nickelpipeline.convenience.display_fits import display_many_nickel

rawdir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26/raw/')

redfiles = reduce_all(rawdir)

display_many_nickel(redfiles)

# reddir = rawdir.parent / 'reduced'
# directories = [dir for dir in reddir.iterdir()]

# display_many_nickel(directories)