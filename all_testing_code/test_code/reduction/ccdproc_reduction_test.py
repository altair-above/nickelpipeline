from pathlib import Path
from nickelpipeline.reduction.reduction_ccdproc import reduce_all
from nickelpipeline.convenience.display_fits import display_many_nickel
from nickelpipeline.convenience.fits_class import Fits_Simple

rawdir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/raw/')

redfiles = reduce_all(rawdir, True)

# display_many_nickel(redfiles)

# images = [Fits_Simple(image) for image in rawdir.iterdir()]
# images = [image.path for image in images if image.object == 'Flat']
# display_many_nickel(images)

# procdir = rawdir.parent/'processing'
# display_many_nickel([procdir/'master_bias.fits', procdir/'master_flat_B.fits', procdir/'master_flat_R.fits'])