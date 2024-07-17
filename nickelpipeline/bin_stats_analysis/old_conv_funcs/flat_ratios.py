#######################################################################
###### Calculate the ratio of flats relative to a reference flat ######
#######################################################################

# bias & overscan subtract all flats. For each series of 5, pick one and take ratio of all other 4
# Calculate what the ratio is for ^ (to find a value for each)
# Do for the four sets of flats

import numpy as np
from astropy.io import fits
from pathlib import Path
from fits_convenience_class import Fits_Simple

all_flats = []

# Path to directory with bias & overscan subtracted flats
subtracted_dir = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-proc/unbias")

for i in range(11,31):
    this_flat = subtracted_dir / f'd10{i}.fits'
    all_flats.append(Fits_Simple(this_flat))

categ_flats = {}
categ_ratios_complex = {}
categ_ratios_simple = {}

for flat in all_flats:
    flat.data = np.delete(flat.data, [256, 783, 784, 1002], axis=1)
    flat.data = flat.data[5:-5, 5:-5]
        
    categ_flats.setdefault((flat.filtnam, flat.exptime), []).append(flat)
    
    categ_ratios_complex.setdefault((flat.filtnam, flat.exptime), []).append(None)
    categ_ratios_simple.setdefault((flat.filtnam, flat.exptime), []).append(None)

print('Let complex_ratio = the mean of the ratios between every pixel in image [x] relative to the baseline image of its type')
print('Let simple_ratio = the ratio of the mean of image [x] to the mean of the baseline image')

for categ, group_flats in categ_flats.items():
    baseline = group_flats[0]
    rest = group_flats[1:]
    print(f"Filt = {categ[0]}, Exposure Time = {categ[1]}")
    
    baseline_mean = np.mean(baseline.data)
    for i, flat in enumerate(rest):
        complex_ratio = np.mean(flat.data / baseline.data)
        simple_ratio = np.mean(flat.data) / baseline_mean
        categ_ratios_complex[categ][i] = complex_ratio
        categ_ratios_simple[categ][i] = simple_ratio

        print(f'Image {flat.filename}: complex_ratio = {complex_ratio:.6f} and simple_ratio = {simple_ratio:.6f}')
        



