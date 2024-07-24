from pathlib import Path
from nickelpipeline.reduction.reduction_ccdproc import reduce_all
from nickelpipeline.convenience.display_fits import display_many_nickel
from nickelpipeline.convenience.fits_class import Fits_Simple


rawdir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/raw/')

redfiles = reduce_all(rawdir=rawdir, save_inters=True, excl_files=['d1113'])
redfiles = reduce_all(file_table_in='reduction_files_table.yml', save_inters=True)

# redfiles = reduce_all(rawdir, True, exclude_files=['d1113'])

# display_many_nickel(redfiles)


# images = [Fits_Simple(image) for image in rawdir.iterdir()]
# images = [image.path for image in images if image.object == 'Flat']
# display_many_nickel(images)

# procdir = rawdir.parent/'processing'
# display_many_nickel([procdir/'master_bias.fits', procdir/'master_flat_B.fits', procdir/'master_flat_R.fits'])



# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.visualization import ZScaleInterval

# interval = ZScaleInterval()
# for i in range(0,50,2):
#     plt.figure(figsize=(12, 12))
#     vmin, vmax = interval.get_limits(np.array(redfiles[i].data))
#     plt.imshow(np.array(redfiles[i].data), origin='lower', vmin=vmin, vmax=vmax)
#     plt.show()
#     plt.figure(figsize=(12, 12))
#     plt.imshow(redfiles[i].mask, origin='lower')
#     plt.show()




# redfiles = [Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1026.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1027.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1039.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1040.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1050.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1051.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1061.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1062.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1071.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1072.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1102.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1103.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/NGC_6543_R/d1106.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/NGC_6543_R/d1107.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/NGC_6543_R/d1108.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1028.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1029.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1041.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1042.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1052.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1053.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1063.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1064.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1073.fits_red.fits'),
#             Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_B/d1074.fits_red.fits'),]

