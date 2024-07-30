

from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.photometry.starfind import analyze_sources
from nickelpipeline.photometry.accuracy_check import check_stats


adjust_global_logger('DEBUG', __name__)
logger = logging.getLogger(__name__)

from nickelpipeline.photometry.make_test_img import make_img

fluxes = range(5000, 50000, 10000)
names = [f'test_img_{flux}.fits' for flux in fluxes]
phot_datas = {}
for flux, filename in zip(fluxes, names):
    make_img(flux, 4.5, 3.5, name=filename)

for flux, filename in zip(fluxes, names):
    test_img = Path(filename)
    phot_datas[flux] = analyze_sources(test_img, plot=True)
    
for flux, filename in zip(fluxes, names):
    check_stats(phot_datas[flux], flux)

# test_img = Path('test_img1.fits')
# phot_data_1 = analyze_sources(test_img, plot=True)

# test_img = Path('test_img2.fits')
# phot_data_2 = analyze_sources(test_img, plot=True)

# image = Path('C:/Users/allis/Documents/2024-2025_Local/Akamai_Internship/nickelpipeline/all_testing_code/test-data-06-26-2/reduced/110_232_R/d1040.fits_red.fits')
# phot_data_3 = analyze_sources(image, plot=True)

# check_stats(phot_data_1, 20000)
# check_stats(phot_data_2, 35000)
