from pathlib import Path
import logging

from nickelpipeline.convenience.log import adjust_global_logger
from nickelpipeline.photometry.psf_photometry import psf_analysis
from all_testing_code.test_code.photometry.test_img import make_img, check_stats


adjust_global_logger('INFO', __name__)
logger = logging.getLogger(__name__)


# fluxes = list(range(200, 1000, 200))
fluxes = list(range(1000, 8000, 2000))
fluxes += list(range(10000, 50000, 10000))
names = [f'test_imgs/test_img_{flux}.fits' for flux in fluxes]
phot_datas = {}
for flux, filename in zip(fluxes, names):
    make_img(flux, 4.5, 3.5, name=filename, show_img=False)

for flux, filename in zip(fluxes, names):
    test_img = Path(filename)
    try:
        phot_datas[flux] = psf_analysis(test_img, plot_final=True, plot_inters=True)
    except:
        logger.warning("No sources found (probably)")
        continue
    
for flux in phot_datas.keys():
    check_stats(phot_datas[flux], flux)