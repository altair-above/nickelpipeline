import os
import glob

from IPython import embed

# fwhm(Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/NGC_3587_V/d1057_red.fits'))

import numpy as np
from matplotlib import pyplot

from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.visualization import SqrtStretch


from photutils import IRAFStarFinder
from photutils import CircularAperture
from photutils.psf import IterativePSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.psf import IntegratedGaussianPRF, SourceGrouper
from photutils.aperture import ApertureStats

from pathlib import Path
from astropy.table import Table
import warnings

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from convenience_funcs.convenience_funcs import Fits_Simple


def show(f, img, bkg, x, y, fwhm, ofile=None):
    positions = np.transpose((x, y))
    apertures = CircularAperture(positions, r=1.5*fwhm/2)

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))
    ax = fig.add_axes([0.08, 0.1, 0.8, img.shape[0]/img.shape[1]*0.8])
    cax = fig.add_axes([0.89, 0.1, 0.01, img.shape[0]/img.shape[1]*0.8])
    imp = ax.imshow(img - bkg, origin='lower', interpolation='nearest',
                    norm=ImageNormalize(stretch=AsinhStretch()))
    apertures.plot(axes=ax, color='w', lw=1.5, alpha=0.5)
    ax.set_title(f.name)
    pyplot.colorbar(imp, cax=cax)
    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

def plot_sources(f, img, x, y, fwhm, ofile=None):
    positions = np.transpose((x, y))
    apertures = CircularAperture(positions, r=1.5*fwhm/2)
    norm = ImageNormalize(stretch=SqrtStretch())
    pyplot.imshow(img, origin='lower', norm=norm,
            interpolation='nearest')
    pyplot.colorbar()
    apertures.plot(color='w', lw=0.5, alpha=0.5)
    pyplot.show()


def fwhm(image, default_fwhm=8.0, thresh=35, aper_size=10, local_bkg_range=(15,20)):
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    
    sig2fwhm = np.sqrt(8*np.log(2))
    # aper_size = 1.5*default_fwhm/2

    img = image.data
    img = np.delete(img, [255, 256, 783, 784, 1002], axis=1)
    img = img[5:-50, 5:-50]

    #----------------------------------------------------------------------
    # Do a first source detection using the default FWHM
    #----------------------------------------------------------------------
    _, median, std = sigma_clipped_stats(img, sigma=3.)
    starfind = IRAFStarFinder(fwhm=default_fwhm, threshold=thresh*std, minsep_fwhm=0.1, sky=0.0, peakmax=45000)
    sources = starfind(img - median)
    print(f'Found {len(sources)} sources in {image}.')

    # plot_sources(image, img, sources['xcentroid'], sources['ycentroid'], default_fwhm)
    
    # This determines the distance of each source to every other
    # source and removes sources that are too close to one another
    i, j = np.meshgrid(np.arange(len(sources)), np.arange(len(sources)),
                            indexing='ij')
    dist = np.sqrt(np.square(sources['xcentroid'][:,None] - sources['xcentroid'][None,:])
                + np.square(sources['ycentroid'][:,None] - sources['ycentroid'][None,:]))
    indx = (dist < default_fwhm/1.7) & (j > i)
    sources = sources[np.logical_not(np.any(indx, axis=0))]

    #----------------------------------------------------------------------
    # Attempt to improve the source detection by improving the FWHM estimate
    #----------------------------------------------------------------------
    
    win = int(np.ceil(2*default_fwhm))
    if win % 2 == 0:
        win += 1
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(img)
    # source finder
    iraffind = IRAFStarFinder(fwhm=default_fwhm, threshold=thresh*std, minsep_fwhm=0.1, peakmax=45000)
    # Grouping algorithm
    grouper = SourceGrouper(min_separation=2*default_fwhm)
    # This determine how to measure the background of the image
    mmm_bkg = MMMBackground()
    local_bkg = LocalBackground(*local_bkg_range, mmm_bkg)
    # This is the optimization algorithm
    fitter = LevMarLSQFitter()
    # This is the model of the PSF
    gaussian_prf = IntegratedGaussianPRF(sigma=default_fwhm/sig2fwhm)
    # Here is where I turn-off fixing the sigma to be a constant value
    gaussian_prf.sigma.fixed = False
    # This is the object that performs the photometry
    phot = IterativePSFPhotometry(finder=iraffind,
                                  grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=gaussian_prf,
                                  fitter=fitter, fit_shape=win, maxiters=4,
                                  fitter_maxiters=4, aperture_radius = aper_size)
    # print('first fit')
    # This is actually when the fitting is done
    phot_data = phot(data=img,
                     init_params=Table(sources['xcentroid', 'ycentroid', 'flux'],
                                       names=('x_0', 'y_0', 'flux_0')))
    indx = phot_data['iter_detected'] == 1
    # Here I'm taking the median FWHM of the all detected/fitted stars
    psf_fwhm_median = np.median(phot_data['sigma_fit'][indx])*sig2fwhm
    psf_fwhm_std = np.std(phot_data['sigma_fit'][indx]*sig2fwhm)

    # plot_sources(image, img, phot_data['x_fit'][indx], phot_data['y_fit'][indx],fwhm)
    
    #----------------------------------------------------------------------
    # Testing AperatureStats
    #----------------------------------------------------------------------
    
    coordinates = list(zip(phot_data['x_fit'][indx], phot_data['y_fit'][indx]))
    local_bkg_values = local_bkg(img, phot_data['x_fit'][indx], phot_data['y_fit'][indx])
    apertures = CircularAperture(coordinates, r=aper_size)
    aperstats = ApertureStats(img, apertures, local_bkg=local_bkg_values)
    
    aper_fwhm = aperstats.fwhm
    # aper_max = aperstats.max
    # print(f"FWHM calculated for each indivdual aperture: {aper_fwhm}")
    # print(f"Max calculated for each indivdual aperture: {aper_max}")
    # delete_indices = []
    # for index, max in np.ndenumerate(aper_max):
    #     if max < thresh*std:
    #         print('deleting')
    #         delete_indices.append(index)
    # aper_fwhm = np.delete(aper_fwhm, delete_indices)
    aper_fwhm_median = np.nanmedian(aper_fwhm)
    aper_fwhm_std = np.nanstd(aper_fwhm)
    # count = len(aper_fwhm)
    # for num in aper_fwhm:
    #     if np.isnan(num):
    #         count -= 1
    # print(f"{len(phot_data['sigma_fit'][indx])} stars used for PDF FWHM")
    # print(f"{count} stars used for aper FWHM")

    # print(f'{image}:')
    # print(f'FWHM calculated using PSF Photometry = {fwhm}')
    # print(f"FWHM calculated using Aperture Photometry = {aper_fwhm_median}")
    return (psf_fwhm_median, aper_fwhm_median, psf_fwhm_std, aper_fwhm_std)


def fwhm_batch(directories, default_fwhm=8.0, thresh=35, aper_size=10, local_bkg_range=(15,20)):
    if not isinstance(directories, list):
        directories = [directories,]
    images = []
    for dir in directories:
        dir = Path(dir)
        images += [Fits_Simple(file) for file in dir.iterdir()]
    
    results = Table(names=('Image', 'FWHM (PSF)', 'FWHM (Aper)', 'STD in FWHM (PSF)', 'STD in FWHM (Aper)'), dtype=('str', 'f4', 'f4', 'f4', 'f4'))
    # fwhm_results = []
    for image in images:
        fwhm_stats = fwhm(image, default_fwhm, thresh, aper_size, local_bkg_range)
        results.add_row((f"{image.filename} ({image.object})", *fwhm_stats))
        # fwhm_results.append((image, fwhms))
    
    for colname in results.colnames[1:]:
        results[colname].format = "{:.3f}"
    
    # for result in fwhm_results:
    #     print(f'{result[0]}:')
    #     print(f'FWHM calculated using PSF Photometry = {result[1][0]}')
    #     print(f"FWHM calculated using Aperture Photometry = {result[1][1]}")
    results.meta['title'] = f'default_fwhm={default_fwhm}, thresh={thresh}, aper_size={aper_size}, local_bkg_range={local_bkg_range}'
    print("Table Title:", results.meta['title'])
    print(results)
    return results

def parameter_test(images, default_fwhm=8.0, thresh=35, aper_size=10, local_bkg_range=(15,20)):
    psf_fwhms = []
    aper_fwhms = []
    diffs = []
    psf_stds = []
    aper_stds = []
    for image in images:
        fwhms = fwhm(image, default_fwhm, thresh, aper_size, local_bkg_range)
        psf_fwhms.append(fwhms[0])
        aper_fwhms.append(fwhms[1].value)
        diffs.append(fwhms[0]-fwhms[1].value)
        psf_stds.append(fwhms[2])
        aper_stds.append(fwhms[3].value)
    
    # print(f'Avg. FWHM calculated using PSF Photometry = {np.mean(psf_fwhms)}')
    # print(f"Avg. FWHM calculated using Aperture Photometry = {np.mean(aper_fwhms)}")
    # print(f"Difference between PSF and Aperture FWHM's = {np.mean(diffs)}")
    
    return np.mean(psf_fwhms), np.mean(aper_fwhms), np.mean(diffs), np.mean(psf_stds), np.mean(aper_stds), np.std(diffs)

def parameter_matrix(directories):
    if not isinstance(directories, list):
        directories = [directories,]
    images = []
    for dir in directories:
        dir = Path(dir)
        images += [Fits_Simple(file) for file in dir.iterdir()]
    
    # default_fwhms = [6.0, 7.0, 8.0, 9.0, 10.0]
    # threshs = [10, 30, 40, 50, 70]
    # aper_sizes = [6, 8, 10, 12, 15, 20, 30]
    
    default_fwhms = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    aper_sizes = [6, 8, 10, 12, 15]
    
    # default_fwhm vs thresh
    psf_fwhm = Table()
    aper_fwhm = Table()
    diffs = Table()
    psf_std = Table()
    aper_std = Table()
    diffs_std = Table()
    results = [psf_fwhm, aper_fwhm, diffs, psf_std, aper_std, diffs_std]
    
    for table in results:
        table['Info'] = [f'default_fwhm = {fwhm}' for fwhm in default_fwhms]
    for aper in aper_sizes:
        cols = [[] for _ in range(len(results))]
        for fwhm in default_fwhms:
            values = parameter_test(images, default_fwhm=fwhm, aper_size=aper)
            print(values)
            for i, val in enumerate(values):
                cols[i].append(val)
        for i, table in enumerate(results):
            table[f'aper_size = {aper}'] = cols[i]
    
    for table in results:
        for colname in table.colnames[1:]:
            table[colname].format = "{:.3f}"
            
        print(table)
    
    return results

# def parameter_matrix(directories, mode):
    if not isinstance(directories, list):
        directories = [directories,]
    images = []
    for dir in directories:
        dir = Path(dir)
        images += [Fits_Simple(file) for file in dir.iterdir()]
    
    default_fwhms = [6.0, 7.0, 8.0, 9.0, 10.0]
    # threshs = [10, 30, 40, 50, 70]
    aper_sizes = [6, 8, 10, 12, 15, 20, 30]
    tables = []
    
    for aper in aper_sizes:
        print(f'default_fwhm vs. thresh for aper_size = {aper}')
        # default_fwhm vs thresh
        fwhm_vs_thresh = Table()
        fwhm_vs_thresh['Info'] = [f'default_fwhm = {fwhm}' for fwhm in default_fwhms]
        for thresh in threshs:
            col = []
            for fwhm in default_fwhms:
                val = parameter_test(images, fwhm, thresh, aper)
                print(val)
                col.append(val)
            fwhm_vs_thresh[f'thresh = {thresh}'] = col
        print(fwhm_vs_thresh)
        tables.append(fwhm_vs_thresh)
    
    return tables



# result = fwhm_batch([Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_I'), Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V'), Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1530+057_I')])
warnings.filterwarnings('ignore')

# results = parameter_matrix([Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_I'),
#                            Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V'),
#                            Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1530+057_I'),
#                           ],
#                           0)
# results

# image_num

# results['Image'] = []
# results['FWHM (PSF)'] = []
# results['FWHM (Aperture)'] = []

# results['Image'].append(f"{image.filename} ({image.object})")
# results['FWHM (PSF)'].append(fwhms[0])
# results['FWHM (Aperture)'].append(fwhms[1])