from IPython import embed

# fwhm(Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/NGC_3587_V/d1057_red.fits'))
# Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V')
# Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V/d1079_red.fits')


import numpy as np
from matplotlib import pyplot

from astropy.io import fits
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
from matplotlib import pyplot as plt

from nickelpipeline.convenience.fits_class import Fits_Simple

default_fwhm_default=8.0
thresh_default=35
aper_size_default=8
local_bkg_range_default=(15,20)

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

def plot_sources(f, img, x, y, given_fwhm, ofile=None):
    positions = np.transpose((x, y))
    apertures = CircularAperture(positions, r=1.5*given_fwhm)
    norm = ImageNormalize(stretch=SqrtStretch())
    pyplot.imshow(img, origin='lower', norm=norm,
                  interpolation='nearest')
    pyplot.colorbar()
    apertures.plot(color='w', lw=1.5, alpha=0.5)
    pyplot.show()


def calc_fwhm(image, default_fwhm=default_fwhm_default, thresh=thresh_default,
         aper_size=aper_size_default, local_bkg_range=local_bkg_range_default,
         which_source=None):
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
    starfind = IRAFStarFinder(fwhm=default_fwhm, threshold=thresh*std,
                              minsep_fwhm=0.1, sky=0.0, peakmax=45000,)
    sources = starfind(img - median)
    print(f'Found {len(sources)} sources in {image}.')
    # This determines the distance of each source to every other
    # source and removes sources that are too close to one another
    i, j = np.meshgrid(np.arange(len(sources)), np.arange(len(sources)),
                            indexing='ij')
    dist = np.sqrt(np.square(sources['xcentroid'][:,None]
                             - sources['xcentroid'][None,:])
                + np.square(sources['ycentroid'][:,None]
                            - sources['ycentroid'][None,:]))
    indx = (dist < default_fwhm/1.7) & (j > i)
    sources = sources[np.logical_not(np.any(indx, axis=0))]
    # return sources

    #----------------------------------------------------------------------
    # Attempt to improve the source detection by improving the FWHM estimate
    #----------------------------------------------------------------------
    win = int(np.ceil(2*default_fwhm))
    if win % 2 == 0:
        win += 1
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(img)
    # Source finder
    iraffind = IRAFStarFinder(fwhm=default_fwhm, threshold=thresh*std, 
                              minsep_fwhm=0.1, peakmax=45000)
    grouper = SourceGrouper(min_separation=2*default_fwhm)  # Grouping algorithm
    mmm_bkg = MMMBackground()   # Background-determining function
    local_bkg = LocalBackground(*local_bkg_range, mmm_bkg)
    fitter = LevMarLSQFitter()  # This is the optimization algorithm
    # This is the model of the PSF
    gaussian_prf = IntegratedGaussianPRF(sigma=default_fwhm/sig2fwhm)
    gaussian_prf.sigma.fixed = False    # Turn off fixing the sigma to be a constant
    # This is the object that performs the photometry
    phot = IterativePSFPhotometry(finder=iraffind,
                                  grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=gaussian_prf,
                                  fitter=fitter, fit_shape=win, maxiters=4,
                                  fitter_maxiters=4, aperture_radius = aper_size)
    # This is actually when the fitting is done
    phot_data = phot(data=img,
                     init_params=Table(sources['xcentroid', 'ycentroid', 'flux'],
                                       names=('x_0', 'y_0', 'flux_0')))
    
    indx = phot_data['iter_detected'] == 1
    fwhm = np.median(phot_data['sigma_fit'][indx])*sig2fwhm
    
    #----------------------------------------------------------------------
    # Refit using the "improved" FWHM

    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=thresh*std,
                              minsep_fwhm=0.1, peakmax=45000)
    grouper = SourceGrouper(min_separation=2*fwhm)
    gaussian_prf = IntegratedGaussianPRF(sigma=fwhm/sig2fwhm)
    gaussian_prf.sigma.fixed = False
    phot = IterativePSFPhotometry(finder=iraffind,
                                  grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=gaussian_prf,
                                  fitter=fitter, fit_shape=win, maxiters=2,
                                  fitter_maxiters=2, aperture_radius = aper_size)
    print('second fit')
    phot_data = phot(data=img,
                     init_params=Table([phot_data['x_fit'][indx],
                                        phot_data['y_fit'][indx],
                                        phot_data['flux_fit'][indx]],
                                        names=('x_0', 'y_0', 'flux_0')))
    #----------------------------------------------------------------------
    
    brightest_parameter = 'peak'
    # Extracts only the which_source brightest src from phot_data to analyze
    if which_source is not None:
        indices_by_peak = sources.argsort(brightest_parameter)
        chosen_star = indices_by_peak[which_source]
        print(brightest_parameter + f" of star chosen = {sources[brightest_parameter][chosen_star]}")
        phot_data = phot_data[indices_by_peak[chosen_star]]
    
    # # Extracts only the which_source brightest src from phot_data to analyze
    # if which_source is not None:
    #     phot_data = phot_data[phot_data['iter_detected']]
    #     indices_by_peak = phot_data.argsort('flux_init')
    #     chosen_star = indices_by_peak[which_source]
    #     print(f"flux of star chosen = {phot_data['flux_init'][chosen_star]}")
    #     phot_data = phot_data[indices_by_peak[chosen_star]]
    
    indx = phot_data['iter_detected'] == 1
    psf_fwhm_median = np.median(phot_data['sigma_fit'][indx])*sig2fwhm
    psf_fwhm_std = np.std(phot_data['sigma_fit'][indx]*sig2fwhm)

    plot_sources(image, img, phot_data['x_fit'][indx], phot_data['y_fit'][indx],psf_fwhm_median)
    
    #----------------------------------------------------------------------
    # Testing AperatureStats
    #----------------------------------------------------------------------
    coordinates = list(zip(phot_data['x_fit'][indx], phot_data['y_fit'][indx]))
    local_bkg_values = local_bkg(img, phot_data['x_fit'][indx], phot_data['y_fit'][indx])
    apertures = CircularAperture(coordinates, r=aper_size)
    aperstats = ApertureStats(img, apertures, local_bkg=local_bkg_values)
    
    aper_fwhm = aperstats.fwhm
    aper_fwhm_median = np.nanmedian(aper_fwhm)
    aper_fwhm_std = np.nanstd(aper_fwhm)
    
    #----------------------------------------------------------------------
    # Return
    #----------------------------------------------------------------------
    return (psf_fwhm_median, aper_fwhm_median, psf_fwhm_std, aper_fwhm_std)


def single_star_compare(directory, which_sources=0):
    dir = Path(directory)
    images = [Fits_Simple(file) for file in dir.iterdir()]
    
    if not isinstance(which_sources, list):
        which_sources = [which_sources]
    
    data = {}
    for source in which_sources:
        image_nums = []
        psf_fwhms = []
        aper_fwhms = []
        
        for image in images:
            image_nums.append(image.filename[3:5])
            psf_fwhm_median, aper_fwhm_median, _, _ = calc_fwhm(image, which_source=source)
            psf_fwhms.append(psf_fwhm_median)
            aper_fwhms.append(aper_fwhm_median.value)
        
        source_data = {}
        source_data['image_nums'] = image_nums
        source_data['psf_fwhms'] = psf_fwhms
        source_data['aper_fwhms'] = aper_fwhms
        
        data[source] = source_data
    
    blue_range = ['#292f56', '#214471', '#005a8a', '#00719c', '#0087a5', 
                  '#009ead', '#00b6b0', '#00cfab', '#12e69e', '#70fa8e']
    red_range = ["#850505", "#962c07", "#a7470a", "#b66010", "#c4791a",
                 "#d09226", "#dcac35", "#e6c546", "#efe05a", "#f7fa70"]

    # Plot the scatter plot
    for i, source in enumerate(which_sources):
        plt.scatter(image_nums, data[source]['psf_fwhms'], c=blue_range[i],
                    label=f'PSF FWHM - Star {source}', s=15)
    for i, source in enumerate(which_sources):
        plt.scatter(image_nums, data[source]['aper_fwhms'], c=red_range[i],
                    label=f'Aper FWHM - Star {source}', s=15)

    # Add titles and labels
    plt.title(f'PSF FWHM and Aper FWHM of the {len(which_sources)} brightest stars in {image.object}')
    plt.xlabel('Image Number (d1075.fits = #75)')
    plt.ylabel('FWHM (px)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()
        


def fwhm_batch(directories, default_fwhm=default_fwhm_default,
               thresh=thresh_default, aper_size=aper_size_default, 
               local_bkg_range=local_bkg_range_default):
    if not isinstance(directories, list):
        directories = [directories,]
    images = []
    for dir in directories:
        dir = Path(dir)
        images += [Fits_Simple(file) for file in dir.iterdir()]
    
    results = Table(names=('Image', 'FWHM (PSF)', 'FWHM (Aper)', 'STD in FWHM (PSF)', 'STD in FWHM (Aper)'), dtype=('str', 'f4', 'f4', 'f4', 'f4'))
    for image in images:
        fwhm_stats = calc_fwhm(image, default_fwhm=default_fwhm, thresh=thresh, aper_size=aper_size, local_bkg_range=local_bkg_range)
        results.add_row((f"{image.filename} ({image.object})", *fwhm_stats))
    
    for colname in results.colnames[1:]:
        results[colname].format = "{:.3f}"
    
    results.meta['title'] = f'default_fwhm={default_fwhm}, thresh={thresh}, aper_size={aper_size}, local_bkg_range={local_bkg_range}'
    print("Table Title:", results.meta['title'])
    print(results)
    return results

def parameter_test(images, default_fwhm=default_fwhm_default,
                   thresh=thresh_default, aper_size=aper_size_default,
                   local_bkg_range=local_bkg_range_default):
    psf_fwhms = []
    aper_fwhms = []
    diffs = []
    psf_stds = []
    aper_stds = []
    for image in images:
        fwhms = calc_fwhm(image, default_fwhm, thresh, aper_size, local_bkg_range)
        psf_fwhms.append(fwhms[0])
        aper_fwhms.append(fwhms[1].value)
        diffs.append(fwhms[0]-fwhms[1].value)
        psf_stds.append(fwhms[2])
        aper_stds.append(fwhms[3].value)
    
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
    # default_fwhms = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    # aper_sizes = [6, 8, 10, 12, 15]
    default_fwhms = [5.0, 6.0, 7.0, 8.0]
    aper_sizes = [6, 8, 10, 12]
    
    results = [Table() for _ in range(6)]
    
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


# result = fwhm_batch([Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_I'), Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V'), Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1530+057_I')])
warnings.filterwarnings('ignore')

# single_star_compare(Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V'),
#                     [0,1,2,3])

# results = parameter_matrix([Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_I'),
#                            Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V'),
#                            Path('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1530+057_I'),
#                           ],
#                           0)
# results
