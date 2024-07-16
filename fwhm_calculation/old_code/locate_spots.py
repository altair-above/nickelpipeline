
import os
import glob

from IPython import embed

import numpy
from matplotlib import pyplot, ticker
from scipy import optimize

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy import table
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling.fitting import LevMarLSQFitter

from photutils import IRAFStarFinder
from photutils import CircularAperture
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import IntegratedGaussianPRF, DAOGroup

def show(f, img, bkg, x, y, fwhm, ofile=None):
    positions = numpy.transpose((x, y))
    apertures = CircularAperture(positions, r=1.5*fwhm/2)

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))
    ax = fig.add_axes([0.08, 0.1, 0.8, img.shape[0]/img.shape[1]*0.8])
    cax = fig.add_axes([0.89, 0.1, 0.01, img.shape[0]/img.shape[1]*0.8])
    imp = ax.imshow(img - bkg, origin='lower', interpolation='nearest',
                    norm=ImageNormalize(stretch=AsinhStretch()))
    apertures.plot(axes=ax, color='w', lw=1.5, alpha=0.5)
    ax.set_title(f)
    pyplot.colorbar(imp, cax=cax)
    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)


def find_spots(spot_file):

    sig2fwhm = numpy.sqrt(8*numpy.log(2))

#    ssc_files = numpy.sort(glob.glob('ssc_*.fits'))
    ssc_files = ['ssc_143054_158.fits', 'ssc_143142_182.fits', 'ssc_143211_867.fits',
                 'ssc_143222_937.fits', 'ssc_143233_987.fits', 'ssc_143244_027.fits',
                 'ssc_143255_089.fits', 'ssc_144106_390.fits', 'ssc_144116_408.fits',
                 'ssc_144127_473.fits', 'ssc_144138_564.fits', 'ssc_144148_619.fits',
                 'ssc_144601_225.fits', 'ssc_144611_285.fits', 'ssc_144622_414.fits',
                 'ssc_144633_496.fits', 'ssc_144644_576.fits', 'ssc_145035_031.fits',
                 'ssc_145046_091.fits', 'ssc_145057_144.fits', 'ssc_145108_221.fits',
                 'ssc_145118_262.fits', 'ssc_145537_127.fits', 'ssc_145547_144.fits',
                 'ssc_145558_187.fits', 'ssc_145609_250.fits', 'ssc_145620_358.fits',
                 'ssc_145941_752.fits', 'ssc_145952_792.fits', 'ssc_150002_822.fits',
                 'ssc_150013_927.fits', 'ssc_150024_994.fits', 'ssc_150513_400.fits',
                 'ssc_150523_485.fits', 'ssc_150534_535.fits', 'ssc_150545_579.fits',
                 'ssc_150556_660.fits']

    # Ignore this
    ssc_log_el = [85., 85., 85., 85., 85., 85., 85., 60., 60., 60., 60., 60., 45., 45., 45.,
                  45., 45., 30., 30., 30., 30., 30., 45., 45., 45., 45., 45., 60., 60., 60.,
                  60., 60., 85., 85., 85., 85., 85.]

    default_fwhm = 5.0
    src = None

    for k, (f, el) in enumerate(zip(ssc_files, ssc_log_el)):
#        if f != 'ssc_144138_564.fits':
#            continue
#        if f != 'ssc_144622_414.fits':
#            continue
#        if f != 'ssc_143211_867.fits':
#            continue
#        if f != 'ssc_144601_225.fits':
#            continue
#        if f != 'ssc_145118_262.fits':
#            continue

        with fits.open(f) as hdu:
            img = hdu[0].data

        #----------------------------------------------------------------------
        # Do a first source detection using the default FWHM
        mean, median, std = sigma_clipped_stats(img, sigma=3.)
        starfind = IRAFStarFinder(fwhm=default_fwhm, threshold=5.*std, minsep_fwhm=0.1, sky=0.0)
        sources = starfind(img - median)
        print(f'Found {len(sources)} sources in {f}.')

        show(f, img, median, sources['xcentroid'], sources['ycentroid'], default_fwhm)

        # This determines the distance of each source to every other
        # source and removes sources that are too close to one another
        i, j = numpy.meshgrid(numpy.arange(len(sources)), numpy.arange(len(sources)),
                              indexing='ij')
        dist = numpy.sqrt(numpy.square(sources['xcentroid'][:,None] - sources['xcentroid'][None,:])
                    + numpy.square(sources['ycentroid'][:,None] - sources['ycentroid'][None,:]))
        indx = (dist < default_fwhm/1.7) & (j > i)
        sources = sources[numpy.logical_not(numpy.any(indx, axis=0))]

#        show(f, img, median, sources['xcentroid'], sources['ycentroid'], default_fwhm)

        # Make sure it found a sufficient number of sources
        if len(sources) < 30:
            continue
        #----------------------------------------------------------------------
  

  
        #----------------------------------------------------------------------
        # Attempt to improve the source detection by improving the FWHM
        # estimate
        win = int(numpy.ceil(2*default_fwhm))
        if win % 2 == 0:
            win += 1
        bkgrms = MADStdBackgroundRMS()
        std = bkgrms(img)
        print(f'image std: {std}')
        # source finder
        iraffind = IRAFStarFinder(fwhm=default_fwhm, threshold=5*std, minsep_fwhm=0.1)
        # Grouping algorithm
        daogroup = DAOGroup(crit_separation=2*default_fwhm)
        # This determine how to measure the background of the image
        mmm_bkg = MMMBackground()
        # This is the optimization algorithm
        fitter = LevMarLSQFitter()
        # This is the model of the PSF
        gaussian_prf = IntegratedGaussianPRF(sigma=default_fwhm/sig2fwhm)
        # Here is where I turn-off fixing the sigma to be a constant
        # value
        gaussian_prf.sigma.fixed = False
        # This is the object that performs the photometry
        phot = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                  group_maker=daogroup,
                                                  bkg_estimator=mmm_bkg, psf_model=gaussian_prf,
                                                  fitter=fitter, fitshape=win, niters=4)
        print('first fit')
        # This is actually when the fitting is done
        phot_data = phot(image=img,
                         init_guesses=table.Table(sources['xcentroid', 'ycentroid', 'flux'],
                                                  names=('x_0', 'y_0', 'flux_0')))
        indx = phot_data['iter_detected'] == 1
        # Here I'm taking the median FWHM of the all detected/fitted
        # stars
        fwhm = numpy.median(phot_data['sigma_fit'][indx])*sig2fwhm
        print(f'fwhm: {fwhm}')

        show(f, img, phot.bkg_estimator(img), phot_data['x_fit'][indx], phot_data['y_fit'][indx],
             fwhm)
        #----------------------------------------------------------------------





        #----------------------------------------------------------------------
        # Refit using the "improved" FWHM

        iraffind = IRAFStarFinder(fwhm=fwhm, threshold=5*std, minsep_fwhm=0.1)
        daogroup = DAOGroup(crit_separation=2*fwhm)
        gaussian_prf = IntegratedGaussianPRF(sigma=fwhm/sig2fwhm)
        phot = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                  group_maker=daogroup,
                                                  bkg_estimator=mmm_bkg, psf_model=gaussian_prf,
                                                  fitter=fitter, fitshape=win, niters=2)
        print('second fit')
        phot_data = phot(image=img,
                         init_guesses=table.Table([phot_data['x_fit'][indx],
                                                   phot_data['y_fit'][indx],
                                                   phot_data['flux_fit'][indx]],
                                                  names=('x_0', 'y_0', 'flux_0')))
        #----------------------------------------------------------------------


        #----------------------------------------------------------------------
        # This is all just saving the results

        srt = numpy.argsort(phot_data['flux_fit'])[::-1]
        indx = srt[:36]
        phot_data = phot_data[indx]
        ofile = f.replace('.fits', '_find.png')
#        ofile = None
        show(f, img, phot.bkg_estimator(img), phot_data['x_fit'], phot_data['y_fit'],
             fwhm, ofile=ofile)

        phot_data['file'] = f
        phot_data['file_id'] = k
        phot_data['log_el'] = el
        phot_data['fwhm'] = fwhm

        save_cols = ['id', 'x_fit', 'y_fit', 'flux_fit', 'file', 'file_id', 'log_el', 'fwhm']
        if src is None:
            src = phot_data[save_cols]
        else:
            src = table.vstack([src, phot_data[save_cols]])

    src.write(spot_file, format='ascii.fixed_width', overwrite=True)

def match_spots(spot_file, matched_file):
    src = table.Table.read(spot_file, format='ascii.fixed_width')

    fields, count = numpy.unique(src['file_id'].data, return_counts=True)

    f_ref = 0

    src['x_off'] = 0.
    src['y_off'] = 0.
    src['mjd'] = 0.
    src['az'] = 0.
    src['el'] = 0.

    ref_indx = src['file_id'].data == f_ref
    f_id = src['id'][ref_indx]
    x0 = src['x_fit'][ref_indx]
    y0 = src['y_fit'][ref_indx]
    with fits.open(src['file'][ref_indx].data[0]) as hdu:
        src['mjd'][ref_indx] = float(hdu[0].header['MJD-OBS'])
        src['el'][ref_indx] = float(hdu[0].header['EL'])
        src['az'][ref_indx] = float(hdu[0].header['AZ'])

    for f in fields:
        if f == f_ref:
            continue

        indx = src['file_id'].data == f
        x1 = src['x_fit'][indx]
        y1 = src['y_fit'][indx]

        offset = [ numpy.mean(x1) - numpy.mean(x0), numpy.mean(y1) - numpy.mean(y0) ]
        src['x_off'][indx] = offset[0]
        src['y_off'][indx] = offset[1]

        x1 -= offset[0]
        y1 -= offset[1]

        dist = (x0[:,None] - x1[None,:])**2 + (y0[:,None] - y1[None,:])**2
        r, c = optimize.linear_sum_assignment(dist)

        src[indx] = src[indx][c]
        src['id'][indx] = f_id
        with fits.open(src['file'][indx].data[0]) as hdu:
            src['mjd'][indx] = float(hdu[0].header['MJD-OBS'])
            src['el'][indx] = float(hdu[0].header['EL'])
            src['az'][indx] = float(hdu[0].header['AZ'])

    src.write(matched_file, format='ascii.fixed_width', overwrite=True)


def init_ax(fig, pos):
    ax = fig.add_axes(pos)
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=3, direction='in', top=True, right=True)
    ax.grid(True, which='major', color='0.85', zorder=0, linestyle='-')
    return ax


def spot_track_plot(spot_id, src, ofile=None):

    mean_mjd = numpy.mean(src['mjd'].data)

    indx = numpy.where(src['id'] == spot_id)[0]
    x_ref = src['x_fit'][indx[0]] - src['x_off'][indx[0]]
    y_ref = src['y_fit'][indx[0]] - src['y_off'][indx[0]]

    dist = numpy.sqrt((src['x_fit'].data[indx] - src['x_off'].data[indx] - x_ref)**2
                        + (src['y_fit'].data[indx] - src['y_off'].data[indx] -y_ref)**2)


    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    mjd_lim = [-0.015, 0.015]
    az_lim = [-95, 50]
    el_lim = [25, 90]
    dist_lim = [-0.1, 6.5]
    fwhm_lim = [2.5, 7.5]

    # Azimuth
    ax = init_ax(fig, [0.12, 0.75, 0.83, 0.21])
    ax.set_xlim(mjd_lim)
    ax.set_ylim(az_lim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.scatter(src['mjd'].data[indx] - mean_mjd, src['az'].data[indx],
               marker='.', color='k', lw=0, s=50, zorder=4, alpha=0.5)
    ax.text(-0.08, 0.5, 'Tel. Azimuth', ha='center', va='center', rotation='vertical',
            transform=ax.transAxes)

    ax.text(0.5, 1.05, f'Spot ID: {spot_id}', ha='center', va='center',
            transform=ax.transAxes)

    # Elevation
    ax = init_ax(fig, [0.12, 0.54, 0.83, 0.21])
    ax.set_xlim(mjd_lim)
    ax.set_ylim(el_lim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.scatter(src['mjd'].data[indx] - mean_mjd, src['el'].data[indx],
               marker='.', color='k', lw=0, s=50, zorder=4, alpha=0.5)
    ax.text(-0.08, 0.5, 'Tel. Elevation', ha='center', va='center', rotation='vertical',
            transform=ax.transAxes)

    # Offset
    ax = init_ax(fig, [0.12, 0.33, 0.83, 0.21])
    ax.set_xlim(mjd_lim)
    ax.set_ylim(dist_lim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.scatter(src['mjd'].data[indx] - mean_mjd, dist,
               marker='.', color='k', lw=0, s=50, zorder=4, alpha=0.5)
    ax.text(-0.08, 0.5, 'Offset Dist (pix)', ha='center', va='center', rotation='vertical',
            transform=ax.transAxes)

    # FWHM
    ax = init_ax(fig, [0.12, 0.12, 0.83, 0.21])
    ax.set_xlim(mjd_lim)
    ax.set_ylim(fwhm_lim)
    ax.scatter(src['mjd'].data[indx] - mean_mjd, src['fwhm'].data[indx],
               marker='.', color='k', lw=0, s=50, zorder=4, alpha=0.5)
    ax.text(-0.08, 0.5, 'FWHM (pix)', ha='center', va='center', rotation='vertical',
            transform=ax.transAxes)
    ax.text(0.5, -0.20, r'$\Delta$ MJD', ha='center', va='center',
            transform=ax.transAxes)

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)


def main():

    spot_file = 'ssc_spot_centers.txt'
    if not os.path.isfile(spot_file):
        find_spots(spot_file)
    matched_file = spot_file.replace('.txt', '_matched.txt')
    if not os.path.isfile(matched_file):
        match_spots(spot_file, matched_file)
     
    src = table.Table.read(matched_file, format='ascii.fixed_width')

    ids = numpy.unique(src['id'].data)

    for spot_id in ids:
        spot_track_plot(spot_id, src, ofile=f'spot_id_{spot_id:02}.png')


if __name__ == '__main__':
    main()

    

