
from pathlib import Path
from itertools import chain, combinations
import warnings

from IPython import embed

import numpy
from matplotlib import pyplot, ticker, patches
from matplotlib.backends.backend_pdf import PdfPages

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import table

from sklearn.neighbors import KDTree
from skimage import transform

#from photutils import IRAFStarFinder
from photutils.detection import DAOStarFinder
from photutils import CircularAperture
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import IntegratedGaussianPRF, DAOGroup

from pathlib import Path

# from pypeit.spectrographs.keck_deimos import deimos_read_1chip
# from pypeit.core.procimg import subtract_overscan

from pypeit_funcs import deimos_read_1chip, subtract_overscan

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from convenience_funcs.all_funcs import Fits_Simple



def sub_overscan(data, oscan):
    raw_img = numpy.append(data, oscan, axis=1)
    datasec_img = numpy.append(numpy.ones(data.shape, dtype=int),
                               numpy.zeros(oscan.shape, dtype=int), axis=1)
    oscansec_img = numpy.append(numpy.zeros(data.shape, dtype=int),
                                numpy.ones(oscan.shape, dtype=int), axis=1)
    rdx_img, _ = subtract_overscan(raw_img, datasec_img, oscansec_img)
    return rdx_img[:data.shape[0],:data.shape[1]]


#def make_dark():
#    
#    root = Path('../2021aug24').resolve()
#    darks = [root / f'd0824_00{n}.fits' for n in range(20,27)]
#    nchips = 4
#    img_shape = (2601, 2048)
#    dark_stack = numpy.zeros((len(darks),)+img_shape, dtype=float)
#    master_dark = numpy.zeros((nchips,)+img_shape, dtype=float)
#
#    for i in range(nchips):
#        for j,d in enumerate(darks):
#            print(d)
#            hdu = fits.open(str(d))
#            data, oscan = map(lambda x : x.astype(float), deimos_read_1chip(hdu, i+1))
#            dark_stack[j] = sub_overscan(data, oscan)
#        _cmb_dark = sigma_clipped_stats(dark_stack, sigma=3., axis=0)[0]
#        if not numpy.all(numpy.isfinite(_cmb_dark)):
#            raise ValueError('Need to handle masked pixels.')
#        master_dark[i] = _cmb_dark
#    return master_dark


def make_flat(): #dark):
    
    root = Path('../20151217').resolve()
    flats = [root / f'd1217_00{n}.fits' for n in range(18,21)]
    nchips = 8
    img_shape = (2601, 1024)
    flat_stack = numpy.zeros((len(flats),)+img_shape, dtype=float)
    master_flat = numpy.zeros((nchips,)+img_shape, dtype=float)

    for i in range(nchips):
        for j,d in enumerate(flats):
            print(d)
            hdu = fits.open(str(d))
            data, oscan = map(lambda x : x.astype(float), deimos_read_1chip(hdu, i+1))
            flat_stack[j] = sub_overscan(data, oscan) # - dark[i]
        _cmb_flat = sigma_clipped_stats(flat_stack, sigma=5., axis=0)[0]
        if not numpy.all(numpy.isfinite(_cmb_flat)):
            raise ValueError('Need to handle masked pixels.')
        master_flat[i] = _cmb_flat / numpy.mean(_cmb_flat)
    return master_flat


#def rdx_science(dark, flat, data_file, sat=65535.):
def rdx_science(flat, data_file, sat=65535.):

    nchips = 8
    img_shape = (2601, 1024)

    sci_img = numpy.ma.zeros((nchips,)+img_shape, dtype=float)
    hdu = fits.open(str(data_file))
    for i in range(nchips):
        data, oscan = map(lambda x : x.astype(float), deimos_read_1chip(hdu, i+1))
        # sci_img[i] = numpy.ma.MaskedArray((sub_overscan(data, oscan) - dark[i])/flat[i],
        #                                   mask=(flat[i] < 0.5) | (data > sat))
        sci_img[i] = numpy.ma.MaskedArray(sub_overscan(data, oscan)/flat[i],
                                          mask=(flat[i] < 0.5) | (data > sat))
    return sci_img


def show_deimos(img, vmin, vmax):
    nchips = 4
    img_shape = (2601, 2048)
    buff = 10

    imaspect = img_shape[0]/img_shape[1]

    w,h = pyplot.figaspect(0.5)
    fig = pyplot.figure(figsize=(w,h))

    for i in range(nchips):
        ax = fig.add_axes([0.04 + i*0.23*(1+buff/img_shape[0]), 0.5-0.23*imaspect,
                           0.23, 2*0.23*imaspect])
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.imshow(img[i], origin='lower', interpolation='nearest', aspect='auto',
                  vmin=vmin, vmax=vmax)

    pyplot.show()


def show_stamps(stamps, pdf_file, vmin=None, vmax=None):
    
    dim = 8
    start = 0.01
    buff = 0.01
    dw = (1-2*start-(dim-1)*buff)/dim
    nstamps = stamps.shape[0]
    npages = int(nstamps/dim**2)+1
  
    with PdfPages(pdf_file) as pdf:
        i = 0
        while i < nstamps:
            w,h = pyplot.figaspect(1)
            fig = pyplot.figure(figsize=(1.5*w,1.5*h))
            for j in range(dim):
                for k in range(dim):
                    if i < nstamps:
                        ax = fig.add_axes([start+k*(dw+buff), start+(dim-j-1)*dw+(dim-j-1)*buff,
                                           dw, dw])
                        ax.xaxis.set_major_formatter(ticker.NullFormatter())
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_major_formatter(ticker.NullFormatter())
                        ax.yaxis.set_visible(False)
                        ax.imshow(stamps[i], origin='lower', interpolation='nearest', vmin=vmin,
                                  vmax=vmax)
                        ax.text(0.1, 0.9, str(i), ha='center', va='center',
                                transform=ax.transAxes, color='C1')
                        ax.scatter([stamps.shape[1]//2], [stamps.shape[2]//2], marker='+',
                                   s=50, color='C3', zorder=10, lw=0.5)
                        i += 1
            pdf.savefig()
            fig.clear()
            pyplot.close()


def rdx_science_sources(reduced_img, force=False):

#     # Set up directories & file paths
#     ifile = Path('../20151217').resolve() / sci_file
#     if not ifile.is_file():
#         raise FileNotFoundError(f'No file: {str(ifile)}')

#     print(f'Reducing: {str(ifile)}')

#     base = Path('.').resolve() / ifile.stem / ifile.name
#     ofits = base.with_suffix('.rdx.fits')
#     src_ofile = base.with_suffix('.src.db')

#     if not base.parent.is_dir():
#         base.parent.mkdir()

#     # Check if file already reduced
#     if ofits.is_file() and src_ofile.is_file() and not force:
#         warnings.warn(f'{sci_file} already reduced.  Use force to overwrite.')
#         return

#     # Dark current subtraction
# #    master_dark_file = Path('.').resolve() / 'MasterDark.fits'
# #    if master_dark_file.is_file():
# #        dark = fits.open(str(master_dark_file))[1].data
# #    else:
# #        dark = make_dark()
# #        fits.HDUList([fits.PrimaryHDU(),
# #                      fits.ImageHDU(data=dark, name='DARK')]
# #                    ).writeto(str(master_dark_file), overwrite=True)

#     # Flat division
#     master_flat_file = Path('.').resolve() / 'MasterFlat.fits'
#     if master_flat_file.is_file():
#         flat = fits.open(str(master_flat_file))[1].data
#     else:
#         flat = make_flat() #dark)
#         fits.HDUList([fits.PrimaryHDU(),
#                       fits.ImageHDU(data=flat, name='FLAT')]
#                     ).writeto(str(master_flat_file), overwrite=True)

# #    img = rdx_science(dark, flat, ifile)
#     # Get data & mask (one pair for each chip)
#     img = rdx_science(flat, ifile)

    reduced_img_obj = Fits_Simple(reduced_img)
    img = [reduced_img_obj.masked_array]
    
    ifile = reduced_img_obj.path
    proc_dir = Path('.').resolve() / "proc_files"
    Path.mkdir(proc_dir, exist_ok=True)
    base_parent = proc_dir / ifile.stem
    Path.mkdir(base_parent, exist_ok=True)
    base = proc_dir / ifile.stem / ifile.name
    ofits = base.with_suffix('.rdx.fits')
    src_ofile = base.with_suffix('.src.db')

    # No idea--creating stamps??
    default_fwhm = 7.0

    stamp_width = int(numpy.ceil(8*default_fwhm))
    if stamp_width % 2 == 0:
        stamp_width += 1
    stamp_shape = (stamp_width, stamp_width)
    x, y = numpy.meshgrid(numpy.arange(stamp_width, dtype=float)-stamp_width//2,
                          numpy.arange(stamp_width, dtype=float)-stamp_width//2)
    r = numpy.sqrt(x**2 + y**2)

    nchips = 1
    mean_rkron = numpy.zeros(nchips, dtype=float)
    source_data = numpy.zeros((0, 9), dtype=float)
    all_stamps = numpy.empty(nchips, dtype=object)

    # Detect sources
    for chip in range(nchips):
        print(f'Working on chip {chip+1}')

        mean, median, std = sigma_clipped_stats(img[chip], sigma=3.)
        starfind = DAOStarFinder(fwhm=default_fwhm, threshold=5.*std)
        sources = starfind(img[chip].filled(0.0)-median, mask=numpy.ma.getmaskarray(img[chip]))
        nsources = len(sources)
        ap_centers = numpy.column_stack((sources['xcentroid'], sources['ycentroid']))

        # Scikit-Image uses the bottom left corner to define the
        # coordinate system.  Photutils uses the center of the pixel.
        # ...  I think.  In any case, there seems to be an 0.5 pixel
        # offset, which is why the transformation below has that
        # additional offset
        center = numpy.atleast_1d(stamp_shape)/2
        stamps = numpy.ma.zeros((nsources,)+stamp_shape, dtype=float)
        rkron = numpy.zeros(nsources, dtype=float)
        kflux_raw = numpy.zeros(nsources, dtype=float)
        bkg = numpy.zeros(nsources, dtype=float)

        # Background subtract and other things??
        for i in range(nsources):
            t = transform.EuclideanTransform(translation=-ap_centers[i][::-1]-0.5 + center)
            stamps[i] = transform.warp(img[chip].filled(0.0).T, t.inverse, output_shape=stamp_shape,
                                       cval=0, order=0).T
            stamps[i,stamps[i] == 0.] = numpy.ma.masked
            if numpy.all(numpy.ma.getmaskarray(stamps[i, r > 3*default_fwhm])):
                continue
            bkg[i] = numpy.ma.median(stamps[i, r > 3*default_fwhm])
            stamps[i] -= bkg[i]
            if numpy.all(numpy.ma.getmaskarray(stamps[i])):
                continue
            _rkron = numpy.ma.sum(r*stamps[i])/numpy.ma.sum(stamps[i])
            if _rkron == numpy.ma.masked or _rkron < 0:
                continue
            rkron[i] = _rkron
            indx = r < rkron[i]
            if not numpy.any(indx):
                continue
            kflux_raw[i] = numpy.ma.sum(stamps[i,indx])

        mean_rkron[chip] = sigma_clipped_stats(rkron, sigma=3., maxiters=None)[0]
        kflux = numpy.array([numpy.sum(s[r < mean_rkron[chip]]) for s in stamps])

        # Removing sources that are too close
        i, j = numpy.triu_indices(nsources, k=1)
        dist = numpy.sqrt((sources['xcentroid'][j] - sources['xcentroid'][i])**2 
                          + (sources['ycentroid'][j] - sources['ycentroid'][i])**2)
        too_close = numpy.identity(nsources).astype(bool)
        min_dist_fwhm = 5
        too_close[i,j] = dist < min_dist_fwhm*default_fwhm
        too_close[j,i] = dist < min_dist_fwhm*default_fwhm
    
        _kflux = numpy.ma.MaskedArray(numpy.tile(kflux, (nsources,1)),
                                      mask=numpy.logical_not(too_close))
        keep = (numpy.ma.argmax(_kflux, axis=1) == numpy.arange(nsources)) & (kflux > 1e3)
    
        keep &= (numpy.sum(stamps.mask, axis=(1,2)) == 0)
        nkeep = numpy.sum(keep)
        if nkeep == 0:
            all_stamps[chip] = None
            continue

        # ???
        srt = numpy.argsort(kflux[keep])[::-1]
        all_stamps[chip] = stamps[keep][srt]
        source_data = numpy.append(source_data,
                                   numpy.column_stack((numpy.full(nkeep, chip+1, dtype=float),
                                                       numpy.arange(nkeep, dtype=float),
                                                       sources['xcentroid'][keep][srt],
                                                       sources['ycentroid'][keep][srt],
                                                       bkg[keep][srt],
                                                       rkron[keep][srt],
                                                       kflux_raw[keep][srt],
                                                       kflux[keep][srt],
                                                       numpy.ones(nkeep, dtype=float))),
                                   axis=0)
        
        # Plot the stamps
        pdf_file = base.with_suffix(f'.{chip+1}.stamps.pdf')
        show_stamps(all_stamps[chip], pdf_file)

    # Print the source data
    header_base = 'Mean per-chip Kron Radii:\n'
    for chip in range(nchips):
        header_base += f'    Chip {chip+1}: {mean_rkron[chip]:.2f}\n'
    numpy.savetxt(str(src_ofile), source_data,
                  fmt='%6.0f %4.0f %7.2f %7.2f %8.2f %7.2f %11.4e %11.4e %2.0f',
                  header=header_base + f"{'CHIP':>4} {'ID':>4} {'X':>7} {'Y':>7} {'BKG':>8} "
                                       f"{'R':>7} {'C_RAW':>11} {'C':>11} {'F':>2}")

    # Print the image data
    hdr = fits.Header()
    stamp_hdus = []
    for chip in range(nchips):
        hdr[f'MRK_{chip+1:>02}'] = mean_rkron[chip]
        if all_stamps[chip] is None:
            continue
        stamp_hdus += [fits.ImageHDU(data=all_stamps[chip].data, name=f'STAMPS_{chip+1:>02}')]
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(data=img[0].data, name='IMAGES'),
                  fits.ImageHDU(data=numpy.ma.getmaskarray(img[0]).astype(numpy.uint8),
                                name='MASKS')]
                  + stamp_hdus).writeto(str(ofits), overwrite=True)



def main():
    # sci_files = [f'd1217_01{n:02}.fits' for n in range(4,17,4)]
    # for sci_file in sci_files:
    #     rdx_science_sources(sci_file)
    return


if __name__ == '__main__':
    main()

