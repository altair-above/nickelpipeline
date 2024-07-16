
from pathlib import Path
#from itertools import chain, combinations
import warnings

from IPython import embed

import numpy
from scipy import stats
from scipy import optimize
from matplotlib import pyplot, ticker, patches
#from matplotlib.backends.backend_pdf import PdfPages

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Moffat2D, Moffat1D
from astropy import table
from astropy.visualization import AsinhStretch, ZScaleInterval, ImageNormalize

from skimage import io

from photutils.detection import IRAFStarFinder, DAOStarFinder, find_peaks
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import IntegratedGaussianPRF, DAOGroup


class FitEllipticalMoffat2D:
    """
    Parameters are
        0: x0,
        1: y0,
        2: amplitude,
        3: gamma1 (width 1),
        4: gamma2 (width 2),
        5: phi (rotation),
        6: alpha (fall-off slope),
        7: background
    """
    def __init__(self, c):
        self.c = c
        self.shape = self.c.shape
        self.x, self.y = numpy.meshgrid(numpy.arange(self.shape[0]).astype(float),
                                        numpy.arange(self.shape[1]).astype(float))
        self.par = self.guess_par()
        self.npar = self.par.size
        self.nfree = self.npar
        self.free = numpy.ones(self.npar, dtype=bool)

    def guess_par(self):
        """
        Parameters are x0, y0, amplitude, gamma1, gamma2 (widths), phi (rotation),
                    alpha (fall-off slope), background.
        """
        par = numpy.zeros(8, dtype=float)
        par[0] = self.x[self.shape[0]//2, self.shape[1]//2]
        par[1] = self.y[self.shape[0]//2, self.shape[1]//2]
        par[2] = self.c[self.shape[0]//2, self.shape[1]//2]
        r = numpy.sqrt((self.x-par[0])**2 + (self.y-par[1])**2)
        par[3] = numpy.sum(r*self.c)/numpy.sum(self.c)/2
        par[4] = numpy.sum(r*self.c)/numpy.sum(self.c)/2
        par[5] = 0.
        par[6] = 3.5
        par[7] = 0.
        return par

    def default_bounds(self):
        lb = numpy.zeros(8, dtype=float)
        ub = numpy.zeros(8, dtype=float)
        lb[0], ub[0] = 0., float(self.shape[0])
        lb[1], ub[1] = 0., float(self.shape[1])
        lb[2], ub[2] = 0., 1.2*numpy.amax(self.c)
        lb[3], ub[3] = 1., max(self.shape)/2.
        lb[4], ub[4] = 1., max(self.shape)/2.
        lb[5], ub[5] = 0., numpy.pi
        lb[6], ub[6] = 1., 10.
        lb[7] = numpy.amin(self.c)-0.1*numpy.absolute(numpy.amin(self.c))
        ub[7] = 0.1*numpy.amax(self.c)
        return lb, ub

    def _set_par(self, par):
        if par.ndim != 1:
            raise ValueError('Parameter array must be a 1D vector.')
        if par.size == self.npar:
            self.par = par.copy()
            return
        if par.size != self.nfree:
            raise ValueError(f'Must provide {self.npar} or {self.nfree} parameters.')
        self.par[self.free] = par.copy()


    @staticmethod
    def _eval_moffat(par, x, y):
        cosr = numpy.cos(par[5])
        sinr = numpy.sin(par[5])
        A = (cosr/par[3])**2 + (sinr/par[4])**2
        B = (sinr/par[3])**2 + (cosr/par[4])**2
        C = 2 * sinr * cosr * (par[3]**-2 - par[4]**-2)
        dx = (x - par[0])
        dy = (y - par[1])
        D = 1 + A*dx**2 + B*dy**2 + C*dx*dy
        return par[2] / D**par[6]

    @staticmethod
    def _eval_moffat_deriv(par, x, y):
        cosr = numpy.cos(par[5])
        sinr = numpy.sin(par[5])

        a1 = (cosr/par[3])**2
        a2 = (sinr/par[4])**2
        A = a1 + a2
        dA_dg1 = -2*a1/par[3]
        dA_dg2 = -2*a2/par[4]

        b1 = (sinr/par[3])**2
        b2 = (cosr/par[4])**2
        B = b1 + b2
        dB_dg1 = -2*b1/par[3]
        dB_dg2 = -2*b2/par[4]

        C = 2 * sinr * cosr * (par[3]**-2 - par[4]**-2)
        dC_dg1 = -4 * sinr * cosr / par[3]**3
        dC_dg2 = 4 * sinr * cosr / par[4]**3

        dA_dphi = -C
        dB_dphi = C
        dC_dphi = C * (cosr / sinr - sinr / cosr) # = 2 * C / tan(2 * phi)

        dx = (x.ravel() - par[0])
        dy = (y.ravel() - par[1])

        D = 1 + A*dx**2 + B*dy**2 + C*dx*dy
        dD_dx0 = -2*A*dx - C
        dD_dy0 = -2*B*dy - C
        dD_dg1 = dA_dg1*dx**2 + dB_dg1*dy**2 + dC_dg1*dx*dy
        dD_dg2 = dA_dg2*dx**2 + dB_dg2*dy**2 + dC_dg2*dx*dy
        dD_dphi = dA_dphi*dx**2 + dB_dphi*dy**2 + dC_dphi*dx*dy

        I = par[2] / D**par[6]
        f = -par[6] / D

        return [f * I * dD_dx0,
                f * I * dD_dy0,
                I/par[2],
                f * I * dD_dg1,
                f * I * dD_dg2,
                f * I * dD_dphi,
                -I * numpy.log(D)]
        
    def model(self, par=None, x=None, y=None):
        if par is not None:
            self._set_par(par)
        return self._eval_moffat(self.par, self.x if x is None else x, self.y if y is None else y) \
                    + self.par[7]

    def resid(self, par):
        return self.c.ravel() - self.model(par).ravel()

    def deriv_resid(self, par):
        self._set_par(par)
        dmoff = self._eval_moffat_derive(self.par, self.x, self.y)
        return numpy.array([-d for d in dmoff]
                            + [numpy.full(numpy.prod(self.shape), -1., dtype=float)]).T

    def fit(self, p0=None, fix=None, lb=None, ub=None):
        if p0 is None:
            p0 = self.guess_par()
        _p0 = numpy.atleast_1d(p0)
        if _p0.size != self.npar:
            raise ValueError('Incorrect number of model parameters.')
        self.par = _p0.copy()
        _free = numpy.ones(self.npar, dtype=bool) if fix is None else numpy.logical_not(fix)
        if _free.size != self.npar:
            raise ValueError('Incorrect number of model parameter fitting flags.')
        self.free = _free.copy()
        self.nfree = numpy.sum(self.free)
        
        _lb, _ub = self.default_bounds()
        if lb is None:
            lb = _lb
        if ub is None:
            ub = _ub
        if len(lb) != self.npar or len(ub) != self.npar:
            raise ValueError('Length of one or both of the bounds vectors is incorrect.')

        p = self.par[self.free]
        result = optimize.least_squares(self.resid, p, method='trf', xtol=1e-12,
                                        bounds=(lb[self.free], ub[self.free]),
                                        verbose=2) #, jac=self.deriv_resid)
        self._set_par(result.x)

    @staticmethod
    def to_gamma(fwhm, alpha):
        return fwhm / 2 / numpy.sqrt(2**(1/alpha)-1)
    
    @staticmethod
    def to_fwhm(gamma, alpha):
        return 2 * gamma * numpy.sqrt(2**(1/alpha)-1)


def test_elliptical():

    x0 = 6.
    y0 = 6.
    gam_x = 4.
    gam_y = 8.
    alpha = 4.
    phi = numpy.pi/2.
    par = numpy.array([x0, y0, 1., gam_x, gam_y, phi, alpha, 0.])
    c = numpy.ones((13,13), dtype=float)
    fit = FitEllipticalMoffat2D(c)
#    model = fit.model(par=par)

    r = numpy.linspace(0., 20., 100)
    x = r * numpy.cos(phi+numpy.pi/2) + x0
    y = r * numpy.sin(phi+numpy.pi/2) + y0
    model = fit.model(par=par, x=x, y=y)

    oned = Moffat1D()
    oned_model = oned.evaluate(r, 1., 0., gam_y, alpha)

    embed()
    exit()

        
        

class FitMoffat2D:
    """
    Parameters are x0, y0, amplitude, gamma (width), alpha (fall-off slope), background.
    """
    def __init__(self, c):
        self.c = c
        self.shape = self.c.shape
        self.x, self.y = numpy.meshgrid(numpy.arange(self.shape[0]).astype(float),
                                        numpy.arange(self.shape[1]).astype(float))
        self.moff = Moffat2D()
        self.par = self.guess_par()
        self.npar = self.par.size
        self.nfree = self.npar
        self.free = numpy.ones(self.npar, dtype=bool)

    def guess_par(self):
        """
        Parameters are x0, y0, amplitude, gamma (width), alpha (fall-off
        slope), background.
        """
        par = numpy.zeros(6, dtype=float)
        par[0] = self.x[self.shape[0]//2, self.shape[1]//2]
        par[1] = self.y[self.shape[0]//2, self.shape[1]//2]
        par[2] = self.c[self.shape[0]//2, self.shape[1]//2]
        r = numpy.sqrt((self.x-par[0])**2 + (self.y-par[1])**2)
        par[3] = numpy.sum(r*self.c)/numpy.sum(self.c)/2
        par[4] = 3.5
        par[5] = 0.
        return par

    def default_bounds(self):
        lb = numpy.zeros(6, dtype=float)
        ub = numpy.zeros(6, dtype=float)
        lb[0], ub[0] = 0., float(self.shape[0])
        lb[1], ub[1] = 0., float(self.shape[1])
        lb[2], ub[2] = 0., 1.2*numpy.amax(self.c)
        lb[3], ub[3] = 1., max(self.shape)/2.
        lb[4], ub[4] = 1., 10.
        lb[5] = numpy.amin(self.c)-0.1*numpy.absolute(numpy.amin(self.c))
        ub[5] = 0.1*numpy.amax(self.c)
        return lb, ub

    def _set_par(self, par):
        if par.ndim != 1:
            raise ValueError('Parameter array must be a 1D vector.')
        if par.size == self.npar:
            self.par = par.copy()
            return
        if par.size != self.nfree:
            raise ValueError(f'Must provide {self.npar} or {self.nfree} parameters.')
        self.par[self.free] = par.copy()

    def model(self, par=None, x=None, y=None):
        if par is not None:
            self._set_par(par)
        return self.moff.evaluate(self.x if x is None else x,
                                  self.y if y is None else y,
                                  self.par[2], self.par[0], self.par[1],
                                  self.par[3], self.par[4]) + self.par[5]

    def resid(self, par):
        return self.c.ravel() - self.model(par).ravel()

    def deriv_resid(self, par):
        self._set_par(par)
        dmoff = self.moff.fit_deriv(self.x.ravel(), self.y.ravel(), self.par[2], self.par[0],
                                    self.par[1], self.par[3], self.par[4])
        return numpy.array([-d for d in dmoff]
                            + [numpy.full(numpy.prod(self.shape), -1., dtype=float)]).T

    def fit(self, p0=None, fix=None, lb=None, ub=None):
        if p0 is None:
            p0 = self.guess_par()
        _p0 = numpy.atleast_1d(p0)
        if _p0.size != self.npar:
            raise ValueError('Incorrect number of model parameters.')
        self.par = _p0.copy()
        _free = numpy.ones(self.npar, dtype=bool) if fix is None else numpy.logical_not(fix)
        if _free.size != self.npar:
            raise ValueError('Incorrect number of model parameter fitting flags.')
        self.free = _free.copy()
        self.nfree = numpy.sum(self.free)
        
        _lb, _ub = self.default_bounds()
        if lb is None:
            lb = _lb
        if ub is None:
            ub = _ub
        if len(lb) != self.npar or len(ub) != self.npar:
            raise ValueError('Length of one or both of the bounds vectors is incorrect.')

        p = self.par[self.free]
        result = optimize.least_squares(self.resid, p, method='trf', xtol=1e-12,
                                        bounds=(lb[self.free], ub[self.free]),
                                        verbose=2) #, jac=self.deriv_resid)
        self._set_par(result.x)

    @staticmethod
    def to_gamma(fwhm, alpha):
        return fwhm / 2 / numpy.sqrt(2**(1/alpha)-1)
    
    @staticmethod
    def to_fwhm(gamma, alpha):
        return 2 * gamma * numpy.sqrt(2**(1/alpha)-1)


def make_ellipse(a, b, phi, n=1000):
    cosr = numpy.cos(phi)
    sinr = numpy.sin(phi)
    theta = numpy.linspace(0., 2*numpy.pi, num=n) #, endpoint=False)
    x = a * numpy.cos(theta)
    y = b * numpy.sin(theta)
    return x*cosr - y*sinr, y*cosr + x*sinr


def get_bg(img, clip_iter=None, sigma_lower=100., sigma_upper=5.):
    """
    Measure the background in an image.

    Args:
        img (array):
            2D array with image data.
        clip_iter (:obj:`int`, optional):
            Number of clipping iterations.  If None, no clipping is
            performed.
        sigma_lower (:obj:`float`, optional):
            Sigma level for clipping.  Clipping only removes negative outliers.
            Ignored if clip_iter is None.
        sigma_upper (:obj:`float`, optional):
            Sigma level for clipping.  Clipping only removes positive
            outliers.  Ignored if clip_iter is None.

    Returns:
        :obj:`tuple`: Returns the background level, the standard
        deviation in the background, and the number of rejected values
        excluded from the computation.
    """
    if clip_iter is None:
        # Assume the image has sufficient background pixels relative to
        # pixels with the fiber output image to find the background
        # using a simple median
        bkg = numpy.median(img)
        sig = stats.median_abs_deviation(img, axis=None, nan_policy='omit', scale='normal')
        return bkg, sig, 0

    # Clip the high values of the image to get the background and
    # background error
    clipped_img = sigma_clip(img, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                             stdfunc='mad_std', maxiters=clip_iter)
    bkg = numpy.ma.median(clipped_img)
    sig = stats.median_abs_deviation(clipped_img.compressed(), scale='normal')
    nrej = numpy.sum(numpy.ma.getmaskarray(clipped_img))

    return bkg, sig, nrej


def bench_image(ifile, ext=0):
    """
    Read an image from a bench imaging camera.

    Args:
        ifile (:obj:`str`, `Path`_):
            File name
        ext (:obj:`int`, optional):
            If the file is a multi-extension fits file, this selects the
            extension with the relevant data.

    Returns:
        `numpy.ndarray`_: Array with the (floating-point) image data.
    """
    if ifile is None:
        return None

    _ifile = Path(ifile).resolve()
    if not _ifile.exists():
        raise FileNotFoundError(f'{_ifile} does not exist!')

    if any([s in ['.fit', '.fits'] for s in _ifile.suffixes]):
        return fits.open(_ifile)[ext].data.astype(float)
    return io.imread(_ifile).astype(float)


def find_spots(img, nmax=100, border=0.):

    default_fwhm = 3.
    sig2fwhm = numpy.sqrt(8*numpy.log(2))

    # Do a first source detection using the default FWHM
    mean, median, std = sigma_clipped_stats(img, sigma=3.)
    return find_peaks(img-median, 5.*std, box_size=11, npeaks=nmax, border_width=border)


def plot_fit(fit, ofile=None):

    w,h = pyplot.figaspect(1.)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    stack = fit.c
    model = fit.model()

    amp = fit.par[2]
    if isinstance(fit, FitMoffat2D):
        beta = fit.par[4]
        fwhm1 = FitMoffat2D.to_fwhm(fit.par[3], beta)
        ell_x, ell_y = make_ellipse(fwhm1, fwhm1, 0.)
    else:
        beta = fit.par[6]
        fwhm1 = FitMoffat2D.to_fwhm(fit.par[3], beta)
        fwhm2 = FitMoffat2D.to_fwhm(fit.par[4], beta)
        ell_x, ell_y = make_ellipse(fwhm1, fwhm2, fit.par[5])
        if fwhm1 < fwhm2:
            fwhm1, fwhm2 = fwhm2, fwhm1
    ell_x += fit.par[0]
    ell_y += fit.par[1]
        
    norm = ImageNormalize(numpy.concatenate((stack, model, stack-model)),
                          interval=ZScaleInterval(contrast=0.10),
                          stretch=AsinhStretch())

    ax = fig.add_axes([0.03, 0.4, 0.2, 0.2])
    ax.imshow(stack, origin='lower', interpolation='nearest', norm=norm)
    ax.contour(stack, [amp/8, amp/4, amp/2, amp/1.1], colors='k', linewidths=0.5)
    ax.set_axis_off()
    ax.text(0.5, 1.01, 'Observed', ha='center', va='bottom', transform=ax.transAxes)

    ax = fig.add_axes([0.24, 0.4, 0.2, 0.2])
    ax.imshow(model, origin='lower', interpolation='nearest', norm=norm)
    ax.contour(model, [amp/8, amp/4, amp/2, amp/1.1], colors='k', linewidths=0.5)
    ax.plot(ell_x, ell_y, color='C3', lw=0.5)
    ax.set_axis_off()
    ax.text(0.5, 1.01, 'Model', ha='center', va='bottom', transform=ax.transAxes)

    ax = fig.add_axes([0.45, 0.4, 0.2, 0.2])
    ax.imshow(stack-model, origin='lower', interpolation='nearest', norm=norm)
    ax.contour(stack-model, [-amp/40, amp/40], colors=['w','k'], linewidths=0.5)
    ax.set_axis_off()
    ax.text(0.5, 1.01, 'Residual', ha='center', va='bottom', transform=ax.transAxes)

    r = numpy.sqrt((fit.x - fit.par[0])**2 + (fit.y - fit.par[1])**2).ravel()
#    rlim = numpy.array([0, numpy.amax(r)])
    rlim = numpy.array([0, 5*fwhm1])
 
    oned = Moffat1D()
    r_mod = numpy.linspace(*rlim, 100)
    if isinstance(fit, FitMoffat2D):
        models = [oned.evaluate(r_mod, amp, 0., fit.par[3], beta) + fit.par[5]]
    else:
        models = [oned.evaluate(r_mod, amp, 0., fit.par[3], beta) + fit.par[7],
                  oned.evaluate(r_mod, amp, 0., fit.par[4], beta) + fit.par[7]]
    
    ax = fig.add_axes([0.66, 0.4, 0.3, 0.2])
    ax.minorticks_on()
    ax.set_xlim(rlim)
    ax.tick_params(axis='x', which='both', direction='in')
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    for model in models:
        ax.plot(r_mod, model, color='C3')
    ax.scatter(r, stack.ravel(), marker='.', lw=0, s=30, alpha=0.5, color='k')
#    ax.text(0.95, 0.9, f'FWHM = {fwhm:.1f} pix; {fwhm*0.1185:.2f}"', ha='right',
#            va='center', transform=ax.transAxes)
    if isinstance(fit, FitMoffat2D):
        ax.text(0.95, 0.9, f'FWHM = {fwhm1:.1f} pix', ha='right',
                va='center', transform=ax.transAxes)
        ax.text(0.95, 0.78, f'beta = {beta:.2f}', ha='right', va='center',
                transform=ax.transAxes)
    else:
        ax.text(0.95, 0.9, f'FWHM_1 = {fwhm1:.1f} pix', ha='right',
                va='center', transform=ax.transAxes)
        ax.text(0.95, 0.78, f'FWHM_2 = {fwhm2:.1f} pix', ha='right',
                va='center', transform=ax.transAxes)
        ax.text(0.95, 0.66, f'beta = {beta:.2f}', ha='right', va='center',
                transform=ax.transAxes)
#    ax.text(0.5, 1.15, 'R [arcsec]', ha='center', va='bottom',
#            transform=ax.transAxes)
    ax.text(0.5, -0.15, 'R [pix]', ha='center', va='top', transform=ax.transAxes)

#    axt = ax.twiny()
#    axt.set_xlim(rlim*0.1185)
#    axt.minorticks_on()
#    axt.tick_params(axis='x', which='both', direction='in')
#    axt.tick_params(axis='y', which='both', left=False, right=False)

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

            




def main():

#    x, y = make_ellipse(1., 2., numpy.pi/4)
#    pyplot.plot(x, y)
#    pyplot.show()
#    exit()

#    test_elliptical()
#    exit()

    # ifiles = ['IFU-InFocus-Full_Reversed_16bit_focus3.bmp',
    #           'IFU-InFocus-Full_16bit_focus2.bmp',
    #           'IFU-InFocus-BottomLeft_16bit_focus2.bmp', 'IFU-InFocus-TopLeft_16bit_focus2.bmp',
    #           'IFU-InFocus-BottomRight_16bit_focus2.bmp', 'IFU-InFocus-TopRight_16bit_focus2.bmp']

    reduced_img = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V/d1046_red.fits')

    ifiles = []

    rdx_root = Path('./redux').resolve()
    if not rdx_root.exists():
        rdx_root.mkdir(parents=True)

    nmax = [37, 37, 37, 33, 37, 37]
    border = [50, 50, 50, 50, 50, 10]
    stamp = 15
    #fit_class = FitMoffat2D
    fit_class = FitEllipticalMoffat2D
    npar = 6 if fit_class == FitMoffat2D else 8

    for i in range(len(ifiles)):

        _ifile = Path(ifiles[i]).resolve()

        img = bench_image(_ifile)
        if img.ndim == 3:
           img = numpy.sum(img, axis=2)

        spot_coo = find_spots(img, nmax=nmax[i], border=border[i])
        nspots = len(spot_coo)

        best_fit_par = numpy.empty((nspots,npar), dtype=float)

        for j in range(nspots):
            sub_img = img[spot_coo['y_peak'][j]-stamp:spot_coo['y_peak'][j]+stamp,
                          spot_coo['x_peak'][j]-stamp:spot_coo['x_peak'][j]+stamp]

            fwhm = 3
            alpha = 3.5
            gamma = fwhm / 2 / numpy.sqrt(2**(1/alpha)-1)
#            fit = FitMoffat2D(sub_img)
#            p0 = numpy.array([float(stamp), float(stamp), numpy.amax(sub_img), gamma, alpha,
#                             numpy.median(sub_img)])
            fit = FitEllipticalMoffat2D(sub_img)
            p0 = numpy.array([float(stamp), float(stamp), numpy.amax(sub_img), gamma, gamma, 0.,
                              alpha, numpy.median(sub_img)])
            fit.fit(p0=p0)
            best_fit_par[j,...] = fit.par
            best_fit_par[j,0] += spot_coo['x_peak'][j] - stamp
            best_fit_par[j,1] += spot_coo['y_peak'][j] - stamp
#            print(fit.par)
#            print(stamp)

#            pyplot.imshow(img, origin='lower', interpolation='nearest', vmin=8., vmax=20.)
#            pyplot.scatter(spot_coo['x_peak'][j], spot_coo['y_peak'][j], facecolor='none',
#                           edgecolor='C3', marker='o', s=100)
#            pyplot.scatter(fit.par[0]-stamp+spot_coo['x_peak'][j],
#                           fit.par[1]-stamp+spot_coo['y_peak'][j],
#                           facecolor='none', edgecolor='C1', marker='s', s=100)
#            pyplot.show()

            plot_fit(fit, ofile=rdx_root / f'{_ifile.stem}_{j+1}.png')

        tab = table.Table()

        tab['x0'] = best_fit_par[:,0]
        tab['y0'] = best_fit_par[:,1]
        tab['amp'] = best_fit_par[:,2]
        tab['gamma1'] = best_fit_par[:,3]
        tab['gamma2'] = best_fit_par[:,4]
        tab['phi'] = best_fit_par[:,5]
        tab['beta'] = best_fit_par[:,6]
        tab['bkg'] = best_fit_par[:,7]
        tab['fwhm1'] = FitMoffat2D.to_fwhm(best_fit_par[:,3], best_fit_par[:,6])
        tab['fwhm2'] = FitMoffat2D.to_fwhm(best_fit_par[:,4], best_fit_par[:,6])
        ofile = rdx_root / f'{_ifile.stem}.fits'
        tab.write(ofile, overwrite=True)

#        ofile = None
        ofile = rdx_root / f'{_ifile.stem}_full.png'
        w,h = pyplot.figaspect(1.)
        fig = pyplot.figure(figsize=(1.5*w,1.5*h))

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_axis_off()
        ax.imshow(img, origin='lower', interpolation='nearest', vmin=8., vmax=12.)
        for j in range(nspots):
            fwhm1 = FitMoffat2D.to_fwhm(best_fit_par[j,3], best_fit_par[j,6])
            fwhm2 = FitMoffat2D.to_fwhm(best_fit_par[j,4], best_fit_par[j,6])
            ell_x, ell_y = make_ellipse(10*fwhm1, 10*fwhm2, best_fit_par[j,5])
            ell_x += best_fit_par[j,0]
            ell_y += best_fit_par[j,1]
            ax.plot(ell_x, ell_y, color='C3', lw=1)
        
        if ofile is None:
            pyplot.show()
        else:
            fig.canvas.print_figure(ofile, bbox_inches='tight')
        fig.clear()
        pyplot.close(fig)


if __name__ == '__main__':
    main()



