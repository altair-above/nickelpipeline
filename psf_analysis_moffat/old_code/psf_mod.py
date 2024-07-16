
from pathlib import Path
from IPython import embed

import numpy
from scipy import optimize
from matplotlib import pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling.functional_models import Moffat2D, Moffat1D
from astropy.visualization import AsinhStretch, ZScaleInterval, ImageNormalize

from spot_analysis_mod import FitEllipticalMoffat2D, plot_fit, make_ellipse
from reduce_mod import rdx_science_sources

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
        lb[4], ub[4] = 1., 6.
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

    def model(self, par=None):
        if par is not None:
            self._set_par(par)
        return self.moff.evaluate(self.x, self.y, self.par[2], self.par[0], self.par[1],
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
        
        

def show_deimos(img, srcdb=None, vmin=None, vmax=None):
    nchips = 1
    img_shape = (2601, 1024)
    buff = 10
    frac = 1/nchips*0.9
    order = [1, 0, 3, 2, 5, 4, 7, 6]

    imaspect = img_shape[0]/img_shape[1]

    w,h = pyplot.figaspect(0.5)
    fig = pyplot.figure(figsize=(w,h))

    for i in range(nchips):
        ax = fig.add_axes([0.04 + i*frac*(1+buff/img_shape[0]), 0.5-frac*imaspect,
                           frac, 2*frac*imaspect])
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.imshow(img[order[i]], origin='lower', interpolation='nearest', # aspect='auto',
                  vmin=vmin, vmax=vmax)
        if srcdb is not None:
            indx = (srcdb[:,0].astype(int) == order[i]+1) & (srcdb[:,-1].astype(int) == 1)
            ax.scatter(srcdb[indx,2], srcdb[indx,3], marker='o', s=50, facecolor='none',
                       edgecolor='C1', zorder=10)

    pyplot.show()


def stack_and_fit_psf(file_root, ofile):
    
    fit_type = FitEllipticalMoffat2D
    # fit_type = FitMoffat2D
    
    proc_dir = Path('.').resolve() / "proc_files"
    base_parent = proc_dir / file_root
    base = proc_dir / file_root / file_root
    ofits = base.with_suffix('.rdx.fits')
    src_ofile = base.with_suffix('.src.db')

    hdu = fits.open(ofits)
    img = numpy.ma.MaskedArray(hdu['IMAGES'].data, mask=hdu['MASKS'].data.astype(bool))
    srcdb = numpy.genfromtxt(src_ofile, dtype=float)

    img_shape = tuple(hdu['IMAGES'].data.shape[1:])
    stamp_shape = tuple(hdu['STAMPS_01'].data.shape[1:])
    stamp_width = stamp_shape[0]

    x, y = numpy.meshgrid(numpy.arange(stamp_width, dtype=float)-stamp_width//2,
                          numpy.arange(stamp_width, dtype=float)-stamp_width//2)
    r = numpy.sqrt(x**2 + y**2)


#    show_deimos(img, srcdb=srcdb, vmin=80, vmax=500)

    indx = (srcdb[:,8].astype(int) == 1) & (numpy.log10(srcdb[:,7]) > 3.8)
#    q_indx = numpy.array([indx & (srcdb[:,2] < img_shape[1]/2) & (srcdb[:,3] < img_shape[0]/2),
#                          indx & (srcdb[:,2] > img_shape[1]/2) & (srcdb[:,3] < img_shape[0]/2),
#                          indx & (srcdb[:,2] < img_shape[1]/2) & (srcdb[:,3] > img_shape[0]/2),
#                          indx & (srcdb[:,2] > img_shape[1]/2) & (srcdb[:,3] > img_shape[0]/2)])
    nchips = 1

    psf_sum_stack = numpy.zeros((nchips,) + stamp_shape, dtype=float)
    psf_sum_model = numpy.zeros((nchips,) + stamp_shape, dtype=float)
    if fit_type == FitMoffat2D:
        psf_sum_model_par = numpy.zeros((nchips, 6), dtype=float)
    elif fit_type == FitEllipticalMoffat2D:
        psf_sum_model_par = numpy.zeros((nchips, 8), dtype=float)
#    psf_wgt_stack = numpy.zeros((nchips, 4) + stamp_shape, dtype=float)

    for i in range(nchips):
        on_chip = (srcdb[:,0] == i+1)
        stamp_indx = numpy.full(on_chip.size, -1, dtype=int)
        stamp_indx[on_chip] = numpy.arange(numpy.sum(on_chip))
        ext = f'STAMPS_{i+1:02}'

        in_q = on_chip & indx # & q_indx[j]
        flux = srcdb[in_q,7]
        psf_sum_stack[i,...] = numpy.sum(hdu[ext].data[stamp_indx[in_q]], axis=0) \
                                    / numpy.sum(flux)
#        clipped_stamps = sigma_clip(hdu[ext].data[stamp_indx[in_q]] / flux[:,None,None],
#                                    sigma=7., axis=0)
#        print(numpy.sum(clipped_stamps.mask))
#        psf_wgt_stack[i,j,...] \
#                = numpy.ma.sum(flux[:,None,None] * clipped_stamps, axis=0) \
#                    / numpy.sum(flux[:,None,None]
#                                * numpy.logical_not(clipped_stamps.mask).astype(float), axis=0)

#        pyplot.imshow(psf_sum_stack[i], origin='lower', interpolation='nearest')
#        pyplot.show()

        # fit = FitMoffat2D(psf_sum_stack[i])
        fit = fit_type(psf_sum_stack[i])
        fwhm = 11
        alpha = 3.5
        # fwhm =  5
        # alpha = 2.0
        gamma = fwhm / 2 / numpy.sqrt(2**(1/alpha)-1)
        if fit_type == FitMoffat2D:
            p0 = numpy.array([float(stamp_width//2), float(stamp_width//2),
                            numpy.amax(psf_sum_stack[i]), gamma, alpha, 0.0])
            # Parameters are x0, y0, amplitude, gamma (width), alpha (fall-off slope), background.
        elif fit_type == FitEllipticalMoffat2D:
            p0 = numpy.array([float(stamp_width//2), float(stamp_width//2),
                            numpy.amax(psf_sum_stack[i]), gamma, gamma, 0.0,
                            alpha, 0.0])
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
            # p0 = numpy.array([float(stamp), float(stamp), numpy.amax(sub_img), gamma, 
            #                  gamma, 0., alpha, numpy.median(sub_img)])
            
        fit.fit(p0=p0)
        psf_sum_model[i,...] = fit.model()
        psf_sum_model_par[i,...] = fit.par

#        r = numpy.sqrt((fit.x - fit.par[0])**2 + (fit.y - fit.par[1])**2)
#
#        pyplot.scatter(r.ravel(), psf_sum_stack[i].ravel(), marker='.', lw=0, s=30,
#                       color='k')
#        pyplot.scatter(r.ravel(), fit.model().ravel(), lw=0, s=30, color='C3')
##        #pyplot.yscale('log')
#        pyplot.show()

    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=psf_sum_stack, name='STACK'),
                  fits.ImageHDU(data=psf_sum_model, name='MOFFAT'),
                  fits.ImageHDU(data=psf_sum_model_par, name='PAR')
                 ]).writeto(str(ofile), overwrite=True)
    
    return fit


def psf_plot(file_root, psf_file, plot_file):
    
    fit_type = FitEllipticalMoffat2D
    # fit_type = FitMoffat2D

    hdu = fits.open(psf_file)
    stamp_shape = hdu['STACK'].data.shape[1:]
    nchips = 1

    x, y = numpy.meshgrid(numpy.arange(stamp_shape[0]).astype(float),
                          numpy.arange(stamp_shape[1]).astype(float))

#    ploty = [0.71, 0.50, 0.29, 0.08]
    ploty = 0.71

    with PdfPages(plot_file) as pdf:
        for i in range(nchips):

            w,h = pyplot.figaspect(1.)
            fig = pyplot.figure(figsize=(1.5*w,1.5*h))

            stack = hdu['STACK'].data[i]
            model = hdu['MOFFAT'].data[i]
            amp = hdu['PAR'].data[i,2]
            beta = hdu['PAR'].data[i,4]
            # fwhm = FitMoffat2D.to_fwhm(hdu['PAR'].data[i,3], beta)
            fwhm = fit_type.to_fwhm(hdu['PAR'].data[i,3], beta)
            norm = ImageNormalize(numpy.concatenate((stack, model, stack-model)),
                                  interval=ZScaleInterval(contrast=0.10),
                                  stretch=AsinhStretch())

            ax = fig.add_axes([0.05, ploty, 0.2, 0.2])
            ax.imshow(stack, origin='lower', interpolation='nearest', norm=norm)
            ax.contour(stack, [amp/8, amp/4, amp/2, amp/1.1], colors='k', linewidths=0.5)
            ax.set_axis_off()
            ax.hlines([5], [5], [5+1./0.1185], color='C1', lw=2)
            ax.text(5+0.5/0.1185, 5+1, '1"', color='C1', ha='center', va='bottom')
#            ax.text(-0.05, 0.5, f'Q{j+1}', ha='center', va='center', rotation='vertical',
#                    transform=ax.transAxes)
#            if j == 0:
            ax.text(0.5, 1.01, 'Observed', ha='center', va='bottom', transform=ax.transAxes)
#            elif j == 3:
            ax.text(0.3, -0.2, f'{file_root} Amp {i+1}', ha='left', va='center',
                    transform=ax.transAxes, fontsize=14)


            ax = fig.add_axes([0.26, ploty, 0.2, 0.2])
            ax.imshow(model, origin='lower', interpolation='nearest', norm=norm)
            ax.contour(model, [amp/8, amp/4, amp/2, amp/1.1], colors='k', linewidths=0.5)
            ax.set_axis_off()
#            if j == 0:
            ax.text(0.5, 1.01, 'Model', ha='center', va='bottom', transform=ax.transAxes)

            ax = fig.add_axes([0.47, ploty, 0.2, 0.2])
            ax.imshow(stack-model, origin='lower', interpolation='nearest', norm=norm)
            ax.contour(stack-model, [-amp/40, amp/40], colors=['w','k'], linewidths=0.5)
            ax.set_axis_off()
#            if j == 0:
            ax.text(0.5, 1.01, 'Residual', ha='center', va='bottom', transform=ax.transAxes)

            r = numpy.sqrt((x - hdu['PAR'].data[i,0])**2 
                            + (y - hdu['PAR'].data[i,1])**2).ravel()
            srt = numpy.argsort(r)

            rlim = numpy.array([0, 40])
            ax = fig.add_axes([0.68, ploty, 0.3, 0.2])
            ax.plot(r[srt], model.ravel()[srt], color='C3')
            ax.minorticks_on()
            ax.set_xlim(rlim)
            ax.tick_params(axis='x', which='both', direction='in')
            ax.tick_params(axis='y', which='both', left=False, right=False)
            ax.scatter(r, stack.ravel(), marker='.', lw=0, s=30, alpha=0.5, color='k')
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.text(0.95, 0.9, f'FWHM = {fwhm:.1f} pix; {fwhm*0.1185:.2f}"', ha='right',
                    va='center', transform=ax.transAxes)
            ax.text(0.95, 0.78, f'beta = {beta:.2f}', ha='right', va='center',
                    transform=ax.transAxes)
#            if j == 0:
            ax.text(0.5, 1.15, 'R [arcsec]', ha='center', va='bottom',
                    transform=ax.transAxes)
#            elif j == 3:
            ax.text(0.5, -0.15, 'R [pix]', ha='center', va='top', transform=ax.transAxes)

            axt = ax.twiny()
            axt.set_xlim(rlim*0.1185)
            axt.minorticks_on()
            axt.tick_params(axis='x', which='both', direction='in')
            axt.tick_params(axis='y', which='both', left=False, right=False)

#            if j == 3:
#                axt.xaxis.set_major_formatter(ticker.NullFormatter())
#            elif j == 0:
#                ax.xaxis.set_major_formatter(ticker.NullFormatter())
#            else:
#                axt.xaxis.set_major_formatter(ticker.NullFormatter())
#                ax.xaxis.set_major_formatter(ticker.NullFormatter())

            pdf.savefig()
            fig.clear()
            pyplot.close()



def psf_plot_elliptical(file_root, psf_file, plot_file, fit, ofile=None):
    
    fit_type = FitEllipticalMoffat2D
    # fit_type = FitMoffat2D

    hdu = fits.open(psf_file)
    stamp_shape = hdu['STACK'].data.shape[1:]
    nchips = 1

    x, y = numpy.meshgrid(numpy.arange(stamp_shape[0]).astype(float),
                          numpy.arange(stamp_shape[1]).astype(float))

#    ploty = [0.71, 0.50, 0.29, 0.08]
    ploty = 0.71

    with PdfPages(plot_file) as pdf:
        
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
    
        pdf.savefig()
        fig.clear()
        pyplot.close()

        # if ofile is None:
        #     pyplot.show()
        # else:
        #     fig.canvas.print_figure(ofile, bbox_inches='tight')
        # fig.clear()
        # pyplot.close(fig)



def run_all(file, fit_type=FitEllipticalMoffat2D):
    # reduced_img = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw-reduced/PG1323-085_V/d1046_red.fits')
    reduced_img = Path(file)
    rdx_science_sources(reduced_img)
    
    file_root = reduced_img.stem
    
    proc_dir = Path('.').resolve() / "proc_files"
    base = proc_dir / file_root / file_root
    
    if fit_type == FitMoffat2D:
        psf_file = Path(f'{str(base)}.psf.fits').resolve()
        if not psf_file.is_file():
            fit = stack_and_fit_psf(file_root, psf_file)
            # plot_fit(fit)
        plot_file = Path(f'{str(base)}.psf.pdf').resolve()
        if not plot_file.is_file():
            psf_plot(file_root, psf_file, plot_file)
    elif fit_type == FitEllipticalMoffat2D:
        psf_file = Path(f'{str(base)}.psf.fits').resolve()
        fit = stack_and_fit_psf(file_root, psf_file)
        plot_fit(fit)
        
        plot_file = Path(f'{str(base)}.psf.pdf').resolve()
        psf_plot_elliptical(file_root, psf_file, plot_file, fit)


def main():
    
    # file_root = 'd1046_red'
    
    # proc_dir = Path('.').resolve() / "proc_files"
    # base = proc_dir / file_root / file_root
    
    # psf_file = Path(f'{str(base)}.psf.fits').resolve()
    # if not psf_file.is_file():
    #     stack_and_fit_psf(file_root, psf_file)
    # plot_file = Path(f'{str(base)}.psf.pdf').resolve()
    # if not plot_file.is_file():
    #     psf_plot(file_root, psf_file, plot_file)
    
    return

if __name__ == '__main__':
    main()

