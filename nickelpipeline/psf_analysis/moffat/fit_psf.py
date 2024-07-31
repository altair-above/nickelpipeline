
###################################################################################
########  Fit Moffat function PSFs to stamps, plot these stamps and plots  ########
###################################################################################

import numpy as np
from pathlib import Path
from matplotlib import pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
import logging

from astropy.io import fits
from astropy.modeling.functional_models import Moffat1D
from astropy.visualization import AsinhStretch, ZScaleInterval, ImageNormalize
from astropy.stats import SigmaClip

from nickelpipeline.psf_analysis.moffat.model_psf import FitEllipticalMoffat2D, FitMoffat2D, make_ellipse

logger = logging.getLogger(__name__)


def fit_psf_stack(input_base, num_images, fittype='elliptical', ofile=None):
    """
    Fit one PSF to the stack of all sources found in the directory specified,
    and save this information to relevant files

    Args:
        input_base (Path): Base of path to files w/ stamp data
        num_images (int): Number of images to process.
        ofile (str, optional): Output file path.
        verbose (bool, optional): If True, print detailed output during processing.
    """
    return fit_psf_generic('stack', input_base, num_images, fittype, ofile)

def fit_psf_single(input_base, num_images, fittype='elliptical'):
    """
    Fit a PSF to each source found in the directory specified, and return the
    source coordinates, fit parameters, and image number

    Args:
        input_base (Path): Base of path to files w/ stamp data
        num_images (int): Number of images to process.
        verbose (bool, optional): If True, print detailed output during processing.
        
    Returns:
        ndarray: Coordinates of all sources
        ndarray: Fit parameters of all sources
        ndarray: Image number of all sources
    """
    return fit_psf_generic('single', input_base, num_images, fittype, None)

def fit_psf_generic(mode, input_base, num_images, fittype='elliptical', 
                    ofile=None):
    """
    Generic function to fit PSFs to images.

    Args:
        mode (str): Mode of fitting ('stack' or 'single').
        input_base (Path): Base of path to files w/ stamp data
        num_images (int): Number of images in directory to process.
        fittype (str, optional): Type of model to fit ('elliptical' or 'circular')
        ofile (str, optional): Output file path for 'stack' mode.
        verbose (bool, optional): If True, print detailed output during processing.
    """
    if fittype == 'elliptical':
        fitter = FitEllipticalMoffat2D  # Type of fitting function to use
        num_pars = 8
    elif fittype == 'circular':
        fitter = FitMoffat2D  # Type of fitting function to use
        num_pars = 6
    else:
        raise ValueError("fitter must be 'elliptical' or 'circular'")

    # Set up directories and files
    # proc_dir = Path('.').resolve() / "proc_files"
    # base = proc_dir / category_str / (category_str) #+ f"_{fittype[:4]}")
    ofits = input_base.with_suffix('.rdx.fits')  # Path to FITS file
    src_ofile = input_base.with_suffix('.src.db')  # Path to source database file

    # Load data from FITS and source files
    hdu = fits.open(ofits)  # Open FITS file
    srcdb = np.genfromtxt(src_ofile, dtype=float)  # Load source database

    # Pull stamp shape and width from FITS file
    stamp_shape = tuple(hdu['STAMPS_01'].data.shape[1:])  # Shape of the stamp images
    stamp_width = stamp_shape[0]  # Width of the stamps
    indx = (srcdb[:,8].astype(int) == 1) & (np.log10(srcdb[:,7]) > 2.0)  # Selection criteria

    # Initialize arrays for 'stack' mode
    if mode == 'stack':
        psf_sum_stack = np.zeros((num_images,) + stamp_shape, dtype=float)
        psf_sum_model = np.zeros((num_images,) + stamp_shape, dtype=float)
        psf_sum_model_par = np.zeros((num_images, num_pars), dtype=float)
    
    # Initialize arrays for 'single' mode
    elif mode == 'single':
        centroid_coords = []  # List to store centroid coordinates
        fit_pars = []  # List to store fit parameters
        source_images = []  # List to store the image number of all sources
        

    # Process each image
    for i in range(num_images):
        on_chip = (srcdb[:,0] == i+1)
        stamp_indx = np.full(on_chip.size, -1, dtype=int)
        stamp_indx[on_chip] = np.arange(np.sum(on_chip))
        ext = f'STAMPS_{i+1:02}'  # Extension name for FITS file
        in_q = on_chip & indx
        
        # Initial parameters for Moffat function
        default_fwhm = 8
        alpha = 3.5
        gamma = default_fwhm / 2 / np.sqrt(2**(1/alpha)-1)
        
        def get_p0(fittype, stamp):
            if fittype == 'elliptical':
                return np.array([float(stamp_width//2), float(stamp_width//2),
                                 np.amax(stamp[i]), gamma, gamma, 0.0,
                                 alpha, 0.0])
            elif fittype == 'circular':
                return np.array([float(stamp_width//2), float(stamp_width//2),
                                 np.amax(stamp[i]), gamma, alpha, 0.0])
        
        if mode == 'stack':
            # Stack mode: Sum the stamps and divide by flux before fitting
            psf_sum_stack[i,...] = np.sum(hdu[ext].data[stamp_indx[in_q]], axis=0) \
                                        / np.sum(srcdb[in_q,7])
            # Initial guess for Moffat parameters
            p0 = get_p0(fittype, psf_sum_stack[i])
            
            fit = fitter(psf_sum_stack[i])  # Initialize fit object
            fit.fit(p0=p0)  # Perform the fit
            psf_sum_model[i,...] = fit.model()  # Get the model image
            psf_sum_model_par[i,...] = fit.par  # Get the fit parameters
        
        elif mode == 'single':
            # Single mode: Fit each individual stamp
            for step_num, stamp_img in enumerate(hdu[ext].data[stamp_indx[in_q]]):
                # Initial guess for Moffat parameters
                p0 = get_p0(fittype, stamp_img)
                
                fit = fitter(stamp_img)  # Initialize fit object
                try:
                    fit.fit(p0=p0)  # Perform the fit
                except ValueError:
                    continue
                fit_par = fit.par  # Get the fit parameters
                
                # Find centroid coordinates
                condition = (srcdb[:, 0] == i+1) & (srcdb[:, 1] == step_num)
                centroid_x = srcdb[condition][0][2]
                centroid_y = srcdb[condition][0][3]
                
                centroid_coords.append((centroid_x, centroid_y))  # Store coordinates
                fit_pars.append(fit_par)  # Store fit parameters
                source_images.append(i)
            
    hdu.close()  # Close the FITS file
    # ^ interrupting code during run may leave hdu open--just restart kernel
    
    if mode == 'stack':
        # Save the results to a new FITS file
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(data=psf_sum_stack, name='STACK'),
                      fits.ImageHDU(data=psf_sum_model, name='MOFFAT'),
                      fits.ImageHDU(data=psf_sum_model_par, name='PAR')
                     ]).writeto(str(ofile), overwrite=True)
        return fit
    
    elif mode == 'single':
        # Eliminate sources with irregular FWHMs
        fit_pars = np.array(fit_pars)
        source_images = np.array(source_images)
        centroid_coords = np.array(centroid_coords)
        if fittype == 'elliptical':
            fwhm1 = FitMoffat2D.to_fwhm(fit_pars[:,3], fit_pars[:,6])
        elif fittype == 'circular':
            fwhm1 = FitMoffat2D.to_fwhm(fit_pars[:,3], fit_pars[:,4])
        
        # Create a SigmaClip object and apply it to get a mask
        sigma_clip = SigmaClip(sigma=4, maxiters=5)
        masked_fwhm1 = sigma_clip(fwhm1)
        clipped_fit_pars = fit_pars[~masked_fwhm1.mask]
        clipped_coords = centroid_coords[~masked_fwhm1.mask]
        clipped_source_images = source_images[~masked_fwhm1.mask]
        
        if fittype == 'elliptical':
            fwhm2 = FitMoffat2D.to_fwhm(clipped_fit_pars[:,4], clipped_fit_pars[:,6])
            masked_fwhm2 = sigma_clip(fwhm2)
            clipped_fit_pars = clipped_fit_pars[~masked_fwhm2.mask]
            clipped_coords = clipped_coords[~masked_fwhm2.mask]
            clipped_source_images = clipped_source_images[~masked_fwhm2.mask]
        
        logger.info(f"Number of sources removed in sigma clipping = {len(fit_pars) - len(clipped_fit_pars)}")
        logger.info(f"Number of sources remaining = {len(clipped_fit_pars)}")
        
        return clipped_coords, clipped_fit_pars, clipped_source_images


def psf_plot(plot_file, fit, fittype='elliptical', verbose=False):
    """
    Plot the PSF fitting results and save to a PDF

    Args:
        plot_file (str): Output PDF file path.
        fit (object): Fitting results.
        verbose (bool, optional): If True, print detailed output during processing.
    """
    if fittype != 'elliptical':
        raise ValueError(f"psf_plot() not yet implemented for fittype={fittype}")
    with PdfPages(plot_file) as pdf:
        # Set up the figure
        w, h = pyplot.figaspect(1.)
        fig = pyplot.figure(figsize=(1.5*w,1.5*h))
        pyplot.suptitle(plot_file.stem)  # Set the title of the plot

        stack = fit.c  # Observed stack
        model = fit.model()  # Model stack

        amp = fit.par[2]  # Amplitude of the fit
        if isinstance(fit, FitMoffat2D):
            beta = fit.par[4]  # Moffat beta parameter
            fwhm1 = FitMoffat2D.to_fwhm(fit.par[3], beta)  # Calculate FWHM
            ell_x, ell_y = make_ellipse(fwhm1, fwhm1, 0.)  # Create ellipse for plotting
        else:
            beta = fit.par[6]  # Moffat beta parameter
            phi = fit.get_nice_phi(fit.par)  # Calculate rotation angle
            fwhm1 = FitMoffat2D.to_fwhm(fit.par[3], beta)  # Calculate FWHM1
            fwhm2 = FitMoffat2D.to_fwhm(fit.par[4], beta)  # Calculate FWHM2
            ell_x, ell_y = make_ellipse(fwhm1, fwhm2, fit.par[5])  # Create ellipse for plotting
            if fwhm1 < fwhm2:
                fwhm1, fwhm2 = fwhm2, fwhm1  # Swap FWHMs if necessary
        ell_x += fit.par[0]  # Offset ellipse in x
        ell_y += fit.par[1]  # Offset ellipse in y
            
        # Normalize the images for better visualization
        norm = ImageNormalize(np.concatenate((stack, model, stack-model)),
                              interval=ZScaleInterval(contrast=0.10),
                              stretch=AsinhStretch())

        # Plot observed stack
        ax = fig.add_axes([0.03, 0.7, 0.2, 0.2])
        ax.imshow(stack, origin='lower', interpolation='nearest', norm=norm)
        ax.contour(stack, [amp/8, amp/4, amp/2, amp/1.1], colors='k', linewidths=0.5)
        ax.set_axis_off()
        ax.text(0.5, 1.01, 'Observed', ha='center', va='bottom', transform=ax.transAxes)

        # Plot model stack
        ax = fig.add_axes([0.24, 0.7, 0.2, 0.2])
        ax.imshow(model, origin='lower', interpolation='nearest', norm=norm)
        ax.contour(model, [amp/8, amp/4, amp/2, amp/1.1], colors='k', linewidths=0.5)
        ax.plot(ell_x, ell_y, color='C3', lw=0.5)
        ax.set_axis_off()
        ax.text(0.5, 1.01, 'Model', ha='center', va='bottom', transform=ax.transAxes)

        # Plot residuals
        ax = fig.add_axes([0.45, 0.7, 0.2, 0.2])
        ax.imshow(stack-model, origin='lower', interpolation='nearest', norm=norm)
        ax.contour(stack-model, [-amp/40, amp/40], colors=['w','k'], linewidths=0.5)
        ax.set_axis_off()
        ax.text(0.5, 1.01, 'Residual', ha='center', va='bottom', transform=ax.transAxes)

        # Plot 1D profiles of source data and model
        r = np.sqrt((fit.x - fit.par[0])**2 + (fit.y - fit.par[1])**2).ravel()
        rlim = np.array([0, 5*fwhm1])  # Radius limits
    
        oned = Moffat1D()  # Initialize 1D Moffat function
        r_mod = np.linspace(*rlim, 100)  # Radial positions
        if isinstance(fit, FitMoffat2D):
            models = [oned.evaluate(r_mod, amp, 0., fit.par[3], beta) + fit.par[5]]
        else:
            models = [oned.evaluate(r_mod, amp, 0., fit.par[3], beta) + fit.par[7],
                      oned.evaluate(r_mod, amp, 0., fit.par[4], beta) + fit.par[7]]
        
        ax = fig.add_axes([0.66, 0.7, 0.3, 0.2])
        ax.minorticks_on()
        ax.set_xlim(rlim)
        ax.tick_params(axis='x', which='both', direction='in')
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        for model in models:
            ax.plot(r_mod, model, color='C3')
        ax.scatter(r, stack.ravel(), marker='.', lw=0, s=30, alpha=0.5, color='k')

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
        ax.text(0.5, -0.15, 'R [pix]', ha='center', va='top', transform=ax.transAxes)
            
        pdf.savefig()  # Save the figure to the PDF
        if verbose:
            pyplot.show()  # Display the plot
            fig.clear()
        pyplot.close()
        
        return fwhm1, fwhm2, phi


def main():
    """
    Main function to run the script.
    """
    return

if __name__ == '__main__':
    main()

