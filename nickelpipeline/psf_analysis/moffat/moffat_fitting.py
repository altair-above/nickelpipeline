

import numpy as np
from pathlib import Path
import logging

from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single, fit_psf_stack, psf_plot
from nickelpipeline.convenience.dir_nav import unzip_directories

logger = logging.getLogger(__name__)


def get_source_pars(path_list, category_str=None, fittype='elliptical'):
    """
    Extract source coordinates and Moffat fit parameters from image data.
    
    Args:
        path_list (list): List of paths (directories or files) to unzip.
        category_str (str): Category string for identifying the path to data
        fittype (str, optional): Type of model to fit ('elliptical' or 'circular')
    
    Returns:
        source_coords (ndarray): Array of source coordinates.
        source_pars (ndarray): Array of source parameters. (x0, y0, amplitude, gamma1, gamma2, phi, alpha, background)
    """
    # Unzip directories to get image files
    images = unzip_directories(path_list, output_format='Path')
    
    # Create output directories
    proc_dir = Path('.').resolve() / "proc_files"
    Path.mkdir(proc_dir, exist_ok=True)
    proc_subdir = proc_dir / fittype
    Path.mkdir(proc_subdir, exist_ok=True)
    base_parent = proc_subdir / category_str
    Path.mkdir(base_parent, exist_ok=True)
    base = proc_subdir / category_str / category_str
    
    # Generate stamps (image of sources) for image data
    generate_stamps(images, output_base=base)
    
    # Fit PSF models and get source coordinates and parameters
    source_coords, source_pars, img_nums = fit_psf_single(base, len(images))
    return source_coords, source_pars, img_nums


def get_graphable_pars(file_paths, group_name, verbose=False):
    """
    Fit PSF and extract parameters for given files (all stars in these files are 
    stacked), storing intermediates in folder proc_files/elliptical/group_name.

    Args:
        file_paths (list): List of file paths to analyze.
        group_name (str): Folder for storing intermediates (proc_files/group_name)
        verbose (bool): If True, print detailed output during processing.

    Returns:
        tuple: Average FWHM, FWHM eccentricity, and rotation angle phi.
    """
    # Define directory and base path for processed files
    proc_dir = Path('.').resolve() / "proc_files"
    Path.mkdir(proc_dir, exist_ok=True)
    proc_subdir = proc_dir / 'elliptical'
    Path.mkdir(proc_subdir, exist_ok=True)
    base_parent = proc_subdir / group_name
    Path.mkdir(base_parent, exist_ok=True)
    base = proc_subdir / group_name / group_name
    
    # Generate image stamps for the given files
    generate_stamps(file_paths, output_base=base)
    
    # Fit PSF stack and get the fit results
    psf_file = Path(f'{str(base)}.psf.fits').resolve()  # PSF info stored here
    fit = fit_psf_stack(base, 1, fittype='elliptical', ofile=psf_file)
    
    # Plot PSF and get FWHM and phi values
    plot_file = Path(f'{str(base)}.psf.pdf').resolve()  # Plots stored here
    fwhm1, fwhm2, phi = psf_plot(plot_file, fit, verbose=verbose)
    
    # Calculate average FWHM and eccentricity
    fwhm = (fwhm1 + fwhm2)/2
    ecc = np.sqrt(np.abs(fwhm1**2 - fwhm2**2))/max(fwhm1, fwhm2)
    
    if verbose:
        print(f"Avg FWHM = {fwhm:3f}")
        print(f"FWHM_ecc = {ecc:3f}")
        print(f"Rotation angle phi = {phi:3f}")
    
    return fwhm, ecc, phi

