
######################################################################################
########  Create plots of PSF fit parameters wrt spacer width & image number  ########
######################################################################################

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import stats

from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps_bulk
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_stack, psf_plot

from nickelpipeline.convenience.dir_nav import categories_from_conditions, unzip_directories


# Labels for plotting
spacer_label = 'Spacer Width (in)'
fwhm_label = 'FWHM (pix)'
ecc_label = 'FWHM Eccentricity'
phi_label = 'Rotation Angle Phi'

def graph_psf_pars_bulk(path_list, condition_tuples, verbose=False):
    """
    Plot PSF parameters relative to category conditions (i.e. camera displacement 
    from tub). All source PSFs in all images within a category are stacked before 
    fitting a Moffat function. Parameters are extracted from this fit.

    Args:
        path_list (list): List of paths (directories or files) to unzip.
        condition_tuples (tuple): Conditions for categorizing images.
        verbose (bool): If True, print detailed output during processing.
    """
    # Gather files from all directories, sort into dict = {spacer_width: file_list}
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    # Fit PSFs & extract parameters
    results = zip(*[(category,) + extract_psf_par(file_list, 
                                                  str(category).replace('.', '_'), 
                                                  verbose=verbose)
                    for category, file_list in categories.items()])
    widths, fwhms, eccs, phis = results
    
    # Plot each parameter wrt category
    for param_list, label in zip([fwhms, eccs, phis], 
                                 [fwhm_label, ecc_label, phi_label]):
        plot_graph(widths, param_list, 
                   xlabel=spacer_label, ylabel=label, 
                   title=f'{label} vs. {spacer_label}')

def graph_psf_pars_many(path_list, condition_tuples, verbose=False):
    """
    Plot PSF parameters relative to category conditions (i.e. camera displacement 
    from tub). All source PSFs in an image are stacked before fitting a Moffat 
    function. Parameters are extracted from these fits and averaged for graphing.

    Args:
        path_list (list): List of paths (directories or files) to unzip.
        condition_tuples (tuple): Conditions for categorizing images.
        files (list): Files to process. Alternative to directories--default is None.
        verbose (bool): If True, print detailed output during processing.
    """
    # Gather files from all directories, sort into dict = {spacer_width: file_list}
    images = unzip_directories(path_list, output_format='Fits_Simple')
    categories = categories_from_conditions(condition_tuples, images)
    
    # Initialize lists
    widths, mean_fwhms, mean_eccs, mean_phis = [], [], [], []
    fwhms_conf_intervals, eccs_conf_intervals, phis_conf_intervals = [], [], []

    # Process each category and its file list
    for category, file_list in categories.items():
        # Fit PSFs & extract parameters
        fwhms, eccs, phis = zip(*(extract_psf_par([file], file.stem, verbose=verbose) 
                                  for file in file_list))
        
        # Calculate mean and confidence intervals
        widths.append(category)
        mean_fwhms.append(np.mean(fwhms))
        mean_eccs.append(np.mean(eccs))
        mean_phis.append(np.mean(phis))
        
        fwhms_conf_intervals.append(calc_conf_intervals(fwhms))
        eccs_conf_intervals.append(calc_conf_intervals(eccs))
        phis_conf_intervals.append(calc_conf_intervals(phis))

        # Print the results
        if verbose:
            print(f"For width {category}, mean FWHM = {np.mean(fwhms):.3f}")
            print(f"For width {category}, mean FWHM eccentricity = {np.mean(eccs):.3f}")
            print(f"For width {category}, mean phi = {np.mean(phis):.3f}")
    
    # Transpose confidence intervals to correct input format for error bar plots
    eccs_conf_intervals = np.array(eccs_conf_intervals).T
    phis_conf_intervals = np.array(phis_conf_intervals).T
    fwhms_conf_intervals = np.array(fwhms_conf_intervals).T

    # Plot each parameter with respect to category (using error bars)
    plot_objects = zip([mean_fwhms, mean_eccs, mean_phis,], 
                       [fwhm_label, ecc_label, phi_label],
                       [fwhms_conf_intervals, eccs_conf_intervals, phis_conf_intervals])
    
    for param_list, label, conf_intervals in plot_objects:
        plot_graph(widths, param_list, yerr=[conf_intervals[0], conf_intervals[1]], 
                   xlabel=spacer_label, ylabel=label, 
                   title=f'{label} vs. {spacer_label}', 
                   legend_label=f'{label} w/ {0.95*100}% Conf. Interval')

def graph_psf_pars_individuals(path_list, verbose=False):
    """
    Plot PSF parameters for every image. All source PSFs in an image are stacked before 
    fitting a Moffat function. Parameters are extracted from this fits.

    Args:
        path_list (list): List of paths (directories or files) to unzip.
        verbose (bool): If True, print detailed output during processing.
    """
    # Extract images from directories or specific files, converting to 'Fits_Simple' format
    images = unzip_directories(path_list, output_format='Fits_Simple')
    
    # Fit PSFs and extract parameters for each image
    results = [(image.image_num,) + extract_psf_par([image,], image.path.stem, verbose=verbose)
               for image in images]
    
    # Sort results based on image numbers
    results.sort()
    
    # Unzip results into separate lists for plotting
    img_nums, fwhms, eccs, phis = zip(*results)
    
    # Plot graphs for FWHM, eccentricity, and phi against image numbers
    for param_list, label in zip([fwhms, eccs, phis], [fwhm_label, ecc_label, phi_label]):
        img_num_label = "Image Number"
        plot_graph(img_nums, param_list, 
                xlabel=img_num_label, ylabel=label, 
                title=f'{label} vs. {img_num_label}')

def extract_psf_par(file_paths, group_name, verbose=False):
    """
    Fit PSF and extract parameters for given files (all stars in these files are 
    stacked), storing intermediates in folder proc_files/group_name.

    Args:
        file_paths (list): List of file paths to analyze.
        group_name (str): Folder for storing intermediates (proc_files/group_name)
        verbose (bool): If True, print detailed output during processing.

    Returns:
        tuple: Average FWHM, FWHM eccentricity, and rotation angle phi.
    """
    # Generate image stamps for the given files
    generate_stamps_bulk(file_paths, group_name, verbose=verbose)
    
    # Define directory and base path for processed files
    proc_dir = Path('.').resolve() / "proc_files"
    base = proc_dir / group_name / group_name
    
    # Fit PSF stack and get the fit results
    psf_file = Path(f'{str(base)}.psf.fits').resolve()  # PSF info stored here
    fit = fit_psf_stack(group_name, 1, psf_file, verbose=verbose)
    
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

def extract_psf_single(file):
    """
    Analyze PSF parameters for a single file.

    Args:
        file (Path): File to analyze.

    Returns:
        tuple: Average FWHM, FWHM eccentricity, and rotation angle phi.
    """
    return extract_psf_par([file], file.stem, verbose=True)

def calc_conf_intervals(data_list, confidence_level=0.95):
    """
    Calculate confidence intervals for a list of data, assuming normal distribution.

    Args:
        data_list (list): List of data points.
        confidence_level (float): Confidence level for interval. Defaults to 0.95.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    mean = np.mean(data_list)
    interval = stats.norm.interval(confidence=confidence_level, 
                                   loc=np.mean(data_list), 
                                   scale=stats.sem(data_list))
    return (abs((interval[0] - mean)), abs(interval[1] - mean))

def plot_graph(x, y, yerr=None, xlabel='', ylabel='', title='', legend_label=None):
    """
    Plot a graph with given x and y values, and optional error bars.

    Args:
        x (list): X-axis values.
        y (list): Y-axis values.
        yerr (list, optional): Y-axis error values. Defaults to None.
        xlabel (str, optional): Label for X-axis. Defaults to ''.
        ylabel (str, optional): Label for Y-axis. Defaults to ''.
        title (str, optional): Title of the graph. Defaults to ''.
        legend_label (str, optional): Label for the legend. Defaults to None.
    """
    plt.figure(figsize=(8, 5))
    
    # Plot with error bars if provided
    if yerr is not None:
        plt.errorbar(x, y, yerr=yerr, fmt='o', linestyle='-', ecolor='r', capsize=5, label=legend_label)
        plt.legend(loc='upper right')
    else:
        plt.plot(x, y, marker='o', linestyle='-')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()



