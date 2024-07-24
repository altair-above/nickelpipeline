
from pathlib import Path
from IPython import embed
import warnings
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import QTable, Table, Column
import astropy.units as u
import ccdproc

from nickelpipeline.convenience.nickel_data import gain, read_noise
from nickelpipeline.convenience.fits_class import nickel_fov_mask_cols_only
from nickelpipeline.convenience.log import default_logger

module_name = __name__.split('.')[-1]
logger = default_logger(module_name)

# Labels for different types of astronomical frames
bias_label = 'Bias'
dome_flat_label = 'Dome flat'
sky_flat_label = 'Flat'
dark_label = 'dark'
focus_label = 'focus'

def reduce_all(rawdir=None, table_path_in=None, table_path_out='reduction_files_table.yml',
               save_inters=False, excl_files=[], excl_obj_strs=[], excl_filts=[]):
    """
    Perform reduction of raw astronomical data frames (overscan subtraction,
    bias subtraction, flat division, cosmic ray masking)

    Args:
        rawdir (Path): Directory containing raw FITS files if no table_path_in.
        table_path_in (str): Path to input table file with raw FITS file information.
        table_path_out (str): Path to output table file for storing the raw FITS file information.
        save_inters (bool): If True, save intermediate results during processing.
        excl_files (list): List of file stems to exclude (exact match not necessary).
        excl_obj_strs (list): List of object strings to exclude (exact match not necessary).
        excl_filts (list): List of filter names to exclude.

    Returns:
        list: Paths to the saved reduced images.
    """
    # Organize raw files based on input directory or table
    file_df = organize_files(rawdir, table_path_in, table_path_out, excl_files, excl_obj_strs, excl_filts)
    
    # Set up directories for saving results
    parent_dir = file_df.paths[0].parent.parent if rawdir is None else rawdir.parent
    reddir = parent_dir / 'reduced'
    procdir = parent_dir / 'processing'
    Path.mkdir(reddir, exist_ok=True)
    Path.mkdir(procdir, exist_ok=True)
    
    # Initialize CCDData objects and remove cosmic rays
    logger.info("Initializing CCDData objects & removing cosmic rays")
    ccd_objs = [init_ccddata(file) for file in file_df.files]
    file_df.files = ccd_objs
    
    # Generate master bias and master flats
    master_bias = get_master_bias(file_df, save=save_inters, save_dir=procdir)
    master_flats = get_master_flats(file_df, save=save_inters, save_dir=procdir)

    # Filter out non-science files
    scifiles_mask = ((file_df.objects != bias_label) &
                    (file_df.objects != dark_label) &
                    (file_df.objects != dome_flat_label) &
                    (file_df.objects != sky_flat_label) &
                    (file_df.objects != focus_label)).values
    scifile_df = file_df.copy()[scifiles_mask]

    # Perform overscan subtraction & trimming
    logger.info(f"Performing overscan subtraction & trimming on {len(scifile_df.files)} science images")
    scifile_df.files = [trim_overscan(scifile) for scifile in scifile_df.files]
    if save_inters:
        save_results(scifile_df, 'over', procdir/'overscan')
    
    # Perform bias subtraction
    logger.info(f"Performing bias subtraction on {len(scifile_df.files)} science images")
    scifile_df.files = [ccdproc.subtract_bias(scifile, master_bias) 
                    for scifile in scifile_df.files]
    if save_inters:
        save_results(scifile_df, 'unbias', procdir/'unbias')

    # Perform flat division for each filter
    logger.info("Performing flat division")
    all_red_paths = []
    for filt in master_flats.keys():
        logger.debug(f"{filt} Filter:")
        scienceobjects = list(set(scifile_df.objects[scifile_df.filters == filt]))
        
        for scienceobject in scienceobjects:
            # Filter science files by object and filter
            sub_scifile_df = scifile_df.copy()[(scifile_df.objects == scienceobject) &
                                        (scifile_df.filters == filt)]
            # Create directory for each science target / filter combination
            sci_dir = reddir / (scienceobject + '_' + filt)
            
            # Perform flat division
            sub_scifile_df.files = [ccdproc.flat_correct(scifile, master_flats[filt]) 
                         for scifile in sub_scifile_df.files]
            
            red_paths = save_results(sub_scifile_df, 'red', sci_dir)
            all_red_paths += red_paths
    
    # Return
    logger.info(f"Flat divided images saved to {reddir}")
    logger.info("---- reduce_all() call ended")
    return all_red_paths

def organize_files(rawdir, table_path_in, table_path_out,
                   excl_files, excl_obj_strs, excl_filts):
    """
    Organize files by extracting metadata and applying exclusions. Saves information
    to / draws information from a table file, and comments out files to be excluded.

    Args:
        rawdir (Path or None): Directory to scan for raw FITS files.
        table_path_in (str or None): Path to input table file with raw FITS file information.
        table_path_out (str): Path to output table file for storing the raw FITS file information.
        excl_files (list): List of file stems to exclude (exact match not necessary).
        excl_obj_strs (list): List of object strings to exclude (exact match not necessary).
        excl_filts (list): List of filter names to exclude.

    Returns:
        pd.DataFrame: DataFrame containing organized file information.
    """
    if table_path_in is not None:
        # Extract raw files from an astropy Table file
        logger.info(f"---- reduce_all() called on Astropy table file {table_path_in}")
        file_table = Table.read(table_path_in, format='ascii.fixed_width')
        # Convert astropy table to pandas DataFrame
        file_df = file_table.to_pandas()
        file_df.insert(1, "files", file_df.paths)
        file_df.paths = [Path(file_path) for file_path in file_df.paths]
        logger.info(f"{len(file_df.paths)} raw files extracted from table file")
    else:
        # Extract raw files from the specified directory
        logger.info(f"---- reduce_all() called on directory {rawdir}")
        rawfiles = [file for file in Path(rawdir).iterdir() if (file.is_file())]
        logger.info(f"{len(rawfiles)} raw files extracted from raw directory")
    
        # Create DataFrame with file metadata
        obj_list = []
        filt_list = []
        for file in rawfiles:
            hdul = fits.open(str(file))
            obj_list.append(norm_str(hdul[0].header["OBJECT"]))
            filt_list.append(hdul[0].header["FILTNAM"])
            hdul.close()

        file_df = pd.DataFrame({
            "names": [file.stem for file in rawfiles],
            "files": rawfiles,
            "objects": obj_list,
            "filters": filt_list,
            "paths": rawfiles
            })
    
        # Save the table file for future reference
        logger.info(f"Saving table of file data to {table_path_out}")
        logger.debug(f"Default output file path is .yml for ease of commenting out files")
        file_table = Table.from_pandas(file_df)
        file_table.remove_column('files')
        
        # Write table as ascii file to table_path_out
        file_table.write(table_path_out, format='ascii.fixed_width', overwrite=True)
    
    # Apply manual exclusions based on provided criteria
    all_excluded_file_names = []
    
    def exclude(exclusion_list, file_df, axis):
        excl_func = create_exclusion_func(exclusion_list)
        excluded_file_names = list(file_df.names[axis.apply(lambda x: not excl_func(x))])
        new_file_df = file_df.copy()[axis.apply(excl_func)]
        return new_file_df, excluded_file_names
        # logger.info(f"Manually excluded files with names {excluded_file_names}")
    
    # Exclude by file name (excludes file if any str in excl_files is in the file name)
    file_df, excluded_file_names = exclude(excl_files, file_df, file_df.names)
    all_excluded_file_names += excluded_file_names
    logger.info(f"Manually excluded files with names {excluded_file_names}")
    
    # Exclude by object name (excludes file if any str in excl_obj_strs is in object name)
    file_df, excluded_file_names = exclude(excl_obj_strs, file_df, file_df.objects)
    all_excluded_file_names += excluded_file_names
    logger.info(f"Manually excluded files with {excl_obj_strs} in object name: {excluded_file_names}")
    
    # Exclude by filter (excludes file if any str in excl_filts is in filter name)
    file_df, excluded_file_names = exclude(excl_filts, file_df, file_df.filters)
    all_excluded_file_names += excluded_file_names
    logger.info(f"Manually excluded files with {excl_filts} filters: {excluded_file_names}")
    
    # Exclude all focus images automatically (excludes file if 'focus' is in object name)
    file_df, excluded_file_names = exclude([focus_label], file_df, file_df.objects)
    all_excluded_file_names += excluded_file_names
    logger.info(f"Automatically excluding files with 'Focus' in object name: {excluded_file_names}")
    
    # # Exclude by file name (excludes file if any str in excl_files is in the file name)
    # excl_func = create_exclusion_func(excl_files)
    # excluded_file_names = list(file_df.names[file_df.names.apply(lambda x: not excl_func(x))])
    # all_excluded_file_names += excluded_file_names
    # file_df = file_df.copy()[file_df.paths.apply(excl_func)]
    # logger.info(f"Manually excluded files with names {excluded_file_names}")
    
    # # Exclude by object name (excludes file if any str in excl_obj_strs is in object name)
    # excl_func = create_exclusion_func(excl_obj_strs)
    # excluded_file_names = list(file_df.names[file_df.objects.apply(lambda x: not excl_func(x))])
    # all_excluded_file_names += excluded_file_names
    # file_df = file_df.copy()[file_df.objects.apply(excl_func)]
    # logger.info(f"Manually excluded files with {excl_obj_strs} in object name: {excluded_file_names}")
    
    # # Exclude by filter (excludes file if any str in excl_filts is in filter name)
    # excl_func = create_exclusion_func(excl_filts)
    # excluded_file_names = list(file_df.names[file_df.filters.apply(lambda x: not excl_func(x))])
    # all_excluded_file_names += excluded_file_names
    # file_df = file_df.copy()[file_df.filters.apply(excl_func)]
    # logger.info(f"Manually excluded files with {excl_filts} filters: {excluded_file_names}")
    
    # # Exclude all focus images automatically (excludes file if 'focus' is in object name)
    # excl_func = create_exclusion_func([focus_label])
    # excluded_file_names = list(file_df.names[file_df.objects.apply(lambda x: not excl_func(x))])
    # all_excluded_file_names += excluded_file_names
    # file_df = file_df.copy()[file_df.objects.apply(excl_func)]
    # logger.info(f"Automatically excluding files with 'Focus' in object name: {excluded_file_names}")

    # If table file already exists, preserve the original table file, but show which files the table says to exclude
    if table_path_in is not None:
        already_excl_lines = comment_out_rows(all_excluded_file_names, table_path_out, modify=False)
        logger.info(f"Automatically excluding files already commented out in the table file: {already_excl_lines}")
        logger.debug(f"Since table_path_in has been provided, the table file is not modified to comment out manual exclusions {all_excluded_file_names}")
    # If table file didn't already exist, add '#' to excluded files' rows in table file to ignore them in future
    else:
        already_excl_lines = comment_out_rows(all_excluded_file_names, table_path_out, modify=True)
        logger.info(f"In {table_path_out}, commenting out ('#') manually excluded files")

    # Return
    return file_df

def comment_out_rows(excluded_file_names, table_file, modify=True):
    """
    Comment out specified rows in a table file based on exclusion criteria.

    Args:
        excluded_file_names (list): List of file names to comment out.
        table_file (str): Path to the table file to modify.
        modify (bool): Whether to modify the file or not.

    Returns:
        list: List of file names that were already commented out.
    """
    with open(table_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    already_excl_lines = []
    for line in lines:
        if line.strip().startswith('#'):
            # Retrieves just the file stem (i.e. 'd1001')
            already_excl_lines.append(line.split('|')[1].split(' ')[1])
            new_lines.append(line)
        elif modify and any(file_name in line for file_name in excluded_file_names):
            new_lines.append('#' + line)    # "comments out" the row
        elif modify:
            new_lines.append(line)
    
    if modify:
        with open(table_file, 'w') as f:
            f.writelines(new_lines)
    
    return already_excl_lines

def init_ccddata(frame):
    """
    Initialize a CCDData object from a FITS file and remove cosmic rays.

    Args:
        frame (str or Path): Path to the FITS file.

    Returns:
        CCDData: Initialized and processed CCDData object.
    """
    ccd = CCDData.read(frame, unit=u.adu)
    ccd.mask = nickel_fov_mask_cols_only
    ccd = ccdproc.cosmicray_lacosmic(ccd, gain_apply=False, gain=gain, 
                                     readnoise=read_noise, verbose=False)
    # Apply gain manually due to a bug in cosmicray_lacosmic function
    ccd.data = ccd.data * gain
    # Bug in cosmicray_lacosmic returns CCDData.data as a Quanity with incorrect
    # units electron/ADU if gain_apply=True. Therefore, we manually apply gain,
    # and leave ccd.data as a numpy array
    return ccd

def trim_overscan(ccd):
    """
    Subtract overscan and trim the overscan region from the image.

    Args:
        ccd (CCDData): CCDData object to process.

    Returns:
        CCDData: Processed CCDData object with overscan subtracted and image trimmed.
    """
    def nickel_oscansec(hdr):
        nc = hdr['NAXIS1']
        no = hdr['COVER']
        nr = hdr['NAXIS2']
        return f'[{nc-no+1}:{nc},1:{nr}]'
    
    oscansec = nickel_oscansec(ccd.header)
    proc_ccd = ccdproc.subtract_overscan(ccd, fits_section=oscansec, overscan_axis=1)
    return ccdproc.trim_image(proc_ccd, fits_section=ccd.header['DATASEC'])

def stack_frames(raw_frames, frame_type):
    """
    Stack frames by trimming overscan and combining them with sigma clipping.

    Args:
        raw_frames (list): List of CCDData objects to combine.
        frame_type (str): Type of frames (e.g., 'flat').

    Returns:
        CCDData: Combined CCDData object.
    """
    trimmed_frames = [trim_overscan(frame) for frame in raw_frames]
    combiner = ccdproc.Combiner(trimmed_frames)
    
    old_n_masked = 0
    new_n_masked = 1
    while new_n_masked > old_n_masked:
        combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.mean)
        old_n_masked = new_n_masked
        new_n_masked = combiner.data_arr.mask.sum()

    if frame_type == 'flat':
        scaling_func = lambda arr: 1/np.ma.average(arr)
        combiner.scaling = scaling_func
    stack = combiner.average_combine()  
    return stack

def get_master_bias(file_df, save=True, save_dir=None):
    """
    Create a master bias frame from individual bias frames.

    Args:
        file_df (pd.DataFrame): DataFrame containing file information.
        save (bool): If True, save the master bias frame to disk.
        save_dir (Path or None): Directory to save the master bias frame.

    Returns:
        CCDData: Master bias CCDData object.
    """
    logger.info("Combining bias files into master bias")
    bias_df = file_df.copy()[file_df.objects == bias_label]
    logger.info(f"Using {len(bias_df.files)} bias frames: {[file.name.split('_')[0] for file in bias_df.paths]}")

    master_bias = stack_frames(bias_df.files, frame_type='bias')
    
    if save:
        master_bias.header["OBJECT"] = "Master_Bias"
        master_bias.write(save_dir / 'master_bias.fits', overwrite=True)
        logger.info(f"Saving master bias to {save_dir / 'master_bias.fits'}")
    
    return master_bias

def get_master_flats(file_df, save=True, save_dir=None):
    """
    Create master flat frames (one per filter) from individual flat frames.

    Args:
        file_df (pd.DataFrame): DataFrame containing file information.
        save (bool): If True, save the master flat frames to disk.
        save_dir (Path or None): Directory to save the master flat frames.

    Returns:
        dict: Dictionary of master flat CCDData objects keyed by filter.
    """
    logger.info("Combining flat files into master flat")
    
    # Use sky flats if available, else use dome flats
    if sky_flat_label in list(set(file_df.objects)):
        flattype = sky_flat_label
    else:
        flattype = dome_flat_label
    logger.debug(f"Using flat type '{flattype}'")
    
    master_flats = {}
    
    # Make a master flat for all filts in which flats have been taken
    for filt in set(file_df.filters[file_df.objects == flattype]):
        flat_df = file_df.copy()[(file_df.objects == flattype) & (file_df.filters == filt)]
        logger.info(f"Using {len(flat_df.files)} flat frames: {[path.name.split('_')[0] for path in flat_df.paths]}")

        master_flat = stack_frames(flat_df.files, frame_type='flat')
        
        if save:
            master_flat.header["OBJECT"] = filt + "-Band_Master_Flat"
            master_flat.write(save_dir / ('master_flat_' + filt + '.fits'), overwrite=True)
            logger.info(f"Saving {filt}-band master flat to {save_dir / ('master_flat_' + filt + '.fits')}")
        master_flats[filt] = master_flat
    
    return master_flats

def save_results(scifile_df, modifier_str, save_dir):
    """
    Save (partially) processed science files to the specified directory.

    Args:
        scifile_df (pd.DataFrame): DataFrame containing processed science file information.
        modifier_str (str): String to append to filenames to indicate processing stage.
        save_dir (Path): Directory to save the processed files.

    Returns:
        list: List of paths to the saved files.
    """
    Path.mkdir(save_dir, exist_ok=True)
    logger.info(f"Saving {len(scifile_df.files)} fully reduced {save_dir.name} images to {save_dir}")
    save_paths = [save_dir / (path.name.split('_')[0] + f"_{modifier_str}" + path.suffix) for path in scifile_df.paths]
    for file, path in zip(scifile_df.files, save_paths):
        file.write(path, overwrite=True)
    return save_paths

def norm_str(s):
    """
    Normalize a string for comparison purposes--all caps, no spaces.
    'Sky flat' -> 'SKYFLAT'

    Args:
        s (str or list): String or list of strings to normalize.

    Returns:
        str or list: Normalized string or list of normalized strings.
    """
    if isinstance(s, list):
        return [norm_str(elem) for elem in s]
    return s.upper().replace(' ', '')

def create_exclusion_func(exclude_list):
    """
    Create a function to determine if a file should be excluded based on a list of criteria.

    Args:
        exclude_list (list): List of criteria for exclusion.

    Returns:
        function: Function that takes a target (string) and returns True if it should be excluded.
    """
    if exclude_list is None:
        return lambda _: True
    exclude_list = [norm_str(obj_str) for obj_str in exclude_list]
    def excl_func(target):
        target_str = norm_str(target)
        is_excluded = any(excluded_str in target_str for excluded_str in exclude_list)
        return not is_excluded
    return excl_func


bias_label = norm_str(bias_label)
dome_flat_label = norm_str(dome_flat_label)
sky_flat_label = norm_str(sky_flat_label)
dark_label = norm_str(dark_label)
focus_label = norm_str(focus_label)


# from pathlib import Path
# from IPython import embed
# import warnings
# import numpy as np
# import pandas as pd

# from astropy.io import fits
# from astropy.nddata import CCDData
# from astropy.table import QTable, Table, Column
# import astropy.units as u
# import ccdproc

# from nickelpipeline.convenience.nickel_data import gain, read_noise
# from nickelpipeline.convenience.fits_class import nickel_fov_mask_cols_only
# from nickelpipeline.convenience.log import default_logger

# module_name = __name__.split('.')[-1]
# logger = default_logger(module_name)

# # Set object type names
# bias_label = 'Bias'
# dome_flat_label = 'Dome flat'
# sky_flat_label = 'Flat'
# dark_label = 'dark'
# focus_label = 'focus'


# def reduce_all(rawdir=None, table_path_in=None, table_path_out='reduction_files_table.yml',
#                save_inters=False, excl_files=[], excl_obj_strs=[], excl_filts=[]):
#     # exclude = list of file stems (i.e. not .fits) from rawdir to be excluded
#     # exclude by range??
    
#     file_df = organize_files(rawdir, table_path_in, table_path_out, 
#                              excl_files, excl_obj_strs, excl_filts)
    
#     # Make directories for saving results
#     if rawdir is None:
#         parent_dir = file_df.paths[0].parent.parent
#     else:
#         parent_dir = rawdir.parent
#     reddir = parent_dir / 'reduced'
#     procdir = parent_dir / 'processing'
#     Path.mkdir(reddir, exist_ok=True)
#     Path.mkdir(procdir, exist_ok=True)
    
#     # Create CCDData objects
#     logger.info(f"Intializing CCDData objects & removing cosmic rays")
#     ccd_objs = [init_ccddata(file) for file in file_df.files]
#     file_df.files = ccd_objs
    
#     # Generate master bias & master flats
#     master_bias = get_master_bias(file_df, save=save_inters, save_dir=procdir)
#     master_flats = get_master_flats(file_df, save=save_inters, save_dir=procdir)

#     # Create new DataFrame with just object images
#     scifiles_mask = ((file_df.objects != bias_label) &
#                     (file_df.objects != dark_label) &
#                     (file_df.objects != dome_flat_label) &
#                     (file_df.objects != sky_flat_label) &
#                     (file_df.objects != focus_label)).values
#     scifile_df = file_df.copy()[scifiles_mask]

#     # Perform overscan subtraction & trimming
#     logger.info(f"Performing overscan subtraction & trimming on {len(scifile_df.files)} science images")
#     scifile_df.files = [trim_overscan(scifile) for scifile in scifile_df.files]
#     if save_inters:
#         save_results(scifile_df, 'over', procdir/'overscan')
    
#     # Perform bias subtraction
#     logger.info(f"Performing bias subtraction on {len(scifile_df.files)} science images")
#     scifile_df.files = [ccdproc.subtract_bias(scifile, master_bias) 
#                     for scifile in scifile_df.files]
#     if save_inters:
#         save_results(scifile_df, 'unbias', procdir/'unbias')

#     # Perform flat division
#     logger.info("Performing flat division")
#     all_red_paths = []
#     for filt in master_flats.keys():
#         logger.debug(f"{filt} Filter:")
#         scienceobjects = list(set(scifile_df.objects[scifile_df.filters == filt]))
        
#         for scienceobject in scienceobjects:
#             # Take the subset of scifile_df containing scienceobject in filter filt
#             sub_scifile_df = scifile_df.copy()[(scifile_df.objects == scienceobject) &
#                                         (scifile_df.filters == filt)]
#             # Make a new directory for each science target / filter combination
#             sci_dir = reddir / (scienceobject + '_' + filt)
            
#             # Do flat division
#             sub_scifile_df.files = [ccdproc.flat_correct(scifile, master_flats[filt]) 
#                          for scifile in sub_scifile_df.files]
            
#             red_paths = save_results(sub_scifile_df, 'red', sci_dir)
#             all_red_paths += red_paths
    
#     logger.info(f"Flat divided images saved to {reddir}")
#     logger.info("---- reduce_all() call ended")
#     return all_red_paths


# def organize_files(rawdir, table_path_in, table_path_out,
#                    excl_files, excl_obj_strs, excl_filts):

#     if table_path_in is not None:
#         # Extract raw files (as paths) from astropy Table file
#         logger.info(f"---- reduce_all() called on Astropy table file {table_path_in}")
#         file_table = Table.read(table_path_in, format='ascii.fixed_width')
#         file_df = file_table.to_pandas()
#         file_df.insert(1, "files", file_df.paths)
#         file_df.paths = [Path(file_path) for file_path in file_df.paths]
#         logger.info(f"{len(file_df.paths)} raw files extracted from table file")
#     else:
#         # Extract raw files (as paths) from directory
#         logger.info(f"---- reduce_all() called on directory {rawdir}")
#         rawfiles = [file for file in Path(rawdir).iterdir() if (file.is_file())]
#         logger.info(f"{len(rawfiles)} raw files extracted from raw directory")
    
#         # Create DataFrame for files
#         obj_list = []
#         filt_list = []
#         for file in rawfiles:
#             hdul = fits.open(str(file))
#             obj_list.append(norm_str(hdul[0].header["OBJECT"]))
#             filt_list.append(hdul[0].header["FILTNAM"])
#             hdul.close()

#         file_df = pd.DataFrame({
#             "names": [file.stem for file in rawfiles],
#             "files": rawfiles,
#             "objects": obj_list,
#             "filters": filt_list,
#             "paths": rawfiles
#             })
    
#         logger.info(f"Saving table of file data to {table_path_out}")
#         logger.debug(f"Default output file path is .yml for ease of commentting out files")
#         file_table = Table.from_pandas(file_df)
#         file_table.remove_column('files')
        
#         file_table.write(table_path_out, format='ascii.fixed_width', overwrite=True)
    
#     all_excluded_file_names = []
    
#     excl_func = create_exclusion_func(excl_files, mode='path')
#     excluded_file_names = list(file_df.names[file_df.paths.apply(lambda x: not excl_func(x))])
#     all_excluded_file_names += excluded_file_names
#     file_df = file_df.copy()[file_df.paths.apply(excl_func)]
#     logger.info(f"Manually excluded files with names {excluded_file_names}")
    
#     excl_func = create_exclusion_func(excl_obj_strs, mode='str')
#     excluded_file_names = list(file_df.names[file_df.objects.apply(lambda x: not excl_func(x))])
#     all_excluded_file_names += excluded_file_names
#     file_df = file_df.copy()[file_df.objects.apply(excl_func)]
#     logger.info(f"Manually excluded files with {excl_obj_strs} in the object name: {excluded_file_names}")
    
#     excl_func = create_exclusion_func(excl_filts, mode='str')
#     excluded_file_names = list(file_df.names[file_df.filters.apply(lambda x: not excl_func(x))])
#     all_excluded_file_names += excluded_file_names
#     file_df = file_df.copy()[file_df.filters.apply(excl_func)]
#     logger.info(f"Manually excluded files with {excl_filts} filters: {excluded_file_names}")
    
#     excl_func = create_exclusion_func([focus_label], mode='str')
#     excluded_file_names = list(file_df.names[file_df.objects.apply(lambda x: not excl_func(x))])
#     all_excluded_file_names += excluded_file_names
#     file_df = file_df.copy()[file_df.objects.apply(excl_func)]
#     logger.info(f"Automatically excluding files with 'Focus' in the object name: {excluded_file_names}")

#     if table_path_in is None:
#         already_excl_lines = comment_out_rows(all_excluded_file_names, table_path_out, modify=True)
#         logger.info(f"In {table_path_out}, commenting out ('#') manually excluded files")
#     else:
#         already_excl_lines = comment_out_rows(all_excluded_file_names, table_path_out, modify=False)
#         logger.info(f"Automatically excluding files already commented out in the table file: {already_excl_lines}")
#         logger.debug(f"Since table_path_in has been provided, the table file is not modified to comment out manual exclusions {all_excluded_file_names}")

#     return file_df


# def comment_out_rows(excluded_file_names, table_file, modify=True):
    
#     with open(table_file, 'r') as f:
#         lines = f.readlines()
    
#     new_lines = []
#     already_excl_lines = []
#     for line in lines:
#         if line.strip().startswith('#'):
#             already_excl_lines.append(line.split('|')[1].split(' ')[1])
#             new_lines.append(line)
#         elif modify and any(file_name in line for file_name in excluded_file_names):
#             new_lines.append('#' + line)
#         elif modify:
#             new_lines.append(line)
    
#     if modify:
#         with open(table_file, 'w') as f:
#             f.writelines(new_lines)
    
#     return already_excl_lines


# def init_ccddata(frame):
#     """_summary_

#     Args:
#         frame (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     ccd = CCDData.read(frame, unit=u.adu)
#     ccd.mask = nickel_fov_mask_cols_only
#     ccd = ccdproc.cosmicray_lacosmic(ccd, gain_apply=False, gain=gain, 
#                                      readnoise=read_noise, verbose=False)
#     # Bug in cosmicray_lacosmic returns CCDData.data as a Quanity with incorrect
#     # units electron/ADU if gain_apply=True. Therefore, we manually apply gain,
#     # and leave ccd.data as a numpy array
#     ccd.data = ccd.data * gain #* u.electron
#     return ccd


# def trim_overscan(frame):
#     def nickel_oscansec(hdr):
#         nc = hdr['NAXIS1']
#         no = hdr['COVER']
#         nr = hdr['NAXIS2']
#         return f'[{nc-no+1}:{nc},1:{nr}]'
    
#     # ccd = CCDData.read(frame, unit='adu')
#     ccd = frame
#     oscansec = nickel_oscansec(ccd.header)
#     proc_ccd = ccdproc.subtract_overscan(ccd, fits_section=oscansec, overscan_axis=1)
#     return ccdproc.trim_image(proc_ccd, fits_section=ccd.header['DATASEC'])


# def stack_frames(raw_frames, frame_type):
#     trimmed_frames = [trim_overscan(frame) for frame in raw_frames]
#     combiner = ccdproc.Combiner(trimmed_frames)
    
#     old_n_masked = 0
#     new_n_masked = 1
#     while new_n_masked > old_n_masked:
#         combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.mean)
#         old_n_masked = new_n_masked
#         new_n_masked = combiner.data_arr.mask.sum()

#     if frame_type == 'flat':
#         scaling_func = lambda arr: 1/np.ma.average(arr)
#         combiner.scaling = scaling_func
#     stack = combiner.average_combine()  
#     return stack


# def get_master_bias(file_df, save=True, save_dir=None):
#     logger.info(f"Combining bias files into master bias")
#     bias_df = file_df.copy()[file_df.objects == bias_label]
#     logger.info(f"Using {len(bias_df.files)} bias frames: {[file.name.split('_')[0] for file in bias_df.paths]}")

#     master_bias = stack_frames(bias_df.files, frame_type='bias')
    
#     if save:
#         master_bias.header["OBJECT"] = "Master_Bias"
#         master_bias.write(save_dir / 'master_bias.fits', overwrite=True)
#         logger.info(f"Saving master bias to {save_dir / 'master_bias.fits'}")
    
#     return master_bias
    

# def get_master_flats(file_df, save=True, save_dir=None):
#     logger.info(f"Combining flat files into master flat")
    
#     # Use sky flats if available, use dome flats if not
#     if sky_flat_label in list(set(file_df.objects)):
#         flattype = sky_flat_label
#     else:
#         flattype = dome_flat_label
#     logger.debug(f"Using flat type '{flattype}'")
    
#     filts = list(set(file_df.filters[file_df.objects == flattype]))
#     master_flats = {}
#     for filt in filts:
#         flat_df = file_df.copy()[(file_df.objects == flattype) & (file_df.filters == filt)]
#         logger.info(f"Using {len(flat_df.files)} flat frames: {[path.name.split('_')[0] for path in flat_df.paths]}")

#         master_flat = stack_frames(flat_df.files, frame_type='flat')
        
#         if save:
#             master_flat.header["OBJECT"] = filt + "-Band_Master_Flat"
#             master_flat.write(save_dir / ('master_flat_' + filt + '.fits'), overwrite=True)
#             logger.info(f"Saving {filt}-band master flat to {save_dir / ('master_flat_' + filt + '.fits')}")
#         master_flats[filt] = master_flat
    
#     return master_flats


# def save_results(scifile_df, modifier_str, save_dir):
#     Path.mkdir(save_dir, exist_ok=True)
#     logger.info(f"Saving {len(scifile_df.files)} fully reduced {save_dir.name} images to {save_dir}")
#     save_paths = [save_dir / (path.name.split('_')[0] + f"_{modifier_str}" + path.suffix) for path in scifile_df.paths]
#     for file, path in zip(scifile_df.files, save_paths):
#         file.write(path, overwrite=True)
#     return save_paths
    

# # Function to normalize comparison
# def norm_str(s):
#     if isinstance(s, list):
#         return [norm_str(elem) for elem in s]
#     return s.upper().replace(' ', '')

# bias_label = norm_str(bias_label)
# dome_flat_label = norm_str(dome_flat_label)
# sky_flat_label = norm_str(sky_flat_label)
# dark_label = norm_str(dark_label)
# focus_label = norm_str(focus_label)

# def create_exclusion_func(exclude_list, mode='str'):
#     if exclude_list is None:
#         return lambda _: True
#     if mode == 'str':
#         exclude_list = [norm_str(obj_str) for obj_str in exclude_list]
#     def excl_func(target):
#         target_str = target.name if mode == 'path' else target
#         is_excluded = any(excluded_str in target_str for excluded_str in exclude_list)
#         return not is_excluded
#     return excl_func



# def main():
    
#     return
    


# # if __name__ == '__main__':
# #     main()



