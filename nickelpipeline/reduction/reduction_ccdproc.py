
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

# Set object type names
bias_label = 'Bias'
dome_flat_label = 'Dome flat'
sky_flat_label = 'Flat'
dark_label = 'dark'
focus_label = 'focus'


# def reduce_all(rawdir=None, file_df_in=None, file_df_out='reduction_files_df.csv', save_inters=False, 
#                excl_files=None, excl_obj_strs=None, excl_filts=None):
#     # exclude = list of file stems (i.e. not .fits) from rawdir to be excluded
#     # exclude by range??
        
#     if file_df_in is not None:
#         # Extract raw files (as paths) from astropy Table file
#         logger.info(f"---- reduce_all() called on Astropy table file {file_df_in}")
#         # file_table = Table.read(file_table_in, format='ascii.fixed_width')
        
#         file_df = pd.read_csv(file_df_in)
#         # rawfiles = file_df.paths
#         # rawfiles = [Path(file_path) for file_path in file_table['Path']]
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
#             "names": [file_path.stem for file_path in rawfiles],
#             "files": None,
#             "objects": obj_list,
#             "filts": filt_list,
#             "paths": rawfiles
#             })
    
#     # Make directories for saving results
#     if rawdir is None:
#         parent_dir = rawfiles[0].parent.parent
#     else:
#         parent_dir = rawdir.parent
#     reddir = parent_dir / 'reduced'
#     procdir = parent_dir / 'processing'
#     Path.mkdir(reddir, exist_ok=True)
#     Path.mkdir(procdir, exist_ok=True)
    
#     if file_df_in is None:
#         file_df.to_csv(file_df_out)
#         file_df.to_csv('readable_data.csv', index=False, sep='\t', lineterminator='\n')

    
#     # if file_table_in is None:
#     #     file_table = Table()
#     #     file_table['File Name'] = [file.name for file in file_df.paths]
#     #     file_table['Object'] = file_df.objects
#     #     file_table['Filter'] = file_df.filts
#     #     file_table['Path'] = file_df.paths
#     #     file_table.write(file_table_out, format='ascii.fixed_width')
    
#     exclude_func = create_exclusion_func(excl_files, mode='path')
#     file_df = file_df.copy()[file_df.paths.apply(exclude_func)]
#     logger.info(f"Excluded files {excl_files}")
    
#     excl_obj_strs.append(focus_label)
#     exclude_func = create_exclusion_func(excl_obj_strs, mode='str')
#     file_df = file_df.copy()[file_df.objects.apply(exclude_func)]
#     logger.info(f"Excluded files with {excl_obj_strs} in the object name")
    
#     exclude_func = create_exclusion_func(excl_filts, mode='str')
#     file_df = file_df.copy()[file_df.filts.apply(exclude_func)]
#     logger.info(f"Excluded files with {excl_obj_strs} in the object name")

#     # Create CCDData objects
#     logger.info(f"Intializing CCDData objects & removing cosmic rays")
#     ccd_objs = [init_ccddata(path) for path in file_df.paths]
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
#         scienceobjects = list(set(scifile_df.objects[scifile_df.filts == filt]))
        
#         for scienceobject in scienceobjects:
#             # Take the subset of scifile_df containing scienceobject in filter filt
#             sub_scifile_df = scifile_df.copy()[(scifile_df.objects == scienceobject) &
#                                         (scifile_df.filts == filt)]
#             # Make a new directory for each science target / filter combination
#             sci_dir = reddir / (scienceobject + '_' + filt)
            
#             # Do flat division
#             sub_scifile_df.files = [ccdproc.flat_correct(scifile, master_flats[filt]) 
#                          for scifile in sub_scifile_df.files]
            
#             logger.info(f"{filt} Filter - Saving {len(sub_scifile_df.files)} fully reduced {scienceobject} images to {sci_dir}")
#             red_paths = save_results(sub_scifile_df, 'red', sci_dir)
#             all_red_paths += red_paths
    
#     logger.info(f"Flat divided images saved to {reddir}")
#     logger.info("---- reduce_all() call ended")
#     return all_red_paths


def init_ccddata(frame):
    ccd = CCDData.read(frame, unit=u.adu)
    ccd.mask = nickel_fov_mask_cols_only
    ccd = ccdproc.cosmicray_lacosmic(ccd, gain_apply=False, gain=gain, 
                                     readnoise=read_noise, verbose=False)
    # Bug in cosmicray_lacosmic returns CCDData.data as a Quanity with incorrect
    # units electron/ADU if gain_apply=True. Therefore, we manually apply gain,
    # and leave ccd.data as a numpy array
    ccd.data = ccd.data * gain #* u.electron
    return ccd


def trim_overscan(frame):
    def nickel_oscansec(hdr):
        nc = hdr['NAXIS1']
        no = hdr['COVER']
        nr = hdr['NAXIS2']
        return f'[{nc-no+1}:{nc},1:{nr}]'
    
    # ccd = CCDData.read(frame, unit='adu')
    ccd = frame
    oscansec = nickel_oscansec(ccd.header)
    proc_ccd = ccdproc.subtract_overscan(ccd, fits_section=oscansec, overscan_axis=1)
    return ccdproc.trim_image(proc_ccd, fits_section=ccd.header['DATASEC'])


def stack_frames(raw_frames, frame_type):
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
    logger.info(f"Combining bias files into master bias")
    bias_df = file_df.copy()[file_df.objects == bias_label]
    logger.info(f"Using {len(bias_df.files)} bias frames: {[file.name.split('_')[0] for file in bias_df.paths]}")

    master_bias = stack_frames(bias_df.files, frame_type='bias')
    
    if save:
        master_bias.header["OBJECT"] = "Master_Bias"
        master_bias.write(save_dir / 'master_bias.fits', overwrite=True)
        logger.info(f"Saving master bias to {save_dir / 'master_bias.fits'}")
    
    return master_bias
    

def get_master_flats(file_df, save=True, save_dir=None):
    logger.info(f"Combining flat files into master flat")
    
    # Use sky flats if available, use dome flats if not
    if sky_flat_label in list(set(file_df.objects)):
        flattype = sky_flat_label
    else:
        flattype = dome_flat_label
    logger.debug(f"Using flat type '{flattype}'")
    
    filts = list(set(file_df.filts[file_df.objects == flattype]))
    master_flats = {}
    for filt in filts:
        flat_df = file_df.copy()[(file_df.objects == flattype) & (file_df.filts == filt)]
        logger.info(f"Using {len(flat_df.files)} flat frames: {[path.name.split('_')[0] for path in flat_df.paths]}")

        master_flat = stack_frames(flat_df.files, frame_type='flat')
        
        if save:
            master_flat.header["OBJECT"] = filt + "-Band_Master_Flat"
            master_flat.write(save_dir / ('master_flat_' + filt + '.fits'), overwrite=True)
            logger.info(f"Saving {filt}-band master flat to {save_dir / ('master_flat_' + filt + '.fits')}")
        master_flats[filt] = master_flat
    
    return master_flats


def save_results(scifile_df, modifier_str, save_dir):
    Path.mkdir(save_dir, exist_ok=True)
    logger.info(f"Saving {len(scifile_df.files)} fully reduced {save_dir.name} images to {save_dir}")
    save_paths = [save_dir / (path.name.split('_')[0] + f"_{modifier_str}" + path.suffix) for path in scifile_df.paths]
    for file, path in zip(scifile_df.files, save_paths):
        file.write(path, overwrite=True)
    return save_paths
    

# Function to normalize comparison
def norm_str(s):
    if isinstance(s, list):
        return [norm_str(elem) for elem in s]
    return s.upper().replace(' ', '')

bias_label = norm_str(bias_label)
dome_flat_label = norm_str(dome_flat_label)
sky_flat_label = norm_str(sky_flat_label)
dark_label = norm_str(dark_label)
focus_label = norm_str(focus_label)

def create_exclusion_func(exclude_list, mode='str'):
    if exclude_list is None:
        return lambda _: True
    if mode == 'str':
        exclude_list = [norm_str(obj_str) for obj_str in exclude_list]
    def exclude_func(target):
        if mode == 'path':
            target = target.name
        for excluded_str in exclude_list:
            if excluded_str in target:
                return False
        return True
    return exclude_func


# def include_file(object, mode, exclude_all):
#     if norm_str(object) == mode:
#         return True
    
#     if mode == 'BIAS':
#         pass
#     elif mode == 'DOMEFLAT':
#         pass
#     elif mode == 'SKYFLAT':
#         if norm_str(object) == 'FLAT':
#             return True
#     elif mode == 




def main():
    
    # parfile = 'rdx.toml'
    # with open(parfile, 'rb') as f:
    #     par = tomllib.load(f)

    # check_par(par)

    # bias_groups = sort(par, 'bias')
    # flat_groups = sort(par, 'flat')
    # object_groups = sort(par, 'object')

    # # Process the biases
    # for key, frames in bias_groups.items():
    #     stacked_bias = proc_bias(par, frames, save=True)

    #     embed()
    #     exit()
    return
    


# if __name__ == '__main__':
#     main()



def reduce_all(rawdir=None, file_table_in=None, file_table_out='reduction_files_table', save_inters=False, 
               excl_files=None, excl_obj_strs=None, excl_filts=None):
    # exclude = list of file stems (i.e. not .fits) from rawdir to be excluded
    # exclude by range??
    
    file_df = organize_files(rawdir, file_table_in, file_table_out, 
                             excl_files, excl_obj_strs, excl_filts)
    
    # Make directories for saving results
    if rawdir is None:
        parent_dir = file_df.paths[0].parent.parent
    else:
        parent_dir = rawdir.parent
    reddir = parent_dir / 'reduced'
    procdir = parent_dir / 'processing'
    Path.mkdir(reddir, exist_ok=True)
    Path.mkdir(procdir, exist_ok=True)
    
    # Create CCDData objects
    logger.info(f"Intializing CCDData objects & removing cosmic rays")
    ccd_objs = [init_ccddata(file) for file in file_df.files]
    file_df.files = ccd_objs
    
    # Generate master bias & master flats
    master_bias = get_master_bias(file_df, save=save_inters, save_dir=procdir)
    master_flats = get_master_flats(file_df, save=save_inters, save_dir=procdir)

    # Create new DataFrame with just object images
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

    # Perform flat division
    logger.info("Performing flat division")
    all_red_paths = []
    for filt in master_flats.keys():
        scienceobjects = list(set(scifile_df.objects[scifile_df.filts == filt]))
        
        for scienceobject in scienceobjects:
            # Take the subset of scifile_df containing scienceobject in filter filt
            sub_scifile_df = scifile_df.copy()[(scifile_df.objects == scienceobject) &
                                        (scifile_df.filts == filt)]
            # Make a new directory for each science target / filter combination
            sci_dir = reddir / (scienceobject + '_' + filt)
            
            # Do flat division
            sub_scifile_df.files = [ccdproc.flat_correct(scifile, master_flats[filt]) 
                         for scifile in sub_scifile_df.files]
            
            logger.info(f"{filt} Filter - Saving {len(sub_scifile_df.files)} fully reduced {scienceobject} images to {sci_dir}")
            red_paths = save_results(sub_scifile_df, 'red', sci_dir)
            all_red_paths += red_paths
    
    logger.info(f"Flat divided images saved to {reddir}")
    logger.info("---- reduce_all() call ended")
    return all_red_paths


def organize_files(rawdir=None, 
                   file_table_in=None, file_table_out='reduction_files_table.txt',
                   excl_files=None, excl_obj_strs=None, excl_filts=None):

    if file_table_in is not None:
        # Extract raw files (as paths) from astropy Table file
        logger.info(f"---- reduce_all() called on Astropy table file {file_table_in}")
        file_table = Table.read(file_table_in, format='ascii.fixed_width')
        rawfiles = [Path(file_path) for file_path in file_table['Path']]
        logger.info(f"{len(rawfiles)} raw files extracted from table file")
    else:
        # Extract raw files (as paths) from directory
        logger.info(f"---- reduce_all() called on directory {rawdir}")
        rawfiles = [file for file in Path(rawdir).iterdir() if (file.is_file())]
        logger.info(f"{len(rawfiles)} raw files extracted from raw directory")
    
    # Create DataFrame for files
    obj_list = []
    filt_list = []
    for file in rawfiles:
        hdul = fits.open(str(file))
        obj_list.append(norm_str(hdul[0].header["OBJECT"]))
        filt_list.append(hdul[0].header["FILTNAM"])
        hdul.close()

    file_df = pd.DataFrame({
        "names": [file.name for file in rawfiles],
        "files": None,
        "objects": obj_list,
        "filts": filt_list,
        "paths": rawfiles
        })
    
    if file_table_in is None:
        file_table = Table()
        file_table['Name'] = file_df.names
        file_table['Object'] = file_df.objects
        file_table['Filter'] = file_df.filts
        file_table['Path'] = file_df.paths
        file_table.write(file_table_out, format='ascii.fixed_width', overwrite=True)
    
    excl_file_names = []
    
    exclude_func = create_exclusion_func(excl_files, mode='path')
    file_df = file_df.copy()[file_df.paths.apply(exclude_func)]
    excl_file_names += file_df[file_df.paths.apply(lambda x: not exclude_func(x))].names
    logger.info(f"Excluded files {excl_files}")
    
    excl_obj_strs.append(focus_label)
    exclude_func = create_exclusion_func(excl_obj_strs, mode='str')
    file_df = file_df.copy()[file_df.objects.apply(exclude_func)]
    excl_file_names += file_df[file_df.paths.apply(lambda x: not exclude_func(x))].names
    logger.info(f"Excluded files with {excl_obj_strs} in the object name")
    
    exclude_func = create_exclusion_func(excl_filts, mode='str')
    file_df = file_df.copy()[file_df.filts.apply(exclude_func)]
    excl_file_names += file_df[file_df.paths.apply(lambda x: not exclude_func(x))].names
    logger.info(f"Excluded files with filters {excl_filts}")
    
    comment_out_rows(excl_file_names, file_table_out)
    
    return file_df


def comment_out_rows(excl_file_names, table_file):
    
    with open(table_file, 'r') as f:
        lines = f.readlines()
    
    lines = ['#' + line if any(name in line and not line.strip().startswith('#') 
                               for name in excl_file_names) 
             else line 
             for line in lines]
    
    with open(table_file, 'w') as f:
        f.writelines(lines)