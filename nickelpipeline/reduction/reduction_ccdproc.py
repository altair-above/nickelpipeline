
from pathlib import Path
from IPython import embed
import warnings
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.nddata import CCDData
from astropy.units.quantity import Quantity
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


def reduce_all(rawdir, save_inters=False, exclude_files=None, exclude_obj_strs=None):
    # exclude = list of file stems (i.e. not .fits) from rawdir to be excluded
    #########exlude by object name as well?
    # exclude by range
    
    if not isinstance(rawdir, Path):
        rawdir = Path(rawdir)
    logger.info(f"---- reduce_all() called on {rawdir}")
    
    # Extract raw files (as paths) from directory, eliminating excluded files
    rawfiles = [file for file in rawdir.iterdir() if (file.is_file())]
    logger.info(f"{len(rawfiles)} raw files extracted")
    
    # Make directories for saving results
    reddir = rawdir.parent / 'reduced'
    Path.mkdir(reddir, exist_ok=True)
    procdir = rawdir.parent / 'processing'
    Path.mkdir(procdir, exist_ok=True)
    
    # Create DataFrame for files
    obj_list = []
    filt_list = []
    for file in rawfiles:
        hdul = fits.open(str(file))
        obj_list.append(norm_str(hdul[0].header["OBJECT"]))
        filt_list.append(hdul[0].header["FILTNAM"])
        hdul.close()

    file_df = pd.DataFrame({
        "files": rawfiles,
        "objects": obj_list,
        "filts": filt_list,
        "paths": rawfiles
        })
    
    exclude_func = create_exclusion_func(exclude_files, mode='path')
    file_df = file_df.copy()[file_df.paths.apply(exclude_func)]
    logger.info(f"Excluded files {exclude_files}")
    
    exclude_obj_strs.append(focus_label)
    exclude_func = create_exclusion_func(exclude_obj_strs, mode='str')
    file_df = file_df.copy()[file_df.objects.apply(exclude_func)]
    logger.info(f"Excluded files with {exclude_obj_strs} in the object name")

    logger.info(f"Intializing CCDData objects & removing cosmic rays")
    # Create CCDData objects
    ccd_objs = [init_ccddata(file) for file in file_df.files]
    file_df.files = ccd_objs
    # return ccd_objs

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


def init_ccddata(frame):
    ccd = CCDData.read(frame, unit=u.adu)
    # print(ccd.unit)
    # print(ccd.data)
    # ccd.data = Quantity(ccd.data, unit='adu')
    # # print(ccd.data)
    # print(ccd.unit)
    # print(ccd.data.unit)
    # print(ccd.data)
    ccd.mask = nickel_fov_mask_cols_only
    ccd = ccdproc.cosmicray_lacosmic(ccd, gain_apply=False, gain=gain, 
                                     readnoise=read_noise, verbose=False)
    ccd.data = ccd.data * gain #* u.electron
    # print(ccd.unit)
    # print(ccd.data.unit)
    # print(ccd.data)
    # print(type(ccd.data))
    # ccd.data.unit = ccd.data
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

