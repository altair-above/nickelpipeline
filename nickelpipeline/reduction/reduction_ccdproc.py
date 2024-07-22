
import operator
from pathlib import Path
import tomllib

from IPython import embed
import warnings

import numpy as np

from astropy.io import fits
from astropy.nddata import CCDData
import ccdproc

import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import logging.config

from nickelpipeline.reduction.overscan_subtraction import overscan_subtraction
from nickelpipeline.reduction.bias_subtraction import bias_subtraction
from nickelpipeline.reduction.flat_division import flat_division
from nickelpipeline.convenience.nickel_data import sat_columns
from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.log import default_logger

module_name = __name__.split('.')[-1]
logger = default_logger(module_name)


def trim_overscan(frame):
    def nickel_oscansec(hdr):
        nc = hdr['NAXIS1']
        no = hdr['COVER']
        nr = hdr['NAXIS2']
        return f'[{nc-no+1}:{nc},1:{nr}]'
    
    ccd = CCDData.read(frame, unit='adu')
    oscansec = nickel_oscansec(ccd.header)
    proc_ccd = ccdproc.subtract_overscan(ccd, fits_section=oscansec, overscan_axis=1)
    return ccdproc.trim_image(proc_ccd, fits_section=ccd.header['DATASEC'])


def stack_frames(raw_frames, frame_type):
    trimmed_frames = [trim_overscan(frame) for frame in raw_frames]
    # trimmed_frames = [ccdproc.cosmicray_median(frame, error_image=np.ones(frame.data.shape)*np.std(frame.data)) for frame in trimmed_frames]
    
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


def get_master_bias(files_df):


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

def reduce_all(rawdir, save_inters=False):

    
    if not isinstance(rawdir, Path):
        rawdir = Path(rawdir)
    
    logger.info(f"---- reduce_all called on {rawdir}")
    
    rawfiles = [file for file in rawdir.iterdir() if file.is_file()]
    logger.info(f"{len(rawfiles)} raw files extracted")
    
    reddir = rawdir.parent / 'reduced'
    Path.mkdir(reddir, exist_ok=True)
    
    if save_inters:
        procdir = rawdir.parent / 'processing'
        overscan_dir = procdir / 'overscan'
        unbias_dir = procdir / 'unbias'
        Path.mkdir(procdir, exist_ok=True)
        Path.mkdir(overscan_dir, exist_ok=True)
        Path.mkdir(unbias_dir, exist_ok=True)
    
    
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
        "filts": filt_list
        })

    logger.info(f"Combining bias files into master bias")
    bias_files = file_df.files[file_df.objects == bias_label]
    logger.info(f"Using {len(bias_files)} bias frames: {[file.name.split('_')[0] for file in bias_files]}")

    master_bias = stack_frames(bias_files, frame_type='bias')
    
    if save_inters:
        master_bias.header["OBJECT"] = "Master_Bias"
        master_bias.write(procdir / 'master_bias.fits', overwrite=True)
        logger.info(f"Saving master bias to {procdir / 'master_bias.fits'}")
    
    logger.info(f"Combining flat files into master flat")
    # use sky flats if available, use dome flats if not
    if sky_flat_label in list(set(obj_list)):
        flattype = sky_flat_label
    else:
        flattype = dome_flat_label
    logger.debug(f"Using flat type '{flattype}'")
    
    filts = list(set(file_df.filts[file_df.objects == flattype]))
    master_flats = {}
    for filt in filts:
        flat_files = list(file_df.files[(file_df.objects == flattype) & (file_df.filts == filt)])
        logger.info(f"Using {len(flat_files)} flat frames: {[path.name.split('_')[0] for path in flat_files]}")

        master_flat = stack_frames(flat_files, frame_type='flat')
        
        if save_inters:
            master_flat.header["OBJECT"] = filt + "-Band_Master_Flat"
            master_flat.write(procdir / ('master_flat_' + filt + '.fits'), overwrite=True)
            print(f"Saving {filt}-band master flat to {procdir / ('master_flat_' + filt + '.fits')}")
        master_flats[filt] = master_flat

    scifiles_mask = ((file_df.objects != bias_label) &
                    (file_df.objects != dark_label) &
                    (file_df.objects != dome_flat_label) &
                    (file_df.objects != sky_flat_label) &
                    (file_df.objects != focus_label)).values
    # return scifiles_mask
    scifile_df = pd.DataFrame({
        "files": rawfiles,
        "objects": obj_list,
        "filts": filt_list,
        "paths": rawfiles
        })
    scifile_df = scifile_df[scifiles_mask]


    logger.info(f"Performing overscan subtraction & trimming on {len(scifile_df.files)} science images")
    
    overscan_files = [trim_overscan(scifile) for scifile in scifile_df.files]
    
    if save_inters:
        overscan_paths = [overscan_dir / (path.stem + "_over" + path.suffix) for path in scifile_df.paths]
        for file, path in zip(overscan_files, overscan_paths):
            file.write(path, overwrite=True)
    
    scifile_df.files = overscan_files
            
    logger.info(f"Performing bias subtraction on {len(scifile_df.files)} science images")
    
    unbias_files = [ccdproc.subtract_bias(scifile, master_bias) 
                      for scifile in scifile_df.files]
    
    if save_inters:
        unbias_paths = [unbias_dir / (paths.stem.split('_')[0] + "_unbias" + paths.suffix) for paths in scifile_df.paths]
        for file, path in zip(unbias_files, unbias_paths):
            file.write(path, overwrite=True)
    scifile_df.files = unbias_files


    logger.info("Performing flat division")
    
    all_red_paths = []
    
    for filt in filts:
        scienceobjects = list(set(scifile_df.objects[scifile_df.filts == filt]))
        
        # if len(scienceobjects) == 0:
        #     continue
        for scienceobject in scienceobjects:
            sci_files = list(scifile_df.files[(scifile_df.objects == scienceobject) &
                                            (scifile_df.filts == filt)])
            sci_paths = list(scifile_df.paths[(scifile_df.objects == scienceobject) &
                                            (scifile_df.filts == filt)])
            
            # make a new directory for each science target / filter combination
            sci_dir = reddir / (scienceobject + '_' + filt)
            Path.mkdir(sci_dir, exist_ok=True)
            # define reduced file names
            red_paths = [sci_dir / (file.stem.split('_')[0] + '_red' + file.suffix) for file in sci_paths]
            all_red_paths += red_paths
            
            # do flat division
            red_files = [ccdproc.flat_correct(scifile, master_flats[filt]) 
                         for scifile in sci_files]
            
            logger.info(f"{filt} Filter - Saving {len(red_files)} fully reduced {scienceobject} images to {sci_dir}")
            for file, path in zip(red_files, red_paths):
                file.write(path, overwrite=True)
    
    logger.info(f"Flat divided images saved to {reddir}")
    logger.info("---- reduce_all call ended")
    return all_red_paths
    
    
    
    
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

