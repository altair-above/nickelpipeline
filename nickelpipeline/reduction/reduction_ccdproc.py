
from pathlib import Path
from IPython import embed
import warnings
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.nddata import CCDData
import ccdproc

from nickelpipeline.convenience.nickel_data import sat_columns
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

def reduce_all(rawdir, save_inters=False):
    
    if not isinstance(rawdir, Path):
        rawdir = Path(rawdir)
    logger.info(f"---- reduce_all called on {rawdir}")
    
    rawfiles = [file for file in rawdir.iterdir() if file.is_file()]
    logger.info(f"{len(rawfiles)} raw files extracted")
    
    reddir = rawdir.parent / 'reduced'
    Path.mkdir(reddir, exist_ok=True)
    procdir = rawdir.parent / 'processing'
    Path.mkdir(procdir, exist_ok=True)
    
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

    master_bias = get_master_bias(file_df, save=save_inters, save_dir=procdir)

    master_flats = get_master_flats(file_df, save=save_inters, save_dir=procdir)

    scifiles_mask = ((file_df.objects != bias_label) &
                    (file_df.objects != dark_label) &
                    (file_df.objects != dome_flat_label) &
                    (file_df.objects != sky_flat_label) &
                    (file_df.objects != focus_label)).values
    scifile_df = file_df.copy()[scifiles_mask]

    logger.info(f"Performing overscan subtraction & trimming on {len(scifile_df.files)} science images")
    overscan_files = [trim_overscan(scifile) for scifile in scifile_df.files]
    if save_inters:
        save_results(overscan_files, scifile_df, 'over', procdir/'overscan')
    scifile_df.files = overscan_files
    
    logger.info(f"Performing bias subtraction on {len(scifile_df.files)} science images")
    unbias_files = [ccdproc.subtract_bias(scifile, master_bias) 
                      for scifile in scifile_df.files]
    if save_inters:
        save_results(unbias_files, scifile_df, 'unbias', procdir/'unbias')
    scifile_df.files = unbias_files

    logger.info("Performing flat division")
    all_red_paths = []
    for filt in master_flats.keys():
        scienceobjects = list(set(scifile_df.objects[scifile_df.filts == filt]))
        
        for scienceobject in scienceobjects:
            # Take the subset of scifile_df containing scienceobject in filter filt
            sub_scifile_df = scifile_df[(scifile_df.objects == scienceobject) &
                                        (scifile_df.filts == filt)]
            # Make a new directory for each science target / filter combination
            sci_dir = reddir / (scienceobject + '_' + filt)
            
            # Do flat division
            red_files = [ccdproc.flat_correct(scifile, master_flats[filt]) 
                         for scifile in sub_scifile_df.files]
            
            logger.info(f"{filt} Filter - Saving {len(red_files)} fully reduced {scienceobject} images to {sci_dir}")
            red_paths = save_results(red_files, sub_scifile_df, 'red', sci_dir)
            all_red_paths += red_paths
    
    logger.info(f"Flat divided images saved to {reddir}")
    logger.info("---- reduce_all call ended")
    return all_red_paths


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


def get_master_bias(file_df, save=True, save_dir=None):
    logger.info(f"Combining bias files into master bias")
    bias_files = file_df.files[file_df.objects == bias_label]
    logger.info(f"Using {len(bias_files)} bias frames: {[file.name.split('_')[0] for file in bias_files]}")

    master_bias = stack_frames(bias_files, frame_type='bias')
    
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
        flat_files = list(file_df.files[(file_df.objects == flattype) & (file_df.filts == filt)])
        logger.info(f"Using {len(flat_files)} flat frames: {[path.name.split('_')[0] for path in flat_files]}")

        master_flat = stack_frames(flat_files, frame_type='flat')
        
        if save:
            master_flat.header["OBJECT"] = filt + "-Band_Master_Flat"
            master_flat.write(save_dir / ('master_flat_' + filt + '.fits'), overwrite=True)
            logger.info(f"Saving {filt}-band master flat to {save_dir / ('master_flat_' + filt + '.fits')}")
        master_flats[filt] = master_flat
    
    return master_flats


def save_results(files, scifile_df, modifier_str, save_dir):
    Path.mkdir(save_dir, exist_ok=True)
    save_paths = [save_dir / (path.stem.split('_')[0] + f"_{modifier_str}" + path.suffix) for path in scifile_df.paths]
    for file, path in zip(files, save_paths):
        file.write(path, overwrite=True)
    return save_paths
    
    
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

