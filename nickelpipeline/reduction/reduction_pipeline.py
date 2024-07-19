import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

from nickelpipeline.reduction.overscan_subtraction import overscan_subtraction
from nickelpipeline.reduction.bias_subtraction import bias_subtraction
from nickelpipeline.reduction.flat_division import flat_division
from nickelpipeline.convenience.nickel_data import sat_columns


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
    return s.lower().replace(' ', '')

bias_label = norm_str(bias_label)
dome_flat_label = norm_str(dome_flat_label)
sky_flat_label = norm_str(sky_flat_label)
dark_label = norm_str(dark_label)
focus_label = norm_str(focus_label)

def reduce_all(rawdir):
    
    if not isinstance(rawdir, Path):
        rawdir = Path(rawdir)
    
    rawfiles = [file for file in rawdir.iterdir() if file.is_file()]
    
    rootdir = rawdir.parent
    procdir = rawdir.parent / ("processing")
    reddir = rawdir.parent / ("reduced")
    Path.mkdir(procdir, exist_ok=True)
    Path.mkdir(reddir, exist_ok=True)

    print("Performing overscan subtraction")
    
    overscan_dir = procdir / 'overscan'
    Path.mkdir(overscan_dir, exist_ok=True)
    overscan_files = [overscan_dir / (file.stem + "_over" + file.suffix) for file in rawfiles]
    
    overscan_subtraction(rawfiles, overscan_files, 'yes')
    print(f"Overscan subtracted images saved to {overscan_dir}")

    obj_list = []
    exptime_list = []
    filt_list = []
    for overscan_file in overscan_files:
        hdul = fits.open(str(overscan_file))
        obj_list.append(hdul[0].header["OBJECT"])
        exptime_list.append(hdul[0].header["EXPTIME"])
        filt_list.append(hdul[0].header["FILTNAM"])
        hdul.close()

    df_log = pd.DataFrame({
        "file": overscan_files,
        "object": obj_list,
        "exptime": exptime_list,
        "filt": filt_list
        })

    print("Performing bias subtraction")
    
    # gather all the bias frames
    biasfiles = list(df_log.file[df_log.object.apply(norm_str) == bias_label])

    # average all of them into one
    biasdata = []
    for biasfile in biasfiles:
        hdul = fits.open(str(biasfile))
        biasdata.append(hdul[0].data)
        hdul.close()
    bias = np.stack(biasdata).mean(axis=0)
    
    # omit saturated columns so that they properly flat-fielded out
    # Allison Note: column is saturated to around 65,000 cts in all images; unusable (???)
    for sat_col in sat_columns:
        bias[:,sat_col] = 0

    # gather all non-bias files in a subdirectory of processing
    nonbias_dir = procdir / 'unbias'
    Path.mkdir(nonbias_dir, exist_ok=True)
    nonbias_files_input = list(df_log.file[df_log.object.apply(norm_str) != bias_label])
    nonbias_files_output = [nonbias_dir / (file.stem.split('_')[0] + "_unbias" + file.suffix) for file in nonbias_files_input]

    bias_subtraction(nonbias_files_input, nonbias_files_output, bias)
    print(f"Bias subtracted images saved to {nonbias_dir}")

    df_log["file"] = [nonbias_dir / (file.stem.split('_')[0] + "_unbias" + file.suffix) for file in df_log["file"]]
    
    print("Performing flat division")
    
    # use sky flats if available, use dome flats if not
    if sky_flat_label in list(set(norm_str(obj_list))):
        flattype = sky_flat_label
    else:
        flattype = dome_flat_label
        
    flatfilts = list(set(df_log.filt[df_log.object.apply(norm_str) == flattype]))
    
    all_red_files = []

    for flatfilt in flatfilts:
        # find all the files with this filter
        flatfiles = list(df_log.file[(df_log.object.apply(norm_str) == flattype) & (df_log.filt == flatfilt)])
        
        scienceobjects = list(set(df_log.object[(df_log.object.apply(norm_str) != bias_label) &
                                                (df_log.object.apply(norm_str) != dark_label) &
                                                (df_log.object.apply(norm_str) != dome_flat_label) &
                                                (df_log.object.apply(norm_str) != sky_flat_label) &
                                                (df_log.object.apply(norm_str) != focus_label) &
                                                (df_log.filt == flatfilt)]))
        
        # calculate the average flat frame
        if len(flatfiles) > 1:
            flatdata = []
            for flatfile in flatfiles:
                hdul = fits.open(str(flatfile))
                flatdata.append(hdul[0].data)
                hdul.close()
            flat = np.stack(flatdata).mean(axis=0)
        else:
            hdul = fits.open(str(flatfiles[0]))
            flat = hdul[0].data
            hdul.close()
            
        if len(scienceobjects) > 0:
            for scienceobject in scienceobjects:
                sciencefiles = list(df_log.file[(df_log.object.apply(norm_str) == scienceobject) &
                                                (df_log.filt == flatfilt)])
                
                # make a new directory for each science target / filter combination
                sci_dir = reddir / (scienceobject + '_' + flatfilt)
                Path.mkdir(sci_dir, exist_ok=True)
                # define reduced file names
                redfiles = [sci_dir / (file.stem.split('_')[0] + '_red' + file.suffix) for file in sciencefiles]
                all_red_files += redfiles
                
                # do flat division
                if len(sciencefiles) > 0:
                    flat_division(sciencefiles, redfiles, flat)

    print(f"Flat divided images saved to {reddir}")
    return all_red_files





