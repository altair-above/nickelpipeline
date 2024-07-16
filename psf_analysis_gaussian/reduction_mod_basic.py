import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

from reduction_files.overscan_subtraction import overscan_subtraction
from reduction_files.bias_subtraction import bias_subtraction
from reduction_files.flat_division import flat_division


rawdir = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw")  # path to directory with raw data

# Set object type names
bias_object = 'Bias'
dome_flat_object = 'Dome flat'
sky_flat_object = 'Flat'
dark_object = 'dark'
focus_object = 'focus'

def generate_reduction_files(rawdir):

    rawfiles = [file for file in rawdir.iterdir() if file.is_file()]

    procdir = rawdir.parent / (rawdir.name + "-proc")
    reddir = rawdir.parent / (rawdir.name + "-reduced")

    Path.mkdir(procdir, exist_ok=True)
    Path.mkdir(reddir, exist_ok=True)

    overscan_dir = procdir / 'overscan'
    Path.mkdir(overscan_dir, exist_ok=True)
    overscan_files = [overscan_dir / (file.stem + file.suffix) for file in rawfiles]

    overscan_subtraction(rawfiles, overscan_files, 'yes')

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

    # gather all the bias frames
    biasfiles = list(df_log.file[df_log.object == bias_object])

    # average all of them into one
    biasdata = []
    for biasfile in biasfiles:
        hdul = fits.open(str(biasfile))
        biasdata.append(hdul[0].data)
        hdul.close()
    bias = np.stack(biasdata).mean(axis=0)

    # omit hot column so that it is properly flat-fielded out
    # Allison Note: column is saturated to around 65,000 cts in all images; unusable (???)
    bias[:,256] = 0

    # gather all non-bias files in a subdirectory of -proc
    nonbias_dir = procdir / 'unbias'
    print(nonbias_dir)
    Path.mkdir(nonbias_dir, exist_ok=True)

    nonbias_files_input = list(df_log.file[df_log.object != bias_object])
    nonbias_files_output = [nonbias_dir / (file.stem.split('_')[0] + file.suffix) for file in nonbias_files_input]

    bias_subtraction(nonbias_files_input, nonbias_files_output, bias)

    df_log["file"] = [nonbias_dir / (file.stem.split('_')[0] + file.suffix) for file in df_log["file"]]

    # use sky flats if available, use dome flats if not
    if sky_flat_object in list(set(obj_list)):
        flattype = sky_flat_object
    else:
        flattype = dome_flat_object
        
    flatfilts = list(set(df_log.filt[df_log.object == flattype]))
    flats = {}
    # flats['flattype'] = flattype
    
    for flatfilt in flatfilts:
        # find all the files with this filter
        flatfiles = list(df_log.file[(df_log.object == flattype) & (df_log.filt == flatfilt)])
        scienceobjects = list(set(df_log.object[(df_log.object != bias_object) &
                                                (df_log.object != dark_object) &
                                                (df_log.object != dome_flat_object) &    # modified from 'dome flat' to 'flat' for 05-12-24 data
                                                (df_log.object != sky_flat_object) &
                                                (df_log.object != focus_object) &
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
            hdul = fits.open(str(flatfile))
            flat = hdul[0].data
            hdul.close()
        
        flats[flatfilt] = flat
        
        if len(scienceobjects) > 0:
            for scienceobject in scienceobjects:
                sciencefiles = list(df_log.file[(df_log.object == scienceobject) &
                                                (df_log.filt == flatfilt)])
                
                # make a new directory for each science target / filter combination
                sci_dir = reddir / (scienceobject + '_' + flatfilt)
                Path.mkdir(sci_dir, exist_ok=True)

                # define reduced file names
                redfiles = [sci_dir / (file.stem.split('_')[0] + '_red' + file.suffix) for file in sciencefiles]

                # do flat division
                if len(sciencefiles) > 0:
                    flat_division(sciencefiles, redfiles, flat)
    
    return bias, flats

def process_single(image, bias, flats):
    
    rawfiles = [Path(image),]

    procdir = rawdir.parent / (rawdir.name + "-proc")
    reddir = rawdir.parent / (rawdir.name + "-reduced")
    Path.mkdir(procdir, exist_ok=True)
    Path.mkdir(reddir, exist_ok=True)

    overscan_dir = procdir / 'overscan'
    Path.mkdir(overscan_dir, exist_ok=True)
    overscan_files = [overscan_dir / (file.stem + file.suffix) for file in rawfiles]
    
    overscan_subtraction(rawfiles, overscan_files, 'yes')

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

    # gather all non-bias files in a subdirectory of -proc
    nonbias_dir = procdir / 'unbias'
    Path.mkdir(nonbias_dir, exist_ok=True)
    nonbias_files_input = list(df_log.file[df_log.object != bias_object])
    nonbias_files_output = [nonbias_dir / (file.stem.split('_')[0] + file.suffix) for file in nonbias_files_input]

    bias_subtraction(nonbias_files_input, nonbias_files_output, bias)

    df_log["file"] = [nonbias_dir / (file.stem.split('_')[0] + file.suffix) for file in df_log["file"]]

    # flattype = flats['flattype']
    # del flats['flattype']
    flatfilts = flats.keys()
    
    all_red_files = []

    for flatfilt in flatfilts:
        # find all the files with this filter
        scienceobjects = list(set(df_log.object[(df_log.object != bias_object) &
                                                (df_log.object != dark_object) &
                                                (df_log.object != dome_flat_object) &
                                                (df_log.object != sky_flat_object) &
                                                (df_log.object != focus_object) &
                                                (df_log.filt == flatfilt)]))
            
        if len(scienceobjects) > 0:
            for scienceobject in scienceobjects:
                sciencefiles = list(df_log.file[(df_log.object == scienceobject) &
                                                (df_log.filt == flatfilt)])
                
                # make a new directory for each science target / filter combination
                sci_dir = reddir / (scienceobject + '_' + flatfilt)
                Path.mkdir(sci_dir, exist_ok=True)
                # define reduced file names
                redfiles = [sci_dir / (file.stem.split('_')[0] + '_red' + file.suffix) for file in sciencefiles]
                all_red_files += redfiles
                
                # do flat division
                if len(sciencefiles) > 0:
                    flat_division(sciencefiles, redfiles, flats[flatfilt])

    return all_red_files[0]



def process_multiple(images, bias, flats):
    if not isinstance(images):
        rawfiles = [Path(images),]
    else:
        rawfiles = [Path(image) for image in images]
    
    red_files = []
    for image in images:
        red_files.append(process_single(image, bias, flats))

    return red_files
    





