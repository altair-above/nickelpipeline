# API ACTUAL
import requests
import json
import time
import os
from astropy.io import fits

####################################
#### ASTROMETRY.NET CALIBRATION ####
####################################


# image = 'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/PG1530+057_V/d1035_red.fits'
# output_dir = 'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-astrometric'

def run_astrometry(image_paths, output_dir, mode='image', fast=False):
    """Runs astrometry.net calibration on all images input, and saves the calibrated
    fits files to output_dir. Uses astrometry.net's API.

    Args:
        image_paths (list): list of relative paths (str) to all images
        output_dir (str): path to output image folder

    Returns:
        list: list of relative paths (str) to all calibrated fits images
    """
    if not isinstance(image_paths, list):
        image_paths = [str(image_paths)]
    else:
        image_paths = [str(path) for path in image_paths]
    
    # If running in fast=True, skip all astrometry.net and only access already-
    # processed images
    if fast:
        calibrated_fits_paths = []
        for image_path in image_paths:
            output_path_stem = os.path.basename(image_path)
            if mode == 'image':
                output_path = os.path.join(output_dir, output_path_stem[:-5] + '.fits')
            elif mode == 'corr':
                output_path = os.path.join(output_dir, output_path_stem[:-5] + '_corr' + '.fits')
            if os.path.exists(output_path):
                calibrated_fits_paths.append(output_path)
        return calibrated_fits_paths
    
    
    
    
    # Makes output folder if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Modify images to remove the Nickel Telescope's bad columns
    mod_dir = os.path.join(os.path.dirname(output_dir), 'raw-astro-input')   # Directory for modified images
    if not os.path.exists(mod_dir):
        os.makedirs(mod_dir)
    columns_to_modify = [255, 256, 783, 784, 1002]

    mod_paths = []
    for file in image_paths:
        with fits.open(file) as hdul:
            data = hdul[0].data
            for col in columns_to_modify:
                data[:, col] = 0  # Set the entire column to 0
            # Save the modified FITS file to the output directory
            mod_path = os.path.join(mod_dir, os.path.basename(file))
            mod_paths.append(mod_path)
            hdul.writeto(mod_path, overwrite=True)
    
    print("Connecting to astrometry.net")
    
    # Log in to Nova and get session key
    R = requests.post(
        "http://nova.astrometry.net/api/login",
        data={"request-json": json.dumps({"apikey": "nhvozkuehegpwybk"})},  # API key can be found in your account -- currently using Allison's
    )
    dictionary = json.loads(R.text)
    session_key = dictionary["session"]
    
    # Wrapper function to convert a single photo (to be used on all photos below)
    def convert(image_path):
        # Use session key to upload file(s) to Astrometry
        url = "http://nova.astrometry.net/api/upload"
        files = {
            "file": (os.path.basename(image_path), open(image_path, "rb")),
        }
        data = {
            "request-json": '{"publicly_visible": "y", "allow_modifications": "d", "session": "' + session_key + '", "allow_commercial_use": "d"}' #, "scale_type":"ul", "scale_lower":0.05, "scale_upper":5}'
        }
        R = requests.post(url, files=files, data=data)
        dictionary = json.loads(R.text)
        subid = dictionary['subid']

        # Get jobid from submission status
        # Requires checking every 5 sec & waiting until requests.post actually returns something
        # Otherwise an error will be thrown on some photos
        jobid_found = False
        while jobid_found is False:
            print("Waiting to find Job ID")
            time.sleep(5)

            url = f"http://nova.astrometry.net/api/submissions/{subid}"
            R = requests.post(url)
            dictionary = json.loads(R.text)
            try:
                jobid = dictionary["jobs"][0]
                if jobid is None:
                    continue
                jobid_found = True
            except:
                continue
                
        print(f"JobID = {jobid}")

        # Every t (currently 15) seconds, check if image has been calibrated
        # Exit & fail if taking > 10 minutes to calibrate
        print(f"Time elapsed = {0} seconds")
        url = f"http://nova.astrometry.net/api/jobs/{jobid}"
        def time_check(t, interval):
            time.sleep(interval)
            t += interval
            print(f"Time elapsed = {t} seconds")
            R = requests.post(url)
            data = json.loads(R.text)
            if data['status'] == 'success':
                print('Job succeeded, getting calibrated fits file')
                return True
            if data['status'] == 'failed':
                print('Job failed')
                return False
            if t > 30:
                print('Maximum time elapsed... exiting')
                return False
            time_check(t, interval)
        
        success = time_check(0, 15)
        
        if success:
            # Create path to image folder
            output_path_stem = os.path.basename(image_path)
            
            # Get FITS file & save to output path (in folder just created ^)
            url_image = f"http://nova.astrometry.net/new_fits_file/{jobid}"
            url_corr = f"http://nova.astrometry.net/corr_file/{jobid}"
            output_path_img = os.path.join(output_dir, output_path_stem[:-5] + '.fits')
            output_path_corr = os.path.join(output_dir, output_path_stem[:-5] + '_corr' + '.fits')
            
            new_image = requests.get(url_image)
            corr_table = requests.get(url_corr)
            
            with open(output_path_img, 'wb') as img:
                img.write(new_image.content)
            with open(output_path_corr, 'wb') as corr:
                corr.write(corr_table.content)
            
            print(f"Calibrated image & corr saved to {url_image} (or to _corr.fits)")
            
            if mode == 'image':
                return output_path_img
            elif mode == 'corr':
                return output_path_corr
            
            
            # Return output_path for other functions to find this file
            return output_path
        else:
            return None
    
    # Calibrate each image piece and collect output_paths
    calibrated_fits_paths = []
    for image_path in mod_paths:
        output_path_stem = os.path.basename(image_path)
        if mode == 'image':
            output_path = os.path.join(output_dir, output_path_stem[:-5] + '.fits')
        elif mode == 'corr':
            output_path = os.path.join(output_dir, output_path_stem[:-5] + '_corr' + '.fits')
        if os.path.exists(output_path):
            calibrated_fits_paths.append(output_path)
        else:
            try:
                calib_fits = convert(image_path)
                if calib_fits is not None:
                    calibrated_fits_paths.append(calib_fits)
            except:
                print(f"***Connection Error(?) encountered; skipping this image***")
                time.sleep(15)
        os.remove(image_path)
    return calibrated_fits_paths

# image2 = run_astrometry('C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-reduced/PG1530+057_V/d1035_red.fits', 'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-24/raw-astrometric')[0]

# For if run_astrometry saves bad/un-processed files

# astrdir = Path(f'C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-06-26/astrometric')
# for image in astrdir.iterdir():
#     try:
#         with fits.open(image) as hdul:
#             hdu = hdul[0]
#     except OSError:
#         print(f"deleting {image}")
#         image.unlink()