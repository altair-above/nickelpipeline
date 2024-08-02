

import numpy as np
import logging

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Moffat2D
from photutils.detection import IRAFStarFinder
from photutils.aperture import CircularAperture
from photutils.psf import IterativePSFPhotometry, make_psf_model
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.psf import SourceGrouper

from pathlib import Path
from astropy.table import Table
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval

from scipy.spatial import KDTree

from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.nickel_data import bad_columns
from nickelpipeline.convenience.log import log_astropy_table

from nickelpipeline.photometry.moffat_model_photutils import MoffatElliptical2D
from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single, fit_psf_stack, psf_plot

from astropy.io import ascii, fits
from astropy.wcs import WCS


def convert_coords(phot_data_path, astrometric_img_path):
    phot_data_path = Path(phot_data_path)
    
    phot_data = ascii.read(phot_data_path, format='csv')
    
    # Read the input FITS file
    data, header = fits.getdata(phot_data_path, header=True)
    # Create a WCS object from the input FITS header
    wcs = WCS(header)
    
    x_coords = phot_data['x_fit']
    y_coords = phot_data['y_fit']
    # Get the array of all RA coordinates and all Dec coordinates
    world_coords = wcs.all_pix2world(x_coords, y_coords, 0)  # 0 specifies no origin offset
    longitude_coords = world_coords[0]
    latitude_coords = world_coords[1]
    
    
    

