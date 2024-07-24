import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
import ccdproc

# Intialize CCDData object
data = np.array([[1,2,3,4],
                 [2,3,5,7],
                 [1,4,9,16],
                 [1,3,6,10]])
hdu = fits.PrimaryHDU(data=data)
hdul = fits.HDUList([hdu])
hdul.writeto('new1.fits', overwrite=True)
ccd_orig1 = CCDData.read('new1.fits', unit=u.adu)
ccd_orig2 = CCDData.read('new1.fits', unit=u.adu)

# Apply cosmic ray detection with gain_apply=False
ccd = ccdproc.cosmicray_lacosmic(ccd_orig1, gain_apply=False, gain=1.8, readnoise=10.7)
print(f"type of ccd.data = {type(ccd.data)}")
print(ccd.data)

# This does not throw an error, because ccd.data is a np.array
ccd_subtracted = ccdproc.subtract_overscan(ccd, overscan=ccd[:, 0:1])
print('\n')

#-------------------------------------------

# Apply cosmic ray detection with gain_apply=True
ccd = ccdproc.cosmicray_lacosmic(ccd_orig2, gain_apply=True, gain=1.8, readnoise=10.7)

# ccd.data is now a Quantity w/ incorrect units instead of np.array
print(f"ccd.unit = {ccd.unit} -- this is expected")
print(ccd.data)
print(f"unit of ccd.data = {ccd.data.unit}")
print(f"type of ccd.data = {type(ccd.data)}\n")

# This will throw an error because the unit electron/adu is incorrect
# (Specifically I'm guessing that it can tell when ccd.unit != ccd.data.unit?)
ccd_subtracted = ccdproc.subtract_overscan(ccd, overscan=ccd[:, 0:1])

# Change ccd.data to the correct unit (electron)
ccd.data = ccd.data * u.adu
print(ccd.data)

# This also throws an error--tries to subtract a np.array from a Quantity
ccd_subtracted = ccdproc.subtract_overscan(ccd, overscan=ccd[:, 0:1])
