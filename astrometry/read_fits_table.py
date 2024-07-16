from astropy.io import fits

# Open the FITS file
file_name = "C:/Users/allis/Downloads/corr.fits"  # Replace with your FITS file name

hdul = fits.open(file_name)

# Print the FITS file information
hdul.info()

# Assuming the table is in the first extension
data = hdul[1].data

# Print the column names
print(data.columns.names)

# Print the data
print(data)

data['field_x']

# Close the FITS file
hdul.close()
