from astropy.io import fits
import matplotlib.pyplot as plt

def print_fits_info(image_path):
    """Prints HDU List info, HDU header, and data

    Args:
        image_path (str): path to fits image (greyscale only)
    """

    with fits.open(image_path) as hdul:
        # print("HDU List Info:")
        # print(hdul.info())
        # print("\nHDU Header")
        # print(repr(hdul[0].header))
        
        # print(hdul[0].data)
        
        plt.figure(figsize=(8, 6))  # Set the figure size if needed
        plt.imshow(hdul[0].data, origin='lower')
        # Set the DPI (dots per inch) for the figure
        plt.gcf().set_dpi(300)  # Adjust the DPI value as needed

        # Display the plot
        plt.colorbar()
        plt.show()