from pathlib import Path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_histo(bin_edges, counts):
    """Uses matplotlib to graph a histogram based on output from numpy np.histogram()"""
    
    plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.yscale('log')
    # Add titles and labels
    plt.title('Histogram with Counts (Logarithmic Scale)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()
    
def analyze_fits(image, bin_width=50):
    
    with fits.open(image) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        
        # number of pixels in image
        xsize = header['NAXIS1']
        # number of overscan columns
        cover = header['COVER']
        # Remove overscan portion
        data = data[0:,0:xsize-cover]
        
        # Calculate number of histogram bins, based on range of data
        minval = np.min(data)
        maxval = np.max(data)
        num_bins = int((maxval-minval)/bin_width) + 1
        
    # Create a list to store the values in each bin
    bins = [[] for _ in range(num_bins)]
    # Assign each value to the corresponding bin
    for row in data:
        for value in row:
            bin_index = int(np.floor((value - minval) / bin_width))
            bins[bin_index].append(value)
            
    means = []
    sds = []
    for bin_index in range(len(bins)):
        if len(bins[bin_index]) > 20:
            sd = np.std(bins[bin_index])
            if sd != 0:
                means.append(np.mean(bins[bin_index]))
                sds.append(sd)
    
    plot_log_linear_fit(means, sds, image, header, (0,4))
    return (means, sds, image, header, (0,4))

def plot_means_sds(means, sds, image, header):
    # Plot the scatter plot
    plt.scatter(means, sds)
    plt.xscale('log')
    plt.yscale('log')
    # Add titles and labels
    plt.title(f'SD vs. Mean Plot for {image.name} ({header["OBJECT"]})')
    plt.xlabel('Mean of all values in a bin')
    plt.ylabel('Standard Deviation of all values in a bin')
    # Show the plot
    plt.show()

def plot_log_linear_fit(means, sds, image, header, range):
    # Create a line of best fit in the logarithmic graph
    meanlogs = [math.log(mean) for mean in means]
    sdlogs = [math.log(sd) for sd in sds]
    
    meanlogs_linear = meanlogs[range[0]:range[1]]
    sdlogs_linear = sdlogs[range[0]:range[1]]

    # Calculate the line of best fit
    coefficients = np.polyfit(meanlogs_linear, sdlogs_linear, 1)  # 1 means linear fit (degree 1)
    polynomial = np.poly1d(coefficients)
    line_of_best_fit = polynomial(meanlogs_linear)

    # Print line of best fit equation
    print(f"log(SD) = {coefficients[0]:.2f} log(mean) + {coefficients[1]:.2f}")

    # Plot the scatter plot
    plt.scatter(meanlogs_linear, sdlogs_linear)
    # Plot the line of best fit
    plt.plot(meanlogs_linear, line_of_best_fit, color='red', label='Line of Best Fit')
    # Add titles and labels
    plt.title(f'Log SD vs. Log of Mean Plot for {image.name} ({header["OBJECT"]})')
    plt.xlabel('Log of Mean of all values in a bin')
    plt.ylabel('Log of Standard Deviation of all values in a bin')
    # Show the plot
    plt.show()

    
def analyze_fits_complete(image, bin_width=50):
    
    with fits.open(image) as hdu:
        header = hdu[0].header
        data = hdu[0].data
        
        # number of pixels in image
        xsize = header['NAXIS1']
        # number of overscan columns
        cover = header['COVER']
        # Remove overscan portion
        data = data[0:,0:xsize-cover]
        
        # Calculate number of histogram bins, based on range of data
        minval = np.min(data)
        maxval = np.max(data)
        num_bins = int((maxval-minval)/bin_width) + 1
        
        # Create the histogram -- counts = frequency within each bin, bin_edges = values at bin boundaries
        counts, bin_edges = np.histogram(data, bins=num_bins, )
        
        # Plot histogram using matplotlib
        plot_histo(bin_edges, counts)
        
    # Create a list to store the values in each bin
    bins = [[] for _ in range(num_bins)]
    # Assign each value to the corresponding bin
    for row in data:
        for value in row:
            bin_index = int(np.floor((value - minval) / bin_width))
            bins[bin_index].append(value)
            
    bin_to_analyze = np.argmax(counts)    # int(num_bins/2)
    def analyze_bin(bin_index, plot=False):
        if plot:
            # Create the histogram -- counts = frequency within each bin, bin_edges = values at bin boundaries
            subcounts, subbin_edges = np.histogram(bins[bin_index], bins=int(bin_width/2), )
            # Plot histogram using matplotlib
            plot_histo(subbin_edges, subcounts)
        
        mean = np.mean(bins[bin_index])
        sd = np.std(bins[bin_index])
        return [mean, sd]

    analyze_bin(bin_to_analyze, plot=True)
            
    means = []
    sds = []
    for bin_index in range(len(bins)):
        if len(bins[bin_index]) > 20:
            mean, sd = analyze_bin(bin_index)
            if sd != 0:
                means.append(mean)
                sds.append(sd)

    # Plot the scatter plot
    plt.scatter(means, sds)
    plt.xscale('log')
    plt.yscale('log')

    # Add titles and labels
    plt.title(f'SD vs. Mean Plot for {image.name} ({header["OBJECT"]})')
    plt.xlabel('Mean of all values in a bin')
    plt.ylabel('Standard Deviation of all values in a bin')

    # Show the plot
    plt.show()
    
    meanlogs = [math.log(mean) for mean in means]
    sdlogs = [math.log(sd) for sd in sds]

    meanlogs_linear = meanlogs[:4]
    sdlogs_linear = sdlogs[:4]

    # Calculate the line of best fit
    coefficients = np.polyfit(meanlogs_linear, sdlogs_linear, 1)  # 1 means linear fit (degree 1)
    polynomial = np.poly1d(coefficients)
    line_of_best_fit = polynomial(meanlogs_linear)

    print(f"log(SD) = {coefficients[0]:.2f} log(mean) + {coefficients[1]:.2f}")

    # Plot the scatter plot
    plt.scatter(meanlogs_linear, sdlogs_linear)
    # Plot the line of best fit
    plt.plot(meanlogs_linear, line_of_best_fit, color='red', label='Line of Best Fit')

    # Add titles and labels
    plt.title(f'Log SD vs. Log of Mean Plot for {image.name} ({header["OBJECT"]})')
    plt.xlabel('Log of Mean of all values in a bin')
    plt.ylabel('Log of Standard Deviation of all values in a bin')

    # Show the plot
    plt.show()


image = Path("C:/Users/allis/Documents/2024-2025_Local/Akamai Internship/pipeline-testing/test-data-05-12/raw/d1067.fits")  # Linear + plateau

analyze_fits(image, bin_width=100)
# analyze_fits_complete(image, bin_width=100)

