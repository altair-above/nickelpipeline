import numpy as np
from pathlib import Path
import warnings
from matplotlib import pyplot as plt
from collections import OrderedDict
from scipy import stats

from nickelpipeline.psf_analysis.gaussian.calc_fwhm import batch_fwhm
from nickelpipeline.convenience.dir_nav import categories_from_conditions, unzip_directories
from nickelpipeline.convenience.nickel_data import plate_scale_approx   # For the Nickel Telescope original science camera


def graph_fwhms_by_image(path_list, date=None, plot=False, max_std=0.5):
    avg, data = batch_fwhm(path_list, plot=plot, max_std=max_std)
    data.sort()
    image_numbers, fwhms, stds, objects = zip(*data)
    unique_objects = list(OrderedDict.fromkeys(objects))
    if date is None:
        print("Date in title may be inaccurate")
        date = path_list[0].parent.parent.name[-5:]

    # Plot the values relative to image numbers, with different colors for different objects
    plt.figure(figsize=(10, 6))
    plt.plot(image_numbers, fwhms, marker='o', linestyle='-', color='#808080', zorder=1, label='FWHM')
    plt.plot(image_numbers, stds, marker='s', linestyle='-', color='#B3B3B3', zorder=1, label='STD')
    for obj in unique_objects:
        obj_data = [(img_num, fwhm, std) for img_num, fwhm, std, obj_name in data if obj_name == obj]
        obj_image_numbers, obj_fwhms, obj_stds = zip(*obj_data)
        # plt.scatter(obj_image_numbers, obj_stds, marker='s', label=obj, zorder=2)
        plt.scatter(obj_image_numbers, obj_fwhms, marker='o', label=obj, zorder=2)  # Add lines and points
    
    plt.xlabel('Image Number')
    plt.ylabel('FWHM Value')
    plt.title(f'FWHM Value vs. Image Number - {date}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    min_num = min(image_numbers)
    max_num = max(image_numbers)
    plt.xticks(range(min_num, max_num + 1, max(1, int((max_num-min_num)/20))))
    plt.show()

    return avg


def graph_fwhms_by_setting(path_list, condition_tuples):
    
    images = unzip_directories(path_list, output_format='Fits_Simple')
    
    categories = categories_from_conditions(condition_tuples, images)
    
    data = []
    conf_intervals = []
    # Print out the categories and their corresponding file lists
    for category, file_list in categories.items():
        mean_fwhm, result_matrix = batch_fwhm(file_list, mode='fwhms, std')
        data.append((category, mean_fwhm))

        image_numbers, fwhms, stds, objects = zip(*result_matrix)
        fwhms = np.hstack(fwhms).tolist()
        
        # Normal / Gaussian function confidence intervals
        confidence_level = 0.95 # Confidence level
        interval = stats.norm.interval(confidence=confidence_level, 
                                       loc=np.mean(fwhms), 
                                       scale=stats.sem(fwhms))
        conf_intervals.append((abs((interval[0] - mean_fwhm)), 
                               abs(interval[1] - mean_fwhm)))
    
    data.sort()
    widths, fwhms = zip(*data)
    conf_intervals = np.array(conf_intervals).T
    
    # Plot the plate_scales relative to widths, with different colors for different objects
    plt.figure(figsize=(8, 5))
    plt.errorbar(widths, fwhms, yerr=[conf_intervals[0], conf_intervals[1]], 
                 fmt='o', linestyle='-', ecolor='r', capsize=5, 
                 label=f'FWHM w/ {confidence_level*100}% Conf. Interval')
    plt.xlabel('Spacer Width (in)')
    plt.ylabel('FWHM (pixels)')
    plt.title(f'FWHM vs. Spacer Width')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    return data

def multi_date_graph_fwhms_by_setting(path_dict, condition_tuples_dict):
    # directories_dict is of the form {'06-26': directories (list), '06-24': directories (list)}
    # Same for condition_tuples_dict
    data = []
    conf_intervals = []
    
    plt.figure(figsize=(8, 5))
    
    for date, path_list in path_dict.items():
        images = unzip_directories(path_list, output_format='Fits_Simple')
        categories = categories_from_conditions(condition_tuples_dict[date], images)
        
        subdata = []
        # Print out the categories and their corresponding file lists
        for category, file_list in categories.items():
            mean_fwhm, result_matrix = batch_fwhm(file_list, mode='fwhms, std')
            data.append((category, mean_fwhm))
            subdata.append((category, mean_fwhm))
            image_numbers, fwhms, stds, objects = zip(*result_matrix)
            fwhms = np.hstack(fwhms).tolist()
            std_dev = np.std(fwhms, ddof=1) # Sample standard deviation
            
            confidence_level = 0.95 # Confidence level
            
            # Normal / Gaussian function confidence intervals
            interval = stats.norm.interval(confidence=0.95, 
                                        loc=np.mean(fwhms), 
                                        scale=stats.sem(fwhms))
            conf_intervals.append((abs((interval[0] - mean_fwhm)), 
                                abs(interval[1] - mean_fwhm)))
        
        subdata.sort()
        widths, fwhms = zip(*subdata)
        # fwhms = fwhms * plate_scale_approx
        plt.scatter(widths, fwhms, marker='o', label=date, zorder=2)  # Add points color-coded by date
    
    data.sort()
    widths, fwhms = zip(*data)
    # fwhms = fwhms * plate_scale_approx
    conf_intervals = np.array(conf_intervals).T
    
    # Plot the plate_scales relative to widths, with different colors for different objects
    plt.errorbar(widths, fwhms, yerr=[conf_intervals[0], conf_intervals[1]], 
                 fmt='o', color='g', ecolor='r', capsize=5, zorder=1,
                 label=f'FWHM w/ {confidence_level*100}% Conf. Interval')
    plt.xlabel('Spacer Width (in)')
    plt.ylabel('FWHM (pixels)')
    plt.title(f'FWHM vs. Spacer Width')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    return data


