import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from loess.loess_2d import loess_2d

from photutils.aperture import CircularAperture
from astropy.visualization import ZScaleInterval

from nickelpipeline.convenience.nickel_data import ccd_shape
from nickelpipeline.convenience.fits_class import Fits_Simple

logger = logging.getLogger(__name__)

def smooth_contour(data_x, data_y, data_vals, color_range, backgrd_ax=None, 
                   frac=0.3, title=None, category_str=None):
    
    if backgrd_ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
    else:
        ax = backgrd_ax
    
    # Create a grid for at which to sample the smoothed parameters
    subplot_size = 15
    border = int(subplot_size / 2)
    grid_x, grid_y = np.mgrid[border:ccd_shape[0]-border:subplot_size, 
                                border:ccd_shape[1]-border:subplot_size]
    
    try:
        param_list, _ = loess_2d(data_x, data_y, data_vals, xnew=grid_x.flatten(),
                                 ynew=grid_y.flatten(), frac=frac)
    except np.linalg.LinAlgError:
        logger.warning("LinAlgError: SVD did not converge in Linear Least Squares")
        logger.warning("Skipping this contour plot")
        return ax, None
    param_list = param_list.reshape(grid_x.shape)
    
    
    colors = ["#cd0000", "#cb4000", "#c97f00", "#c7bc00", "#91c400", "#52c200", 
              "#00bc62", "#00ba9c", "#009cb8", "#0061b6", "#1100b1", "#4800af", "#7e00ad"]
    levels = np.linspace(color_range[0], color_range[1], len(colors))

    # Create contour plot
    cp = ax.contourf(grid_x, grid_y, param_list, levels=levels, colors=colors)
    ax.set_title(f'{title} Graph - {category_str}')
    
    return ax, cp


def scatter_sources(data_x, data_y, data_vals, color_range, backgrd_ax=None, 
                    title=None, category_str=None):
    
    if backgrd_ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
    else:
        ax = backgrd_ax
    
    # Define the colors & range of contour levels
    colors = ["#cd0000", "#cb4000", "#c97f00", "#c7bc00", "#91c400", "#52c200", 
                "#00bc62", "#00ba9c", "#009cb8", "#0061b6", "#1100b1", "#4800af"]
    levels = np.linspace(color_range[0], color_range[1], len(colors))
    cmap_custom = ListedColormap(colors)

    # Create a scatter plot, color coded by param_list value
    ax.set_title(f'{title} Graph - {category_str}')
    jitter_x = np.random.normal(scale=7, size=len(data_x))
    jitter_y = np.random.normal(scale=7, size=len(data_y))
    ax.scatter(data_x+jitter_x, data_y+jitter_y, s=15, c=data_vals, cmap=cmap_custom, 
               vmin=levels[0], vmax=levels[-1], alpha=1.0,
               linewidths=0.7, edgecolors='k')
    
    return ax, cmap_custom



def plot_sources(phot_data, given_fwhm, image=None, flux_name='flux_fit',
                 x_name='x_fit', y_name='y_fit', label_name='group_id',
                 scale=1):
    
    if image is None:
        image = Fits_Simple(phot_data.meta['image_path'])
    logger.info(f'Plotting image {image}')
    
    if flux_name == 'flux_fit' and 'flux_fit' not in phot_data.colnames:
        flux_name = 'flux_psf'
    
    def get_apertures(phot_data):
        x = phot_data[x_name]
        y = phot_data[y_name]
        positions = np.transpose((x, y))
        return CircularAperture(positions, r=2*given_fwhm*scale)
    
    good_phot_data = phot_data[phot_data['group_size'] <= 1]
    bad_phot_data = phot_data[phot_data['group_size'] > 1]
    bad_apertures = get_apertures(bad_phot_data)
    good_apertures = get_apertures(good_phot_data)
    
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image.masked_array)
    cmap = plt.get_cmap()
    cmap.set_bad('r', alpha=0.5)
    plt.figure(figsize=(12,10))
    plt.title(image)
    plt.imshow(image.masked_array, origin='lower', vmin=vmin, vmax=vmax,
               cmap=cmap, interpolation='nearest')
    plt.colorbar()
    good_apertures.plot(color='purple', lw=1.5*scale, alpha=1)
    bad_apertures.plot(color='r', lw=1.5*scale, alpha=1)
    
    # Annotate good sources with flux_fit values
    y_offset = 3.5*given_fwhm*scale
    for i in range(len(good_phot_data)):
        plt.text(good_phot_data[x_name][i], good_phot_data[y_name][i]+y_offset, 
                 f'{good_phot_data[label_name][i]}: {good_phot_data[flux_name][i]:.0f}',
                 color='white', fontsize=8*scale, ha='center', va='center')
    
    group_ids = set(bad_phot_data[label_name])
    for id in group_ids:
        group = bad_phot_data[bad_phot_data[label_name] == id]
        group_x = np.mean(group[x_name])
        group_y = np.mean(group[y_name]) + y_offset
        for i in range(len(group)):
            plt.text(group_x, group_y+i*20*scale, 
                     f'{id}: {group[flux_name][i]:.0f}',
                     color='red', fontsize=8*scale, ha='center', va='center')
    
    plt.gcf().set_dpi(300)
    plt.show()
