import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from loess.loess_2d import loess_2d
from nickelpipeline.convenience.nickel_data import ccd_shape


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
        print("LinAlgError: SVD did not converge in Linear Least Squares")
        print("Skipping this contour plot")
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

