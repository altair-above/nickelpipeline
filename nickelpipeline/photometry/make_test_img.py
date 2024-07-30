
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Moffat2D
from astropy.visualization import ZScaleInterval


def make_grid(grid_size, step_size, x=0, y=0):
    half_size = grid_size / 2
    x_start, x_end = -half_size + step_size / 2, half_size - step_size / 2
    y_start, y_end = -half_size + step_size / 2, half_size - step_size / 2
    
    x_start, x_end, y_start, y_end = x_start + x, x_end + x, y_start + y, y_end + y
    
    x_coords = np.arange(x_start, x_end + step_size, step_size)
    y_coords = np.arange(y_start, y_end + step_size, step_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    return grid_x, grid_y


def make_img(amplitude, gamma, alpha):
    
    # Define the dimensions and parameters
    shape = (1024, 1024)
    mean = 70
    std_dev = 15
    # Create the array with noise
    data = np.random.normal(loc=mean, scale=std_dev, size=shape)
    plt.imshow(data)
    plt.colorbar()
    plt.show()
    
    # pixel_coords = make_grid(1024, 1.0)
    source_x, source_y = make_grid(shape[0], 200, shape[0]/2, shape[1]/2)
    # print(source_x)
    # print(source_y)
    source_x = source_x.flatten()
    source_y = source_y.flatten()
    
    source_stamps_coords = [make_grid(20, 1.0, x, y) for x,y in zip(source_x, source_y)]
    # print("source stamp coords")
    # print(source_stamps_coords)
    
    source_stamps = [moffat_integral_pixel(stamp[0], stamp[1], amplitude, gamma, alpha,) 
                     for stamp in source_stamps_coords]
    
    for stamp, stamp_coords in zip(source_stamps, source_stamps_coords):
        # print(stamp)
        # plt.imshow(stamp)
        # plt.show()
        
        for x, y, value in zip(stamp_coords[0].flatten(), stamp_coords[1].flatten(), stamp.flatten()):
            x, y = int(x), int(y)
            try:
                data[x,y] += value
            except IndexError:
                continue
        # plt.imshow(data)
        # plt.show()
    
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    cmap = plt.get_cmap()
    cmap.set_bad('r', alpha=0.5)
    plt.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()
           
    
    
    

def discrete_moffat_integral(amplitude, gamma, alpha, step_size=1.0):
    # Define the grid size and step size
    grid_size = 10

    # Calculate the start and end points
    half_size = grid_size // 2
    x_start, x_end = -half_size + step_size / 2, half_size - step_size / 2
    y_start, y_end = half_size - step_size / 2, -half_size + step_size / 2
    x_coords = np.arange(x_start, x_end + step_size, step_size)
    y_coords = np.arange(y_start, y_end - step_size, -step_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    # print(grid_x)

    pixel_fluxes = Moffat2D.evaluate(grid_x, grid_y, amplitude, 0, 0, gamma, alpha)
    pixel_fluxes *= step_size**2
    # print(pixel_fluxes)
    print(f"total flux = {np.sum(pixel_fluxes)}")
    return np.sum(pixel_fluxes)

def moffat_integral_pixel(all_x, all_y, amplitude, gamma, alpha):
    # if not isinstance(all_x, np.array):
    #     all_x = np.array(all_x)
    # if not isinstance(all_y, np.array):
    #     all_y = np.array(all_y)
    
    # Define the grid size and step size
    grid_size = 1.0
    step_size = 0.01
    
    pixel_flux_grid = np.zeros(all_x.shape)
    # print(all_x.shape)
    # print(pixel_flux_grid.shape)
    
    center_x = all_x[all_x.shape[0]//2, all_x.shape[1]//2]
    center_y = all_y[all_y.shape[0]//2, all_y.shape[1]//2]
    
    for (i,j) in np.ndindex(all_x.shape):
        # print(i,j)
        # # Calculate the start and end points
        # half_size = grid_size / 2
        # x_start, x_end = -half_size + step_size / 2, half_size - step_size / 2
        # y_start, y_end = half_size - step_size / 2, -half_size + step_size / 2
        # x_start, x_end, y_start, y_end = x_start + x, x_end + x, y_start + y, y_end + y
        
        # x_coords = np.arange(x_start, x_end + step_size, step_size)
        # y_coords = np.arange(y_start, y_end - step_size, -step_size)
        # grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        # print(grid_x)
        x = all_x[i,j]
        y = all_y[i,j]
                
        grid_x, grid_y = make_grid(grid_size, step_size, x, y)

        pixel_fluxes = Moffat2D.evaluate(grid_x, grid_y, amplitude, center_x, center_y, gamma, alpha)
        pixel_fluxes *= step_size**2
        # print(pixel_fluxes)
        # print(f"total flux = {np.sum(pixel_fluxes)}")
        flux = np.sum(pixel_fluxes)
        pixel_flux_grid[i,j] = flux
    
    # print(pixel_flux_grid.shape)
    # print(pixel_flux_grid)
    return pixel_flux_grid