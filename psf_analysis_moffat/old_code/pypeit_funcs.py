


""" Module for image processing core methods

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
from IPython import embed

import warnings

from astropy.convolution import convolve, Box2DKernel
from astropy.timeseries import LombScargle
import astropy.stats
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.signal

# from pypeit import msgs
# from pypeit import utils


def deimos_read_1chip(hdu,chipno):
    """ Read one of the DEIMOS detectors

    Args:
        hdu (astropy.io.fits.HDUList):
        chipno (int):

    Returns:
        np.ndarray, np.ndarray:
            data, oscan
    """

    # Extract datasec from header
    datsec = hdu[chipno].header['DATASEC']
    detsec = hdu[chipno].header['DETSEC']
    postpix = hdu[0].header['POSTPIX']
    precol = hdu[0].header['PRECOL']

    x1_dat, x2_dat, y1_dat, y2_dat = np.array(load_sections(datsec)).flatten()
    x1_det, x2_det, y1_det, y2_det = np.array(load_sections(detsec)).flatten()

    # This rotates the image to be increasing wavelength to the top
    #data = np.rot90((hdu[chipno].data).T, k=2)
    #nx=data.shape[0]
    #ny=data.shape[1]


    # Science data
    fullimage = hdu[chipno].data
    data = fullimage[x1_dat:x2_dat,y1_dat:y2_dat]

    # Overscan
    oscan = fullimage[:,y2_dat:]

    # Flip as needed
    if x1_det > x2_det:
        data = np.flipud(data)
        oscan = np.flipud(oscan)
    if y1_det > y2_det:
        data = np.fliplr(data)
        oscan = np.fliplr(oscan)

    # Return
    return data, oscan


def load_sections(string, fmt_iraf=True):
    """
    From the input string, return the coordinate sections.
    In IRAF format (1 index) or Python

    Parameters
    ----------
    string : str
      character string of the form [x1:x2,y1:y2]
      x1 = left pixel
      x2 = right pixel
      y1 = bottom pixel
      y2 = top pixel
    fmt_iraf : bool
      Is the variable string in IRAF format (True) or
      python format (False)

    Returns
    -------
    sections : list or None
      the detector sections
    """
    xyrng = string.strip('[]()').split(',')
    if xyrng[0] == ":":
        xyarrx = [0, 0]
    else:
        xyarrx = xyrng[0].split(':')
        # If a lower/upper limit on the array slicing is not given (e.g. [:100] has no lower index specified),
        # set the lower/upper limit to be the first/last index.
        if len(xyarrx[0]) == 0: xyarrx[0] = 0
        if len(xyarrx[1]) == 0: xyarrx[1] = -1
    if xyrng[1] == ":":
        xyarry = [0, 0]
    else:
        xyarry = xyrng[1].split(':')
        # If a lower/upper limit on the array slicing is not given (e.g. [5:] has no upper index specified),
        # set the lower/upper limit to be the first/last index.
        if len(xyarry[0]) == 0: xyarry[0] = 0
        if len(xyarry[1]) == 0: xyarry[1] = -1
    if fmt_iraf:
        xmin = max(0, int(xyarry[0])-1)
        xmax = int(xyarry[1])
        ymin = max(0, int(xyarrx[0])-1)
        ymax = int(xyarrx[1])
    else:
        xmin = max(0, int(xyarrx[0]))
        xmax = int(xyarrx[1])
        ymin = max(0, int(xyarry[0]))
        ymax = int(xyarry[1])
    return [[xmin, xmax], [ymin, ymax]]


def subtract_overscan(rawframe, datasec_img, oscansec_img, method='savgol', params=[5,65],
                      var=None):
    """
    Subtract the overscan

    Possible values of ``method``:
        - polynomial: Fit a polynomial to the overscan region and subtract it.
        - savgol: Use a Savitzky-Golay filter to fit the overscan region and
            subtract it.
        - median: Use the median of the overscan region to subtract it.
        - odd_even: Use the median of the odd and even rows/columns to subtract (MDM/OSMOS)

    Args:
        rawframe (`numpy.ndarray`_):
            Frame from which to subtract overscan.  Must be 2d.
        datasec_img (`numpy.ndarray`_):
            An array the same shape as ``rawframe`` that identifies the pixels
            associated with the data on each amplifier; 0 for no data, 1 for
            amplifier 1, 2 for amplifier 2, etc.
        oscansec_img (:obj:`numpy.ndarray`):
            An array the same shape as ``rawframe`` that identifies the pixels
            associated with the overscan region on each amplifier; 0 for no
            data, 1 for amplifier 1, 2 for amplifier 2, etc.
        method (:obj:`str`, optional):
            The method used to fit the overscan region.  Options are
            chebyshev, polynomial, savgol, median.  ("polynomial" is deprecated
            and will be removed)
        params (:obj:`list`, optional):
            Parameters for the overscan subtraction.  For ``method=chebyshev``
            or ``method=polynomial``set ``params`` to the order;
            for ``method=savgol``, set ``params`` to the order and window size;
            for ``method=median``, ``params`` are ignored.
        var (`numpy.ndarray`_, optional):
            Variance in the raw frame.  If provided, must have the same shape as
            ``rawframe`` and used to estimate the error in the overscan
            subtraction.  The estimated error is the standard error in the
            median for the pixels included in the overscan correction.  This
            estimate is also used for the ``'savgol'`` method as an upper limit.
            If None, no variance in the overscan subtraction is calculated, and
            the 2nd object in the returned tuple is None.

    Returns:
        :obj:`tuple`: The input frame with the overscan region subtracted and an
        estimate of the variance in the overscan subtraction; both have the same
        shape as the input ``rawframe``.  If ``var`` is not provided, the 2nd
        returned object is None.
    """
    # Check input
    if method.lower() not in ['polynomial', 'chebyshev', 'savgol', 'median', 'odd_even']:
        raise Exception(f'Unrecognized overscan subtraction method: {method}')
    if rawframe.ndim != 2:
        raise Exception('Input raw frame must be 2D.')
    if datasec_img.shape != rawframe.shape:
        raise Exception('Datasec image must have the same shape as the raw frame.')
    if oscansec_img.shape != rawframe.shape:
        raise Exception('Overscan section image must have the same shape as the raw frame.')
    if var is not None and var.shape != rawframe.shape:
        raise Exception('Variance image must have the same shape as the raw frame.')

    # Copy the data so that the subtraction is not done in place
    no_overscan = rawframe.copy()
    _var = None if var is None else np.zeros(var.shape, dtype=float)

    # Amplifiers
    amps = np.unique(datasec_img[datasec_img > 0]).tolist()

    # Perform the overscan subtraction for each amplifier
    for amp in amps:
        # Pull out the overscan data
        if np.sum(oscansec_img == amp) == 0:
            raise Exception(f'No overscan region for amplifier {amp+1}!')
        overscan, os_slice = rect_slice_with_mask(rawframe, oscansec_img, amp)
        if var is not None:
            osvar = var[os_slice]
        # Pull out the real data
        if np.sum(datasec_img == amp) == 0:
            raise Exception(f'No data region for amplifier {amp+1}!')
        data, data_slice = rect_slice_with_mask(rawframe, datasec_img, amp)

        # Shape along at least one axis must match
        if not np.any([dd == do for dd, do in zip(data.shape, overscan.shape)]):
            raise Exception('Overscan sections do not match amplifier sections for '
                       'amplifier {0}'.format(amp))
        compress_axis = 1 if data.shape[0] == overscan.shape[0] else 0

        # Fit/Model the overscan region
        osfit = np.median(overscan) if method.lower() == 'median' \
                    else np.median(overscan, axis=compress_axis)
        if var is not None:
            # pi/2 coefficient yields asymptotic variance in the median relative
            # to the error in the mean
            osvar = np.pi/2*(np.sum(osvar)/osvar.size**2 if method.lower() == 'median' 
                             else np.sum(osvar, axis=compress_axis)/osvar.shape[compress_axis]**2)
        # Method time
        if method.lower() == 'polynomial':
            warnings.warn('Method "polynomial" is identical to "chebyshev".  Former will be deprecated.',
                          DeprecationWarning)
        if method.lower() in ['polynomial', 'chebyshev']:
            poly = np.polynomial.Chebyshev.fit(np.arange(osfit.size), osfit, params[0])
            ossub = poly(np.arange(osfit.size))
        elif method.lower() == 'savgol':
            ossub = scipy.signal.savgol_filter(osfit, params[1], params[0])
        elif method.lower() == 'median':
            # Subtract scalar and continue
            no_overscan[data_slice] -= osfit
            if var is not None:
                _var[data_slice] = osvar
            continue
        elif method.lower() == 'odd_even':
            # Odd/even
            # Different behavior depending on overscan geometry
            _overscan = overscan if compress_axis == 1 else overscan.T
            _no_overscan = no_overscan[data_slice] if compress_axis == 1 \
                               else no_overscan[data_slice].T
            # Compute median overscan of odd and even pixel stripes in overscan
            odd = np.median(_overscan[:,1::2], axis=1)
            even = np.median(_overscan[:,0::2], axis=1)
            # Do the same for the data
            odd_data = np.median(_no_overscan[:,1::2], axis=1)
            even_data = np.median(_no_overscan[:,0::2], axis=1)
            # Check for odd/even row alignment between overscan and data,
            # which can be instrument/data reader-dependent when compress_axis is 0.
            # Could be possibly be improved by removing average odd/even slopes in data
            aligned = np.sign(np.median(odd-even)) == np.sign(np.median(odd_data-even_data))
            if not aligned and compress_axis == 0:
                odd, even = even, odd
            # Now subtract
            _no_overscan[:,1::2] -= odd[:,None]
            _no_overscan[:,0::2] -= even[:,None]
            no_overscan[data_slice] = _no_overscan if compress_axis == 1 else _no_overscan.T
            if var is not None:
                _osvar = var[os_slice] if compress_axis == 1 else var[os_slice].T
                odd_var = np.sum(_osvar[:,1::2],axis=1)/_osvar[:,1::2].size**2
                even_var = np.sum(_osvar[:,0::2],axis=1)/_osvar[:,0::2].size**2
                if not aligned and compress_axis == 0:
                    odd_var, even_var = even_var, odd_var
                __var = _var[data_slice] if compress_axis == 1 else _var[data_slice].T
                __var[:,1::2] = np.pi/2 * odd_var[:,None]
                __var[:,0::2] = np.pi/2 * even_var[:,None]
                _var[data_slice ] = __var if compress_axis == 1 else __var.T
            continue


        # Subtract along the appropriate axis
        no_overscan[data_slice] -= (ossub[:, None] if compress_axis == 1 else ossub[None, :])
        if var is not None:
            _var[data_slice] = (osvar[:,None] if compress_axis == 1 else osvar[None,:])

    # Return
    return no_overscan, _var


def rect_slice_with_mask(image, mask, mask_val=1):
    """
    Generate rectangular slices from a mask image.

    Args:
        image (`numpy.ndarray`_):
            Image to mask
        mask (`numpy.ndarray`_):
            Mask image
        mask_val (:obj:`int`, optional):
            Value to mask on

    Returns:
        :obj:`tuple`: The image at mask values and a 2-tuple with the
        :obj:`slice` objects that select the masked data.
    """
    pix = np.where(mask == mask_val)
    slices = (slice(np.min(pix[0]), np.max(pix[0])+1), slice(np.min(pix[1]), np.max(pix[1])+1))
    return image[slices], slices



