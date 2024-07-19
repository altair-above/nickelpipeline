
import operator
from pathlib import Path
import tomllib

from IPython import embed

import numpy

from astropy.io import fits
from astropy.nddata import CCDData
import ccdproc

valid_operators = {'==':operator.eq, '!=':operator.ne,
                   '>=':operator.ge, '>':operator.gt,
                   '<=':operator.le, '<':operator.lt}


def nickel_oscansec(hdr):
    nc = hdr['NAXIS1']
    no = hdr['COVER']
    nr = hdr['NAXIS2']
    return f'[{nc-no+1}:{nc},1:{nr}]'


def parse_criterion(c):
    for op in valid_operators.keys():
        if op in c:
            hkey, value = c.split(op)
            return hkey, valid_operators[op], value
    raise ValueError(f'Unable to parse {c} as header keyword criterion')


def check_par(par):
    req_keys = numpy.array(['raw_root'])
    indx = [key not in par.keys() for key in req_keys]
    if any(indx):
        raise ValueError(f'required keywords not specified in parameter file: {req_keys[indx]}')


def sort(par, frame_type):

    # Check input
    if frame_type not in par.keys():
        raise ValueError(f'{frame_type} keyword not defined in parameter file.')
    if not isinstance(par[frame_type], dict):
        raise TypeError(f'{frame_type} keyword must have sub-parameters.')

    # Get the conditional criteria
    criteria = []
    if 'find' in par[frame_type].keys():
        # Find frames that match the given header keyword constraints
        for c in par[frame_type]['find'].split(','):
            criteria += [parse_criterion(c)]

    # Get the value groupings
    groups = []
    if 'group' in par[frame_type].keys():
        groups = par[frame_type]['group'].split(',')

    # Path with raw files
    raw_path = Path(par['raw_root']).absolute()
    # Get 'em
    raw_files = sorted(raw_path.glob('*.fits'))

    if len(criteria) == 0 and len(groups) == 0:
        return {'':numpy.asarray(raw_files)}

    # Sort frames based on their selection criteria and groupings
    selected_frames = []
    group_values = []
    for f in raw_files:
        with fits.open(f) as hdu:
            meets_criteria = True
            for c in criteria:
                t = type(hdu[0].header[c[0]])
                if not c[1](hdu[0].header[c[0]], t(c[2])):
                    meets_criteria = False
                    break
            if meets_criteria:
                selected_frames += [f]
                if len(groups) > 0:
                    group_values += ['_'.join([str(hdu[0].header[g]).strip() for g in groups])]

    if len(groups) == 0:
        return {'':numpy.asarray(selected_frames)}

    # Group images
    group_values = numpy.asarray(group_values)
    selected_frames = numpy.asarray(selected_frames)
    uniq_groups = numpy.unique(group_values)
    return {u: selected_frames[group_values == u] for u in uniq_groups}


def overscan_trim(frame, **kwargs):
    ccd = CCDData.read(frame, unit='adu')
    oscansec = nickel_oscansec(ccd.header)
    proc_ccd = ccdproc.subtract_overscan(ccd, fits_section=oscansec, overscan_axis=1)
    return ccdproc.trim_image(proc_ccd, fits_section=ccd.header['DATASEC'])


def proc_bias(par, frames, save=False):

    # Path for reduced files
    rdx_path = Path(par['rdx_root'] if 'rdx_root' in par.keys() else '.').absolute()

    stack = []
    for frame in frames:
        stack += [overscan_trim(frame)]

    combiner = ccdproc.Combiner(stack)
    old_n_masked = 0
    new_n_masked = 1
    i = 0
    while new_n_masked > old_n_masked:
        combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=numpy.ma.mean)
        old_n_masked = new_n_masked
        new_n_masked = combiner.data_arr.mask.sum()
        i += 1

    stacked_bias = combiner.average_combine()
    if save:
        stacked_bias.write(rdx_path / 'Bias.fits')
    return stacked_bias


def main():

    parfile = 'rdx.toml'
    with open(parfile, 'rb') as f:
        par = tomllib.load(f)

    check_par(par)

    bias_groups = sort(par, 'bias')
    flat_groups = sort(par, 'flat')
    object_groups = sort(par, 'object')

    # Process the biases
    for key, frames in bias_groups.items():
        stacked_bias = proc_bias(par, frames, save=True)

        embed()
        exit()
    


if __name__ == '__main__':
    main()

