from astropy.io import fits
import numpy as np

def flat_division(ifilelist, ofilelist, flat):
    """
    Does flat division on a list of FITS files.
    Divides pixel-by-pixel then multiplies by the
    average value of the flat frame.

    Args:
        ifilelist: list of faw fits files
        ofilelist: list of overscan subtracted fits files
                   (will overwrite if same as ifilelist)
        flat: flat frame to subtract (should correspond to
                   same filter as files in ifilelist)
    """

    numifiles = len(ifilelist)
    numofiles = len(ofilelist)

    for i in range(0,numifiles):
        ifile=ifilelist[i]
        ofile=ofilelist[i]
        data, header = fits.getdata(ifile,header=True)
        avgvalue = np.mean(flat)
        datanew = (data/flat) * avgvalue
        updated_history = str(header['HISTORY']) + ', flat divided'
        del header['HISTORY']
        header['HISTORY'] = updated_history
        fits.writeto(ofile,datanew,header,overwrite=True)