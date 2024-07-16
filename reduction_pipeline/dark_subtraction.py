from astropy.io import fits
import numpy as np

def dark_subtraction(ifilelist, ofilelist, dark):
    """
    Does dark subtraction on a list of FITS files.

    Args:
        ifilelist: list of faw fits files
        ofilelist: list of overscan subtracted fits files
                   (will overwrite if same as ifilelist)
        dark: dark frame to subtract (should correspond to same
                   exp time as files in ifilelist)
    """

    numifiles = len(ifilelist)
    numofiles = len(ofilelist)

    for i in range(0,numifiles):
        ifile=ifilelist[i]
        ofile=ofilelist[i]
        data, header = fits.getdata(ifile,header=True)
        datanew = data - dark
        updated_history = str(header['HISTORY']) + ', dark subtracted'
        del header['HISTORY']
        header['HISTORY'] = updated_history
        fits.writeto(ofile,datanew,header,overwrite=True)
