from astropy.io import fits

def bias_subtraction(ifilelist, ofilelist, bias):
    """
    Does bias subtraction on a list of FITS files.

    Args:
        ifilelist: list of faw fits files
        ofilelist: list of overscan subtracted fits files
                   (will overwrite if same as ifilelist)
        bias: bias frame to subtract
    """

    numifiles = len(ifilelist)
    numofiles = len(ofilelist)

    for i in range(0,numifiles):
        ifile=ifilelist[i]
        ofile=ofilelist[i]
        # print(ifile)
        with fits.open(ifile) as hdu:
            data = hdu[0].data
            header = hdu[0].header
        # data, header = fits.getdata(ifile,header=True)
        datanew = data - bias
        updated_history = str(header['HISTORY']) + ', bias subtracted'
        del header['HISTORY']
        header['HISTORY'] = updated_history
        fits.writeto(ofile,datanew,header=header,overwrite=True)
