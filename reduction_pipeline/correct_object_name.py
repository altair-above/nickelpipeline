from astropy.io import fits

def correct_object_name(ifilelist, obj):
    """
    Changes object names in raw fits files.

    Args:
        ifilelist: list of faw fits files
        obj (string): the correct object name
    """

    numifiles = len(ifilelist)

    for i in range(0,numifiles):
        ifile=ifilelist[i]
        data, header = fits.getdata(ifile,header=True)
        header['OBJECT'] = obj
        fits.writeto(ifile,data,header,overwrite=True)