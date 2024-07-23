# #  from https://mthamilton.ucolick.org/techdocs/instruments/nickel_direct/obs_hints/#bias

# #!/bin/python3

# # Note: You may have to change the above path to Python 3.x to the appropriate path to start python3 on your local computer.

# # Version 3.1 -- Elinor Gates, 2020 Aug 4  - Adapted for Hamilton data
# # Version 3.0 -- Elinor Gates, 2019 Oct 4  - Adapted original Python 2.7 script to Python 3
# # Version 2.2 -- Elinor Gates, 2018 Oct 3  - changed clobber to overwrite in fits.writeto
# # Version 2.1 -- Elinor Gates, 2016 Aug 25 - changed math to BITPIX -32
# # Version 2.0 -- Elinor Gates, 2016 Aug 11 - improved file reading
# # Version 1.0 -- Elinor Gates, 2015 Nov 24


# from astropy.io import fits, ascii
# import numpy as np
# import sys, getopt

# def main(argv):
#     inputfilelist = ''
#     outputfilelist = ''
#     fit = 'no'
#     try:
#         opts, args = getopt.getopt(argv,"hfi:o:",["ifilelist=","ofilelist="])
#     except getopt.GetoptError:
#       print('overscanHamP3.py -f -i <inputfilelist> -o <outputfilelist>')
#       print('-f indicates do a Legendre fit to overscan')
#       sys.exit(2)
#     for opt, arg in opts:
#       if opt == '-h':
#          print('overscanHamP3.py -f -i <inputfilelist> -o <outputfilelist>')
#          print('-f indicates do a Legendre fit to overscan')
#          sys.exit(2)
#       elif opt in ("-i", "--ifilelist"):
#          inputfilelist = arg
#       elif opt in ("-o", "--ofilelist"):
#          outputfilelist = arg
#       elif opt == '-f':
#           fit = 'yes'
#     print('Input filelist is ', inputfilelist)
#     print('Output filelist is ', outputfilelist)
#     print('Fit is ', fit)

#     # open input and output filelists
#     ifilelist = [line.rstrip('\n') for line in open(inputfilelist)]
#     ofilelist = [line.rstrip('\n') for line in open(outputfilelist)]

#     # how many files
#     numifiles = len(ifilelist)
#     numofiles = len(ofilelist)
#     if numifiles != numofiles:
#         sys.exit('Input and output file lists have different numbers of files. Exiting.')

#     # for each file in ifilelist, read in file, figure out overscan and data regions, fit
#     # overscan with desired function (if any), and subtract from data.
#     # Write data to ofilelist value.

#     for i in range(0,numifiles):
#         ifile=ifilelist[i]
#         ofile=ofilelist[i]
#         data, header = fits.getdata(ifile,header=True)

#         # change data to float
#         data=data.astype('float32')

#         # read necessary keywords from fits header

#         #number of pixels in image
#         xsize = header['NAXIS1']
#         ysize = header['NAXIS2']
#         #start column and row
#         xorig = header['CRVAL1U']
#         yorig = header['CRVAL2U']
#         #binning and direction of reading pixels
#         cdelt1 = header['CDELT1U']
#         cdelt2 = header['CDELT2U']
#         #number of overscan columns
#         rover = header['ROVER']
#         cover = header['COVER']
#         #unbinned detector size
#         detxsize = header['DNAXIS1']
#         detysize = header['DNAXIS2']
#         #number of amplifiers
#         ampsx = header['AMPSCOL']
#         ampsy = header['AMPSROW']

#         # determine number and sizes of overscan and data regions
#         namps = ampsx*ampsy
#         if rover > 0:
#             over=rover
#             sys.exit('Program does not yet deal with row overscans. Exiting.')
#         else:
#             over = cover
#         if over == 0:
#             sys.exit('No overscan region specified in FITS header. Exiting.')

#         # single amplifier mode (assumes overscan is the righmost columns)
#         if namps == 1:
#             biassec = data[:,xsize-cover:xsize]
#             datasec = data[0:,0:xsize-cover]

#             # median overscan section
#             bias=np.median(biassec, axis=1)

#             # legendre fit
#             if fit == 'yes':
#                 # fit
#                 lfit = np.polynomial.legendre.legfit(range(0,len(bias)),bias,3)
#                 bias = np.polynomial.legendre.legval(range(0,len(bias)),lfit)

#             # subtract overscan
#             datanew = datasec
#             for i in range(datasec.shape[1]):
#                 datanew[:,i] = datasec[:,i]-bias

#         # two amplifier mode (assumes both amplifer overscans are at rightmost columns)
#         if namps == 2:
#             biasseca = data[:,xsize-cover*2:xsize-cover]
#             biassecb = data[:,xsize-cover:xsize]

#             # median overscan sections
#             biasa=np.median(biasseca,axis=1)
#             biasb=np.median(biassecb,axis=1)

#             # legendre fit
#             if fit == 'yes':
#                 lfita = np.polynomial.legendre.legfit(range(0,len(biasa)),biasa,3)
#                 lfitb = np.polynomial.legendre.legfit(range(0,len(biasb)),biasb,3)
#                 biasa = np.polynomial.legendre.legval(range(0,len(biasa)),lfita)
#                 biasb = np.polynomial.legendre.legval(range(0,len(biasb)),lfitb)

#             # extract data regions

#             #determine boundary between amplifiers
#             bd=detxsize/2/abs(cdelt1)

#             # calculate x origin of readout in binned units if cdelt1 negative or positive
#             if cdelt1 < 0:
#                 #if no binning x0=xorig-xsize-2*cover, with binning:
#                 x0=xorig/abs(cdelt1)- (xsize-2*cover)
#             else:
#                 x0=xorig/cdelt1

#             xtest=x0+xsize-cover*2 # need to test if all data on one or two amplifiers

#             # determine which columns are on which amplifier and subtract proper overscan region

#             if xtest < bd: # all data on left amplifier
#                 datanew=data[:,0:xsize-cover*2]
#                 m=datanew.shape[1]
#                 for i in range(0,m):
#                     datanew[:,i]=datanew[:,i]-biasa

#             if x0 >= bd: # all data on right amplifier
#                 datanew=data[:,0:xsize-cover*2]
#                 m=datanew.shape[1]
#                 for i in range(0,m):
#                     datanew[:,i]=datanew[:,i]-biasb

#             if xtest >= bd and x0 < bd:  #data on both amplifiers
#                 x1=int(bd-x0)
#                 dataa=data[:,0:x1]
#                 datab=data[:,x1:-cover*2]
#                 ma=dataa.shape[1]
#                 mb=datab.shape[1]
#                 for i in range(0,ma):
#                     dataa[:,i]=dataa[:,i]-biasa
#                 for i in range(0,mb):
#                     datab[:,i]=datab[:,i]-biasb
#                 # merge dataa and datab into single image
#                 datanew=np.hstack([dataa,datab])

#         if namps > 2:
#             sys.exit('Program does not yet deal with more than two overscan regions. Exiting.')

#         # add info to header
#         header['HISTORY'] = 'Overscan subtracted'

#         # write new fits file
#         fits.writeto(ofile,datanew,header,overwrite=True)

# if __name__ == "__main__":
#    main(sys.argv[1:])

from astropy.io import fits, ascii
import numpy as np
import sys, getopt

def overscan_subtraction(ifilelist, ofilelist, fit = 'no'):
    """
    Does overscan subtraction on a list of FITS files.

    Adapted from code by Elinor Gates at UCO Lick
    Version 3.1 -- Elinor Gates, 2020 Aug 4  - Adapted for Hamilton data
    Version 3.0 -- Elinor Gates, 2019 Oct 4  - Adapted original Python 2.7 script to Python 3
    Version 2.2 -- Elinor Gates, 2018 Oct 3  - changed clobber to overwrite in fits.writeto
    Version 2.1 -- Elinor Gates, 2016 Aug 25 - changed math to BITPIX -32
    Version 2.0 -- Elinor Gates, 2016 Aug 11 - improved file reading
    Version 1.0 -- Elinor Gates, 2015 Nov 24

    Args:
        ifilelist: list of faw fits files
        ofilelist: list of overscan subtracted fits files
        fit: 'yes' if Legendre polynomial fit is desired (recommended)
    """

    # how many files
    numifiles = len(ifilelist)
    numofiles = len(ofilelist)
    if numifiles != numofiles:
        sys.exit('Input and output file lists have different numbers of files. Exiting.')

    # for each file in ifilelist, read in file, figure out overscan and data regions, fit
    # overscan with desired function (if any), and subtract from data.
    # Write data to ofilelist value.

    for i in range(0,numifiles):
        ifile=ifilelist[i]
        ofile=ofilelist[i]
        with fits.open(ifile) as hdu:
            data = hdu[0].data
            header = hdu[0].header
        # data, header = fits.getdata(ifile,header=True)

        # change data to float
        data=data.astype('float32')

        # read necessary keywords from fits header

        #number of pixels in image
        xsize = header['NAXIS1']
        ysize = header['NAXIS2']
        #start column and row
        xorig = header['CRVAL1U']
        yorig = header['CRVAL2U']
        #binning and direction of reading pixels
        cdelt1 = header['CDELT1U']
        cdelt2 = header['CDELT2U']
        #number of overscan columns
        rover = header['ROVER']
        cover = header['COVER']
        #unbinned detector size
        detxsize = header['DNAXIS1']
        detysize = header['DNAXIS2']
        #number of amplifiers
        ampsx = header['AMPSCOL']
        ampsy = header['AMPSROW']

        # determine number and sizes of overscan and data regions
        namps = ampsx*ampsy
        if rover > 0:
            over=rover
            sys.exit('Program does not yet deal with row overscans. Exiting.')
        else:
            over = cover
        if over == 0:
            sys.exit('No overscan region specified in FITS header. Exiting.')

        # single amplifier mode (assumes overscan is the righmost columns)
        if namps == 1:
            biassec = data[:,xsize-cover:xsize]
            datasec = data[0:,0:xsize-cover]

            # median overscan section
            bias=np.median(biassec, axis=1)

            # legendre fit
            if fit == 'yes':
                # fit
                lfit = np.polynomial.legendre.legfit(range(0,len(bias)),bias,3)
                bias = np.polynomial.legendre.legval(range(0,len(bias)),lfit)

            # subtract overscan
            datanew = datasec
            for i in range(datasec.shape[1]):
                datanew[:,i] = datasec[:,i]-bias

        # two amplifier mode (assumes both amplifer overscans are at rightmost columns)
        if namps == 2:
            biasseca = data[:,xsize-cover*2:xsize-cover]
            biassecb = data[:,xsize-cover:xsize]

            # median overscan sections
            biasa=np.median(biasseca,axis=1)
            biasb=np.median(biassecb,axis=1)

            # legendre fit
            if fit == 'yes':
                lfita = np.polynomial.legendre.legfit(range(0,len(biasa)),biasa,3)
                lfitb = np.polynomial.legendre.legfit(range(0,len(biasb)),biasb,3)
                biasa = np.polynomial.legendre.legval(range(0,len(biasa)),lfita)
                biasb = np.polynomial.legendre.legval(range(0,len(biasb)),lfitb)

            # extract data regions

            #determine boundary between amplifiers
            bd=detxsize/2/abs(cdelt1)

            # calculate x origin of readout in binned units if cdelt1 negative or positive
            if cdelt1 < 0:
                #if no binning x0=xorig-xsize-2*cover, with binning:
                x0=xorig/abs(cdelt1)- (xsize-2*cover)
            else:
                x0=xorig/cdelt1

            xtest=x0+xsize-cover*2 # need to test if all data on one or two amplifiers

            # determine which columns are on which amplifier and subtract proper overscan region

            if xtest < bd: # all data on left amplifier
                datanew=data[:,0:xsize-cover*2]
                m=datanew.shape[1]
                for i in range(0,m):
                    datanew[:,i]=datanew[:,i]-biasa

            if x0 >= bd: # all data on right amplifier
                datanew=data[:,0:xsize-cover*2]
                m=datanew.shape[1]
                for i in range(0,m):
                    datanew[:,i]=datanew[:,i]-biasb

            if xtest >= bd and x0 < bd:  #data on both amplifiers
                x1=int(bd-x0)
                dataa=data[:,0:x1]
                datab=data[:,x1:-cover*2]
                ma=dataa.shape[1]
                mb=datab.shape[1]
                for i in range(0,ma):
                    dataa[:,i]=dataa[:,i]-biasa
                for i in range(0,mb):
                    datab[:,i]=datab[:,i]-biasb
                # merge dataa and datab into single image
                datanew=np.hstack([dataa,datab])

        if namps > 2:
            sys.exit('Program does not yet deal with more than two overscan regions. Exiting.')

        # add info to header
        header['HISTORY'] = 'Overscan subtracted'
        header['DATE-OBS'] = str(header['DATE-BEG'])

        # write new fits file
        fits.writeto(ofile,datanew,header=header,overwrite=True)