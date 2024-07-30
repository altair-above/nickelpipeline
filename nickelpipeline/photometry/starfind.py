import numpy as np
import logging

from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Moffat2D
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.aperture import CircularAperture
from photutils.psf import IterativePSFPhotometry, make_psf_model
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.psf import IntegratedGaussianPRF, SourceGrouper

from pathlib import Path
from astropy.table import Table
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.stats import SigmaClip

from nickelpipeline.convenience.fits_class import Fits_Simple
from nickelpipeline.convenience.nickel_data import bad_columns, ccd_shape
from nickelpipeline.convenience.log import log_astropy_table

from nickelpipeline.psf_analysis.moffat.stamps import generate_stamps
from nickelpipeline.psf_analysis.moffat.fit_psf import fit_psf_single


logger = logging.getLogger(__name__)
np.set_printoptions(edgeitems=100)


def plot_sources(image, x, y, given_fwhm):
    logger.info(f'Image {image}')
    positions = np.transpose((x, y))
    apertures = CircularAperture(positions, r=2*given_fwhm)
    
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image.masked_array)
    cmap = plt.get_cmap()
    cmap.set_bad('r', alpha=0.5)

    plt.imshow(image.masked_array, origin='lower', vmin=vmin, vmax=vmax,
               cmap=cmap, interpolation='nearest')
    plt.colorbar()
    apertures.plot(color='r', lw=1.5, alpha=0.5)
    plt.show()


def analyze_sources(image, plot=False, verbose=True):
    
    default_fwhm=5.0
    thresh=10.0
    aper_size=8
    local_bkg_range=(20,30)
    
    if not isinstance(image, Fits_Simple):
        image = Fits_Simple(image)
    logger.debug(f"analyze_sources() called on image {image.filename}")
    
    sig2fwhm = np.sqrt(8*np.log(2))

    img = image.masked_array
    img.data[:,bad_columns] = 0
    
    #----------------------------------------------------------------------
    # Use a Moffat fit to find & fit initial sources
    #----------------------------------------------------------------------
    
    # Create output directories
    img_name = image.filename.split('.')[0]
    proc_dir = Path('.').resolve() / "proc_files"
    Path.mkdir(proc_dir, exist_ok=True)
    proc_subdir = proc_dir / 'circular'
    Path.mkdir(proc_subdir, exist_ok=True)
    base_parent = proc_subdir / img_name
    Path.mkdir(base_parent, exist_ok=True)
    base = proc_subdir / img_name / img_name
    
    # Generate stamps (image of sources) for image data
    source_data = generate_stamps([image], output_base=base, thresh=thresh)
    
    # Convert source data into Astropy table
    column_names = ['chip', 'id', 'xcentroid', 'ycentroid', 'bkg', 'kron_radius', 'raw_flux', 'flux', '?']
    sources = Table(source_data, names=column_names)
    logger.debug(f"Sources Found (Iter 1): \n{log_astropy_table(sources)}")
    
    # Fit PSF models and get source coordinates and parameters
    _, source_pars, _ = fit_psf_single(base, 1, fittype='circular')
    
    avg_par = np.mean(source_pars, axis=0)
    avg_fwhm = gamma_to_fwhm(avg_par[3], avg_par[4])
    logger.debug(f"Averaged-out Moffat fit parameters: {avg_par}")
    logger.debug(f"Averaged-out FWHM = {avg_fwhm}")

    # #----------------------------------------------------------------------
    # # Do a first source detection using the default FWHM
    # #----------------------------------------------------------------------
    # _, median, std = sigma_clipped_stats(img, sigma=3.)
    # starfind = IRAFStarFinder(fwhm=avg_fwhm, threshold=thresh*std,
    #                           minsep_fwhm=0.1, sky=0.0, peakmax=55000,)
    # sources = starfind(data=(img.data - median), mask=img.mask)
    # if sources is None:
    #     logger.info(f'Found {len(sources)} sources in {image}.')
    
    

    #----------------------------------------------------------------------
    # Attempt to improve the source detection by improving the FWHM estimate
    #----------------------------------------------------------------------
    thresh=10.0
    aper_size=avg_fwhm*1.8
    local_bkg_range=(3*avg_fwhm,6*avg_fwhm)
    win = int(np.ceil(2*avg_fwhm))
    if win % 2 == 0:
        win += 1
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(img)
    return std
    # Source finder
    # daofind = DAOStarFinder(fwhm=avg_fwhm, threshold=thresh*std,
    #                         min_separation=0.1*avg_fwhm, peakmax=55000)
    iraffind = IRAFStarFinder(fwhm=avg_fwhm, threshold=thresh*std, 
                              minsep_fwhm=0.1, peakmax=55000)
    grouper = SourceGrouper(min_separation=2*avg_fwhm)  # Grouping algorithm
    mmm_bkg = MMMBackground()   # Background-determining function
    local_bkg = LocalBackground(*local_bkg_range, mmm_bkg)
    fitter = LevMarLSQFitter()  # This is the optimization algorithm
    
    # This is the model of the PSF
    moffat_psf = Moffat2D(gamma=avg_par[3], alpha=avg_par[4])
    moffat_psf = make_psf_model(moffat_psf)
    
    # This is the object that performs the photometry
    phot = IterativePSFPhotometry(finder=iraffind, grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=moffat_psf,
                                  fitter=fitter, fit_shape=win, maxiters=6,
                                  fitter_maxiters=6, aperture_radius=aper_size)
    # This is actually when the fitting is done
    phot_data = phot(data=img.data, mask=img.mask,
                     init_params=Table(sources['xcentroid', 'ycentroid', 'flux'],
                                       names=('x_0', 'y_0', 'flux_0')))
    phot_data = filter_phot_data(phot_data, avg_fwhm)
    # integ = moffat_integral((phot_data['amplitude_2_fit']), phot_data['gamma_2_fit'], phot_data['alpha_2_fit'])
    # print(integ)
    # phot_data.add_column(np.array(integ), name='integral of moffat psf')
    # phot_data.add_column(np.array(integ*phot_data['amplitude_4_fit']), name='integral * amp_4')
    logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")
    plot_sources(image, phot_data['x_fit'], phot_data['y_fit'], avg_fwhm)
    return phot_data

    phot_data = filter_phot_data(phot_data)
    fwhm = np.median(np.abs(phot_data['sigma_fit']))*sig2fwhm
    
    #----------------------------------------------------------------------
    # Refit using the "improved" FWHM
    #----------------------------------------------------------------------
    iraffind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std,
                              minsep_fwhm=0.1, peakmax=55000)
    grouper = SourceGrouper(min_separation=2*fwhm)
    gaussian_prf = IntegratedGaussianPRF(sigma=fwhm/sig2fwhm)
    gaussian_prf.sigma.fixed = False
    phot = IterativePSFPhotometry(finder=iraffind,
                                  grouper=grouper,
                                  localbkg_estimator=local_bkg, psf_model=gaussian_prf,
                                  fitter=fitter, fit_shape=win, maxiters=2,
                                  fitter_maxiters=2, aperture_radius = aper_size)
    phot_data = phot(data=img.data, mask=img.mask,
                     init_params=Table([phot_data['x_fit'],
                                        phot_data['y_fit'],
                                        phot_data['flux_fit']],
                                        names=('x_0', 'y_0', 'flux_0')))
    
    logger.debug(f"Sources Found (Iter 2): \n{log_astropy_table(phot_data)}")
    #----------------------------------------------------------------------
    # Extract the source which_source, & calculate fwhm
    #----------------------------------------------------------------------
    
    phot_data = filter_phot_data(phot_data)
    psf_fwhm_median = np.median(phot_data['sigma_fit'])*sig2fwhm
    psf_fwhm_std = np.std(phot_data['sigma_fit']*sig2fwhm)

    if plot:
        plot_sources(image, phot_data['x_fit'], phot_data['y_fit'],fwhm)

    #----------------------------------------------------------------------
    # Sigma Clip PSF FWHMs
    #----------------------------------------------------------------------
    # (psf_fwhm_median, aper_fwhm_median, psf_fwhm_std, aper_fwhm_std)
    all_fwhms = np.array(phot_data['sigma_fit'])*sig2fwhm
    all_x = np.array(phot_data['x_fit'])
    all_y = np.array(phot_data['y_fit'])
    # Create a SigmaClip object and apply it to get a mask
    sigma_clip = SigmaClip(sigma=3, maxiters=5)
    masked_fwhms = sigma_clip(all_fwhms)

    # Apply the mask to the original data
    clipped_x = np.array(all_x)[~masked_fwhms.mask]
    clipped_y = np.array(all_y)[~masked_fwhms.mask]
    clipped_fwhms = np.array(all_fwhms)[~masked_fwhms.mask]
    
    if verbose:
        print("Number of sources removed =", len(all_x) - len(clipped_x))
        print(clipped_fwhms)
    
    #----------------------------------------------------------------------


def filter_phot_data(table, fwhm):
    """
    Removes all rows from the table with coordinates outside ccd_shape
    and with 'iter_detected' == 1.
    
    Parameters:
    table (astropy.table.Table): The input table.
    ccd_shape (tuple): A tuple (width, height) representing the shape of the CCD.
    
    Returns:
    astropy.table.Table: The filtered table.
    """
    # Create boolean masks for each condition
    indx1 = table['iter_detected'] == 1 #np.max(table['iter_detected'])
    indx2 = table['x_fit'] > 0
    indx3 = table['y_fit'] > 0
    indx4 = table['x_fit'] < ccd_shape[0]
    indx5 = table['y_fit'] < ccd_shape[1]
    
    i, j = np.meshgrid(np.arange(len(table)), np.arange(len(table)),
                            indexing='ij')
    dist = np.sqrt(np.square(table['x_fit'][:,None]
                             - table['x_fit'][None,:])
                + np.square(table['y_fit'][:,None]
                            - table['y_fit'][None,:]))
    indx6 = (dist < fwhm*1.7) & (i != j) & (j > i)
    indx6 = np.logical_not(np.any(indx6, axis=0))
    # logger.debug(indx6)
    logger.debug(f"{len(indx6)-sum(indx6)} sources removed for being too close")
    
    # Combine all masks
    combined_mask = indx2 & indx3 & indx4 & indx5 & indx6 #& indx1
    
    return table[combined_mask]



def fwhm_to_gamma(fwhm, alpha):
    """
    Convert full-width half-maximum (FWHM) to gamma.
    """
    return fwhm / 2 / np.sqrt(2**(1/alpha)-1)

def gamma_to_fwhm(gamma, alpha):
    """
    Convert gamma to full-width half-maximum (FWHM).
    """
    return 2 * gamma * np.sqrt(2**(1/alpha)-1)

def moffat_integral(amplitude, gamma, alpha):
    return amplitude * np.pi * gamma**2 / (alpha - 1)

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

# result = discrete_moffat_integral(0.0381, 4.776, 3.728)




#  id group_id iter_detected     local_bkg            x_init             y_init           flux_init            x_fit              y_fit             flux_fit      x_err y_err flux_err npixfit group_size         qfit                  cfit          flags
# --- -------- ------------- ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ----- ----- -------- ------- ---------- ------------------- ----------------------- -----
#   1        1             1  70.05004123610146  488.6875347091411  701.7996442666049  331453.0536528716 488.66306401659386  701.7639795699582 390950.98196788016   nan   nan      nan     121          1  0.1126783180551065  1.7258071977339836e-05     8
#   2        2             1  71.09214485126839  512.0238785511882  532.1458023589879 316567.42314040434  512.0298823750733  532.1473728776182  369319.4459945994   nan   nan      nan     121          1 0.11688675222314861  -0.0011680093879514848     8
#   3        3             1    65.460355519735  548.1828126794043  454.7533440701112  70982.13125649455   548.167426913451  454.7448889610261  83385.58703188505   nan   nan      nan     121          1 0.11626247448929171 -2.0994521632190126e-05     8
#   4        4             1  64.26834638547163 142.43099929536538 235.94758490415916 31226.109586897077 142.42318743941098 235.96661378709524  37045.79635834786   nan   nan      nan     121          1 0.13869964671340987    -0.00026100330273641     8
#   5        5             1  64.99179232143806 410.73815149498444  750.5850947011412 15115.450148730208  410.6863855666373   750.544672048804   18056.5557633761   nan   nan      nan     121          1  0.1792391190283525   0.0014336892384057202     8
#   6        6             1 63.797595747332934 26.177470236292777   746.878110861582 10915.880133687422 26.157540874545347  746.8754923014287 12937.310639323523   nan   nan      nan     121          1 0.18944212748047506   0.0027346783865799438     8
#   7        7             1  65.33757163835386 462.50789955902724  613.8180928363782  8756.327798675902 462.49847127856896  613.8355749941371 10159.350097242503   nan   nan      nan     121          1 0.19563253438801873   0.0026244216160831173     8
#   8        8             1 62.436254435472705  282.8869183279999  837.1287666667723  7453.642689439557 282.88103333352035  837.1233644892494  8315.483724198144   nan   nan      nan     121          1  0.2444641146973336    0.003331707287302339     8
#   9        9             1  70.47859927762912  978.7225812627422 163.70136453578496  7325.348656067405  978.7026278767696 163.68473342754953  8200.181220778348   nan   nan      nan     121          1 0.21449325369159833  -0.0013847985768003297     8
#  10       10             1 63.491011291378285  727.5280595360816  994.2473710056231  5900.791196682215  727.5368745285163  994.3029683413203  6876.709274109911   nan   nan      nan     121          1  0.2596364312107832  -0.0024707279157227466     8
#  11       11             1  63.82050050630755  670.9383568480662  867.1556313527055  5840.191353561305  670.9658094684517  867.1565265397356  7114.615438248304   nan   nan      nan     121          1  0.2571734076757932  -0.0029075899888694238     8
#  12       12             1  64.30061081750844 192.65876732551519  706.0046836488069  5684.718620020155 192.65096793619503  705.9797272872454 6580.8245310038155   nan   nan      nan     121          1  0.2823926690849965   0.0020504356396439543     8
#  13       13             1  63.74240613777245  405.4769334223915   985.074114372971   4821.88714662583  405.4754022369425  985.1144723434245 5382.6906524321985   nan   nan      nan     121          1  0.3151841410404072  -0.0014203938631088167     8
#  14       14             1  63.83021836777138  77.10060070863177  748.1492769424002  4457.947683589517  77.15385496089216  748.1809452847456  5326.068749331816   nan   nan      nan     121          1 0.32690982336395796  -0.0009477425256572461     8
#  15       15             1  65.20844716130173 454.00386550934877  668.3492076600942  4155.213687305707 454.02069836728526  668.3631085478489   5020.65339610204   nan   nan      nan     121          1  0.3233042186411397  -0.0008433210056546316     8
#  16       16             1  64.82722172205604  596.9603355460376  818.0604840078721 4060.8503771140495  596.9605077704834  818.0641184295605  4735.424631859551   nan   nan      nan     121          1  0.3828338209559315    -0.01059119676595122     8
#  17       17             1   64.7372543000854  292.2953037144781  773.3311780728965 3858.8854580715306  292.3037013007362  773.3456300144239  4648.307768129733   nan   nan      nan     121          1  0.3918375108009163   -0.005516413177338692     8
#  18       18             1  65.99957746450889  945.0212814885684  136.6001603629007  3737.644052883272  944.9843530761275 136.60994902996643  4278.838159667659   nan   nan      nan     121          1 0.41115215431252017   -0.003922992882918311     8
#  19       19             1  63.36553371635975  866.4516481009804  854.5726397476986 3696.4321691446685  866.4462404066039  854.5949784248583  4551.369236801533   nan   nan      nan     121          1  0.3518468697208266  -0.0021512754433238553     8
#  20       20             1 63.272872397515755 416.30404288536045  829.5953492631206 3582.4415884867512 416.36639624203343   829.604423663198  3980.703723628782   nan   nan      nan     121          1   0.399109372647712   -0.006631860216898196     8
#  21       21             1 62.596033338515056  144.1253369687705  893.1025191390751 3489.6235880096647 144.15769075138965   893.110383541747  3920.749140238255   nan   nan      nan     121          1   0.426618206459295   0.0031688052788478196     8
#  22       22             1  64.76231290731596 206.33082265523362  377.7315098378392  3377.972741547691 206.29891574304958 377.71742496923844  3652.716400888628   nan   nan      nan     121          1  0.4607053153053399    -0.00466033663210315     8
#  23       23             1  62.68510675108253  642.4910677287845  788.6150526793197    3317.5605794335   642.451062149762  788.6128479800868 3931.2915473353096   nan   nan      nan     121          1  0.4307384997403343  -0.0016719980077396432     8
#  24       24             1  65.80408059452466 181.32368882272837  606.8271920294834   3317.23913131363  181.2870033117175  606.8404782506632  3536.182961893907   nan   nan      nan     121          1  0.4382076970068219   -0.000919373491438169     8
#  25       25             1  64.60152635402684   519.743166589564  477.2407542033972 3263.4250629439202  519.7547891640078  477.2358154297432 3862.3462621456947   nan   nan      nan     121          1  0.3698364863175289  -0.0010130305798018038     8
#  26       26             1  62.67974031857685  505.1326692704409  980.9086589848903  3192.163844895792  505.1057527491518  980.9052939836076  3765.016048041287   nan   nan      nan     121          1 0.41640136639378295    0.007111656929217706     8
#  27       27             1  64.10684309282956  384.0875925266794 507.58843267399527 3114.0706549146594  384.0441752272848  507.5768413372942  3869.692961182929   nan   nan      nan     121          1 0.40941478032874207    0.004840265300174468     8
#  28       28             1 63.301978976001806  570.7247547093649  986.6872449774351  3065.134508429057  570.7367662059868  986.6388587659019 3734.6076942677596   nan   nan      nan     121          1  0.4476421135273009    0.004450432633985266     8
#  29       29             1  64.20045453923376  552.6462716608188  587.1129334587379   3057.02756643138  552.6638759871009  587.1474421240492 3569.3608066962033   nan   nan      nan     121          1  0.4277597582795009   0.0032097250653109915     8
#  30       30             1  63.96197419616175  383.6683961565374 265.97541178559976  3038.499045880461  383.6446613347026  265.9544258342819  3713.149230712996   nan   nan      nan     121          1 0.41379525789179117   0.0003805177486266096     8
#  31       31             1  61.76733432008329 50.865976743035226 19.759062932850483 3011.0463597808466  50.84542956954737 19.693942204829952 3340.9755176266895   nan   nan      nan     121          1 0.48092709088563007 -0.00018536017624205356     8
#  32       32             1  63.81053788916293  763.4045561905971  251.4521488744227 2955.5439523779223  763.3799387611514 251.48358420013983 3619.2964887479793   nan   nan      nan     121          1  0.3900125176666346   -0.007717331200007763     8
#  33       33             1   64.2262780421502  739.4613604789183   780.063726262098  2842.181839096545  739.4904546196404  780.0937539961022 3221.4342082819453   nan   nan      nan     121          1  0.5093770053671868   -0.003971969187395461     8
#  34       34             1  62.21743181072836 199.40119565452122  838.8504903793848 2673.7739020455256  199.3827092802994  838.8144621747894 2939.4912975529287   nan   nan      nan     121          1 0.45073656725936867    0.007039492483887478     8
#  35       35             1   63.0938704697939  730.6100313553112  715.6579007731144 2574.6012075346853    730.59332733551  715.6870175375128 2926.2860106206576   nan   nan      nan     121          1  0.5311714526507304    0.005085607911572502     8
#  36       36             1    65.831342900388   684.774637031213  689.2632946607768 2498.2237849697403  684.7750708791499  689.2393329162901  2653.965521795226   nan   nan      nan     121          1  0.5125462637623931    -0.01214749322493835     8
#  37       37             1  64.28805983142487  447.5213736313217  755.6308739354905  2441.691844867834  447.5645100347128  755.6677291211398  2953.760850546507   nan   nan      nan     121          1  0.5293422227202725   0.0006963911015917881     8
#  38       38             1  66.22459111751397  940.4189708088877  786.7897153173211 2326.0772402627736  940.3972007284771   786.807099511359  2838.837787728283   nan   nan      nan     121          1  0.5031222428479983   0.0022179135456748783     8
#  39       39             1  64.49427230267327  882.0548633584597  772.0978353356577 2231.0466999447785  882.1064176019903  772.1083631667099 2758.5372196213602   nan   nan      nan     121          1  0.5100936148758246  -0.0009035028274030155     8
#  40       40             1  64.19377737475003  83.21009119963662  630.1443390962953 2112.5813762124485   83.1767117155572  630.1737401855827 2374.0904359712267   nan   nan      nan     121          1  0.6638979625158689    0.010034085431358416     8
#  41       41             1  64.62733772903063   28.2951331621426  785.9526139345361 2095.4380148007863 28.304136094481375  785.9630602234063   2276.20326468878   nan   nan      nan     121          1  0.6493091362421894   0.0030170911722817413     8
#  42       42             2  64.21370723638896 139.91315613731265 235.59946305934926 15555.159358382842  139.7487744425312 235.36567426115397 1049.5430701206024   nan   nan      nan     121          2   4.547744712745378     0.09836508902749187     8
#  43       42             2  63.91950556245311 145.13280755430492 237.94160125798942 15703.127715393051 145.40312741295278 237.95546333034102 2253.0289543510153   nan   nan      nan     121          2  2.0660773970915214    0.013889886638847328     8
#  44       43             2  65.49729910638271  228.9195417876045 325.01841985532064  41291.62140639078 228.84970875966476  324.9871223669143  27269.94413104964   nan   nan      nan     119          1   0.153115374372935  -6.994230891801122e-05     9
#  45       44             2  65.99658120265343  550.2189602506677 448.81001159997805 20290.865202516103  550.4246169477314  448.5368183551312  2270.332455819447   nan   nan      nan     121          3  2.3636807641063693      0.0322891322970656     8
#  46       44             2  66.26844709174398  545.7226602141598   454.657571286618 18714.118198131735  545.2611319131591  454.2208962586662  2570.058392990155   nan   nan      nan     121          3  3.7943166787526224     0.11496394416756948     8
#  47       44             2  65.10912589549295  550.5695582777751  456.1828510032518 19090.511540999705  551.0808192336784  456.3873819318128 3821.7993740954644   nan   nan      nan     121          3  2.4229948529258265    0.008497048519579385     8
#  48       45             2  70.94621293025395  509.7004577110158  531.8210113962418  37354.31052218794  508.6025116882808  531.1364327020934 11689.039252001734   nan   nan      nan     121          2  3.4184508224407497     0.11395088182958821     8
#  49       45             2  70.66150931890877  514.5763950609105  533.2827415131494  38357.86892626424   515.669868041606  533.7210611522287 18737.319940008645   nan   nan      nan     121          2  1.9895923896875813    0.022637705035377678     8
#  50       46             2  68.26217386318248  780.6655821955263   556.097260036406 216528.96221648582  780.5012764722816  556.2506809908036 235787.49198532652   nan   nan      nan      99          1 0.10003906396444107  -0.0012764427094491594     9
#  51       47             2   75.9370746624704   1010.01814504686  569.8107754237208  47575.45398118863 1010.0471773181968  569.6194116423812  30738.19962424535   nan   nan      nan     121          1 0.14559792662253737  0.00014651436143111067     8
#  52       48             2  69.35262510949946  491.0859267504429  696.4117609380464  41165.06148005284 491.36707952777954  695.5335392205683  9970.036022120541   nan   nan      nan     121          3  2.1747711482164744     0.02840752131153059     8
#  53       48             2  69.57725906727342 485.92395487161383  700.9897435522013  35633.01558729552  484.9662704960009  700.6281963516898  17050.08104317128   nan   nan      nan     121          3   2.475463514147672    0.040465259393819855     8
#  54       48             2  70.19977143878674 491.21228189279935   703.149312148064  33731.84473667344  492.3670976406316  703.6134141038589 14995.462070715002   nan   nan      nan     121          3  2.6938625584394877    0.055960653059889426     8
#  56       49             2   65.1947489329543 413.19402753068914  752.0847321479096 14611.712454717508  413.2817608989215  752.0799697211983  1210.765713445222   nan   nan      nan     121          2    2.59088339291441     0.03230275254345077     8
#  57       50             2  63.23293411475697  683.2746843673708  881.0151579116933 17642.071408808286   683.736089347564    881.05926189914 4095.5426289668394   nan   nan      nan     121          1  0.3870245665936145   -0.001917095647653437     8
#  58       51             2 63.637218511795325  791.2107796604497  885.0672549942917  36594.15656902699  791.5438230436527  885.1924091094528 23553.326792077034   nan   nan      nan     121          1  0.1534319327571238   3.350282161580755e-05     8
#  59       52             2  79.50701209234958 1015.8940462643686 1021.6229608499111 15904.624077424533 1015.6452328927173 1021.6032035129762  1606.976155789258   nan   nan      nan      77          2  1.9615655454321708     0.02704147859794509     9
#  60       52             2  80.26525358127924 1020.0480066286358 1021.6411036362689 11548.010477454915  1020.034141764725 1021.6826283006028 1574.9223018718294   nan   nan      nan      63          2  1.3922004545168456    0.025004469461446778     9
#  64       54             3   65.3756455857368 231.36382966972675  326.2752573854915 15492.619314621023 231.52673701510065  326.3847431076161 1262.3340590396347   nan   nan      nan     119          2   3.011091831900277     0.04178617115511123     9
#  68       56             3  69.71405598458159  519.1371736543512  531.7596165706855  17262.87064634592  519.9616402536893  531.0294006698873 2880.7169767799624   nan   nan      nan     121          3   4.956524639932914    0.030609743876255355     8
#  70       57             3  68.06863784914714  777.8311276399107  555.8033709351121  24604.19036455474  775.6403701250373  554.6447672882349  9192.851806782008   nan   nan      nan     110          1  2.4368876287411814      0.0901788270890951     9
#  71       58             3   75.6735163406469 1007.6011663469616  568.9743839889035 17633.704031522793 1007.2885742731199  568.8471792120116 1089.2764392902682   nan   nan      nan     121          2  3.7615990078174866     0.06546690920448556     8
#  72       58             3   76.3933965474161 1013.0780713183918  571.0803669063841  18546.93250263201  1013.186015700715  571.0867393642784  2151.691574028664   nan   nan      nan     121          2  1.6861473164300174     0.02196970912158994     8
#  73       59             3  69.90708755242389 486.64069933685676  701.6129643336275  4681.356134858831  477.7694168948121  695.2880869933367 -8608.351069666147   nan   nan      nan     121          2 -5.0952507785789125    -0.12337314777152182    12
#  76       61             3   63.8084086422823  788.8026036567798  884.5608180206409 13132.224259933208  788.5624183549394   884.345042456528  986.4751547012766   nan   nan      nan     110          2  3.3717590126384467     0.09342859277542141     9
#  77       61             3  63.71060269119728   794.131048654834  886.3231486364081 14988.379494931161  794.2510726198846  886.4365515451096 1386.0950429389432   nan   nan      nan     121          2   2.472093161214515    0.034973557084329136     8
#  86       66             4  67.48724176151629  773.8203906034272  557.2723444305857  12952.09277504424  772.9651756510793  557.9033785272436 -785.8369287183177   nan   nan      nan     121          2 -18.274617891181297    -0.04815563432742662    12

# 18:09:40 - INFO - Image d1040.fits (110_232 - R)