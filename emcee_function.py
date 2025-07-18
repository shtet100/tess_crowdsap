from astroquery.vizier import Vizier
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import PRF
from astropy.coordinates import SkyCoord, Angle, Distance
from astropy.time import Time
import astropy.units as u
from astroquery.mast import Catalogs
import pandas as pd
from tqdm import tqdm
import math
import seaborn as sns
import emcee
import corner
import argparse
from astroquery.gaia import Gaia
import glob
import os
import sys

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


# Defining mathematical operations in simple names

def add(num1, num2):
    return num1 + num2


def minus(num1, num2):
    return num1 - num2


def divide(num1, num2):
    return num1 / num2


def power(num1, num2):
    return num1 ** num2


def log(num, base):
    return math.log(num, base)


def rightflux(mag):
    a = minus(10, mag)  # Reference Magnitude: 10
    b = divide(a, 2.5)
    c = log(15000, 10)  # Reference Flux: 15,000
    d = add(b, c)
    Flux = power(10, d)
    return Flux


def derivativeF(m):
    a = -6 * np.log(10)  # -6 x log(10)
    b = (2/5) * minus(10, m)  # 2/5 x (10-m)
    c = add(b, 3)  # [2/5 x (10-m)] + 3
    d = power(10, c)  # 10 ^ [ [2/5 x (10-m)] + 3 ]
    derivative = a * d  # -6 x log(10) x 10 ^ [ [2/5 x (10-m)] + 3 ]
    return derivative


def directorymaker(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass


def get_args():
    parser = argparse.ArgumentParser(
        description='Process TIC ID & Sector for delta col and row calculation.')
    parser.add_argument('TIC_ID', type=int, help='TIC ID to process.')
    parser.add_argument('Sector', type=int, help='Sector number to process.')
    return parser.parse_args()


gaia_catalog = 'I/345/gaia2'
magnitude_limit = 20  # 20
pix_scale = 21.0  # arcseconds / pixel
Vizier.ROW_LIMIT = -1
bright_mag = 10
home_del_col_row_path = "del_col_row_selection"
home_emcee_path = "emcee_selection"


def emcee_gather(ID, sector):

    filename = glob.glob(
        f'/shared_data/vanir/TESS/TP/sector{sector:02}/*{ID}*')[0]
    tpf = lk.TessTargetPixelFile(filename)
    # TPF Downloading...

    inaperture = np.where(tpf.pipeline_mask)
    # Downloading TPF with its TIC, author, mission, and sector specifications

    # Selecting random "good" frame number as sample
    frameno = np.random.choice(np.where(tpf.quality == 0)[
                               0], 1, replace=False)[0]

    sample_flux = tpf.flux[frameno].value
    flux_err = tpf.flux_err[frameno].value

    cam = tpf.camera
    ccd = tpf.ccd
    sector = tpf.sector
    stampsize = tpf.shape[1:]
    colnum = tpf.column + stampsize[1]/2 - 0.5
    rownum = tpf.row + stampsize[0]/2 - 0.5

    prfdir = "/shared_data/vanir/TESS/PRF/s0004"

    if sector < 4:
        prfdir = "/shared_data/vanir/TESS/PRF/s0001"

    prf = PRF.TESS_PRF(cam, ccd, sector, colnum, rownum, localdatadir=prfdir)

    del_col_row_df = pd.read_pickle("~/mendel-nas1/" + home_del_col_row_path + "_sec" + str(sector) + "/" +
                                    "/TIC_" + str(ID) + "_" + str(sector) + "_del_col_row_df.pkl")

    ###########################################################

    # Gaia Figure Elements
    magnitude_limit = 20
    # Positions of the Gaia sources
    c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
    # Pixel scale for query size
    pix_scale = 21.0
    # Querying with a diameter as the radius, overfilling by 2x
    Vizier.ROW_LIMIT = -1
    try:
        result_gaia = Vizier.query_region(c1, catalog=[gaia_catalog], radius=Angle(
            np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
    except:
        result_gaia = Vizier.query_region(c1, catalog=[gaia_catalog], radius=Angle(
            np.max(tpf.shape[1:]) * pix_scale, "arcsec"), cache=False)

    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError(
        'No sources found brighter than {:0.1f}'.format(magnitude_limit))
    if result_gaia is None:
        raise no_targets_found_message
    elif len(result_gaia) == 0:
        raise too_few_found_message
    # Recording reference epoch for Gaia source positions

    gaiarefepoch = float(str(result_gaia[0].info).split(
        'at Ep=')[1][:6])  # From column description

    result_gaia = result_gaia[gaia_catalog].to_pandas()
    result_gaia = result_gaia[result_gaia.Gmag < magnitude_limit]
    if len(result_gaia) == 0:
        raise no_targets_found_message

    # Convert parallaxes to distances
    distance = 1000*u.pc/result_gaia['Plx'].values
    distance[distance < 0] = 100000.*u.pc  # replace negative values
    distance[np.where(np.isnan(distance))[0]] = 100000. * \
        u.pc  # replace nan values

    # Propagating star positions by proper motions
    c = SkyCoord(ra=result_gaia['RA_ICRS'].values * u.deg,
                 dec=result_gaia['DE_ICRS'].values * u.deg,
                 distance=distance,
                 # distance=Distance(parallax=result_gaia['Plx'].values * u.mas),
                 pm_ra_cosdec=result_gaia['pmRA'].values * u.mas/u.yr,
                 pm_dec=result_gaia['pmDE'].values * u.mas/u.yr,
                 obstime=Time(gaiarefepoch, format='decimalyear', scale='utc'))

    # Converting to Tess' reference time
    tess_time = tpf.time[int(len(tpf)/2)]
    c_tess = c.apply_space_motion(tess_time)

    # Converting stars' positions back to the position from Tess's reference time
    radecs = np.vstack([c_tess.ra, c_tess.dec]).T
    coords = tpf.wcs.all_world2pix(radecs, 0)  # Back to pixel's version

    # Sizing the points by their Gaia magnitude
    sizes = 64.0 / 2**(result_gaia['Gmag']/5.0)

    # Gathering the data
    gaiadata = dict(source=result_gaia['Source'].astype(str),
                    ra=c_tess.ra,
                    dec=c_tess.dec,
                    Gmag=result_gaia['Gmag'],
                    distance=distance,
                    x=coords[:, 0],
                    y=coords[:, 1],
                    size=sizes)

    ##############################################################
    # TESS Input Catalog Figure Elements
    rad = Angle(np.max(tpf.shape[1:]) * pix_scale * u.arcsec)
    result_tic = Catalogs.query_region(c1, radius=rad, catalog="TIC")

    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError(
        'No sources found brighter than {:0.1f}'.format(magnitude_limit))
    if result_tic is None:
        raise no_targets_found_message
    elif len(result_tic) == 0:
        raise too_few_found_message

    # Sizing the points by their Tess magnitude
    sizes = 64.0 / 2**(result_tic['Tmag']/5.0)
    # one_over_parallax = 1.0 / (result['Plx']/1000.)
    TIC = dict(source=result_tic['GAIA'].astype(str),
               TICID=result_tic['ID'].astype(str),
               Gmag=result_tic['GAIAmag'],
               Tmag=result_tic['Tmag'],
               e_Tmag=result_tic['e_Tmag'],
               objtype=result_tic["objType"]
               )
    
    ###########################################################
    # Extra Check for Mismatch between Gaia Query and TIC Query based on a Bright Magnitude
    if np.size(np.where(result_tic['Tmag'] < bright_mag)) != np.size(np.where(result_gaia['Gmag'] < bright_mag)):
        # Saving the not-working TIC ID and Sector
        error_star = np.array([ID, sector])
        np.save("/home/shtet/mendel-nas1/emcee_error/" +
                "TIC_" + str(ID) + "_" + str(sector) + "_ID_sector.npy", error_star)
        raise ValueError("Not Able to Move Forward!")
        sys.exit()

    ###########################################################
    Gaiadf = pd.DataFrame(gaiadata)  # gaiadata as pandas dataframe
    GDF = Gaiadf[(Gaiadf.x > -3) & (Gaiadf.y > -3) & (Gaiadf.x < tpf.shape[1] + 2) &
                 (Gaiadf.y < tpf.shape[2] + 2) & (Gaiadf.Gmag < magnitude_limit)]
    # Putting restraints on the pixel coordinates and magnitude
    GDF = GDF.sort_values(by=['Gmag'])  # Sort by magnitude
    TICdf = pd.DataFrame(TIC)  # TIC as pandas dataframe
    # Merging gaiadata and TIC as "Merged"
    Merged = pd.merge(GDF, TICdf, on="source")
    Fluxlist = [rightflux(Merged.iloc[i]['Tmag']) for i in range(len(Merged))]
    Merged['Flux'] = Fluxlist

    ###########################################################
    # Yielding magnitude errors into the Merged dataframe
    
    mag_lim = 18

    sigma2 = []  # Magnitude Errors
    for i in range(len(Merged)):
        s = (derivativeF(Merged.iloc[i]['Tmag'])
             * (Merged.iloc[i]['e_Tmag'])) ** 2
        if np.isnan(s):
            s = (derivativeF(Merged.iloc[i]['Tmag'])
                 * (0.1)) ** 2  # Assume large magnitude error if not available
        sigma2.append(s)


    Merged['Sigma2'] = sigma2
    Merged['Sigma'] = np.sqrt(sigma2)

    # Search for duplicates task
    condition = Merged.duplicated(subset='source', keep=False) & ~(
        Merged['TICID'] == str(tpf.targetid))
    Merged.drop_duplicates(subset='source', keep='first',
                           inplace=True, ignore_index=True)
    
    if len(Merged) > 50:
        np.savetxt(f'TIC{ID}_sector{sector}_failed', np.array([len(Merged)]))
        sys.exit()

    target_mag = tpf.header['TESSMAG']

    Merged = Merged.nsmallest(50, 'Tmag')

    target_index = np.where(Merged.TICID == str(tpf.targetid))[0][0]

    Merged = Merged[Merged['Tmag'] <= mag_lim]

    Merged = Merged.reset_index(drop=True)


    ###########################################################
    # Selecting 100 random good tpf frames
    goodframes = np.where((tpf.quality == 0) &
                          np.isfinite(tpf.pos_corr1) &
                          np.isfinite(tpf.pos_corr2))[0]
    np.random.seed(tpf.targetid)  # Setting the random number seed
    randomid = np.random.choice(goodframes, 100, replace=False)  # Random ID

    # 100 "Good" Random TPF Flux Arrays
    FLUX = np.array(tpf[randomid].flux.value)
    FLUX_ERR = np.array(tpf[randomid].flux_err.value)  # Errors along with it
    BKG_FLUX = np.array(tpf[randomid].flux_bkg.value)  # Bkg Flux
    BKG_FLUX_ERR = np.array(tpf[randomid].flux_bkg_err.value)  # Bkg Flux Error
    posI = np.array(tpf.pos_corr1[randomid])  # 100 "Good" PosCorr I values
    posII = np.array(tpf.pos_corr2[randomid])  # 100 "Good" PosCorr II values
    bkg_flux = tpf.flux_bkg[randomid].value.flatten()

    ##########################################################

    del_column = float(del_col_row_df.del_col_value)
    del_row = float(del_col_row_df.del_row_value)

    ##########################################################
    # Putting 100 "good" quality frames of the tpf into the stars_bin

    BUCKET = np.zeros((len(FLUX), len(Merged), tpf.shape[1], tpf.shape[2]))
    for i in tqdm(range(len(FLUX))):
        bucket = np.zeros((len(Merged), tpf.shape[1], tpf.shape[2]))
        for j in range(len(Merged)):
            staritem = prf.locate(
                Merged.iloc[j]['x'] - posI[i] + del_column,
                Merged.iloc[j]['y'] - posII[i] + del_row,
                stampsize=tpf.shape[1:])
            bucket[j, :, :] = staritem
        BUCKET[i, :, :, :] = bucket
    stars_bin = np.array(BUCKET)

    ##########################################################
    # Setting up weights to obtain weighted average flux and its error

    weights_flux = 1/(FLUX_ERR**2)
    weights_bkg_flux = 1/(BKG_FLUX_ERR**2)
    weighted_avg_flux = np.average(FLUX, weights=weights_flux, axis=0)
    weighted_avg_flux_err = 1/np.sqrt(np.sum(weights_flux, axis=0))
    weighted_avg_bkg_flux = np.average(
        BKG_FLUX, weights=weights_bkg_flux, axis=0)
    weighted_avg_bkg_flux_err = 1/np.sqrt(np.sum(weights_bkg_flux, axis=0))
    avg_prf = np.average(stars_bin, axis=0)  # Average PRF map for stars

    #### Figure B as the plot for the weighted_avg_flux ####
    figB, ax = plt.subplots()
    im = ax.imshow(weighted_avg_flux, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    ax.set_title('Weighted Average Flux Image')
    cbar.set_label("Weighted Average Flux (e/s)")

    # directorymaker("~/mendel-nas1/emcee_output")

    plt.savefig(
        "/home/shtet/mendel-nas1/" + home_emcee_path + "_sec" + str(sector) + "/" +
        "TIC_" + str(ID) + "_" + str(sector) + "_weighted_avg_flux.png")

    ##########################################################
    # emcee functions

    def log_likelihood(theta, flux_meas, flux_err, bkg_flux):
        bkg_offset = theta[-3]
        zpe = theta[-2]
        flux_theta = theta[:-3]*(10**(zpe/2.5))
        log_f = theta[-1]
        model = flux_theta[:, None, None] * avg_prf
        Model = np.sum(model, axis=0).flatten() + bkg_offset
        sigma2 = flux_err ** 2 + \
            (((flux_meas) + (bkg_flux))**2) * np.exp(2 * log_f)
        return -0.5 * np.sum((flux_meas - Model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior(theta):
        flux_theta = theta[:-3]
        log_f = theta[-1]  # The last element
        zpe = theta[-2]  # Second to last
        bkg_offset = theta[-3]  # Third to last
        result = 0
        if np.any(flux_theta < 0):
            return -np.inf
        if log_f > 1:
            return -np.inf
        for i in range(len(Merged)):
            sigma2 = Merged['Sigma2'][i]
            priorres = -0.5 * np.sum((Merged['Flux'][i] - flux_theta[i]) ** 2 /
                                     sigma2 + np.log(sigma2))
            result += priorres
        result += -0.5 * (zpe**2/(0.05**2)) + np.log(0.05**2)  # Prior zpe
        result += -0.5 * (bkg_offset**2/(10**2)) + \
            np.log(10**2)  # Prior bkg_offset
        return result

    def log_probability(theta, flux_meas, flux_err, bkg_flux):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, flux_meas, flux_err, bkg_flux)

    ##########################################################
    soln = np.array([rightflux(Merged.iloc[i]['Tmag'])  # Setting prior fluxes from Merged fluxes
                    for i in range(len(Merged))])
    soln = np.append(soln, 0)  # Initial guess of background error offset
    soln = np.append(soln, 0)  # Initial guess of zpe
    soln = np.append(soln, -3)  # Initial guess of log_f

    # Setting walkers' starting position
    nwalkers = len(Merged)*4
    pos = soln + np.zeros((nwalkers, len(soln)))
    # Spread to the walkers
    pos[:, :-3] += Merged["Sigma"].values * \
        np.random.randn(nwalkers, len(soln)-3)
    pos[:, -1] += np.random.randn(nwalkers) * .1  # Spread of log_f
    pos[:, -2] += np.random.randn(nwalkers) * .01  # Spread of zpe
    # Spread of background error offset
    pos[:, -3] += np.random.randn(nwalkers) * .01

    nwalkers, ndim = pos.shape
    
    dtype = [("log_prior", float), ("mean", float)]

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        moves=[(emcee.moves.DEMove(), 0.8),
               (emcee.moves.DESnookerMove(), 0.2),],
        args=(weighted_avg_flux.flatten(),
              weighted_avg_flux_err.flatten(),
              weighted_avg_bkg_flux.flatten()),
        blobs_dtype = dtype
    )

    thin_by = 5
    initial_step = 300

    sampler.run_mcmc(pos, initial_step, thin_by=thin_by,
                     progress=True, store=True)

    samples = sampler.get_chain(thin=1)

    tau = np.nanmax(emcee.autocorr.integrated_time(samples, quiet=True))  # Tau
    more_step = int(np.ceil(50*tau*1.2 - initial_step))  # Extra steps to run

    while more_step > 0:
        sampler.run_mcmc(
            None, more_step, thin_by=thin_by, progress=True, store=True)
        samples = sampler.get_chain(thin=1)
        tau = np.nanmax(emcee.autocorr.integrated_time(samples, quiet=True))
        if samples.shape[0] > 50*tau:  # Long Enough
            more_step = -1
        else:
            more_step = int(np.ceil(50*tau*1.2 - samples.shape[0]))

    samples = sampler.get_chain(thin=1)

    # Saving samples
    np.save("/home/shtet/mendel-nas1/" + home_emcee_path + "_sec" + str(sector) + "/" +
            "TIC_" + str(ID) + "_" + str(sector) + "_samples.npy", samples)

    ###########################################################
    # Corner Plot Saving

    flat_samples = sampler.get_chain(flat=True)
    num_params = samples.shape[-1]

    labels = [
        f"Flux {i + 1}" for i in range(num_params - 3)] + ["bkg_offset", "zpe", "log_f"]
    corner_fig = corner.corner(
        samples.reshape(-1, samples.shape[-1]), labels=labels)
    plt.savefig(
        "/home/shtet/mendel-nas1/" + home_emcee_path + "_sec" + str(sector) + "/" +
        "TIC_" + str(ID) + "_" + str(sector) + "_corner_plot.png")
    

if __name__ == "__main__":
    args = get_args()
    emcee_gather(args.TIC_ID, args.Sector)
