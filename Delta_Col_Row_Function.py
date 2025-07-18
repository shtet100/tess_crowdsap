import lightkurve as lk
import numpy as np
import math
import PRF
import pandas as pd
# from tqdm import tqdm   Imported but not used
from lmfit import minimize, Parameters, fit_report
from astropy.coordinates import SkyCoord, Angle, Distance
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.time import Time
from astroquery.mast import Catalogs
import re
from pathlib import Path
import argparse
import os
import glob
import astropy
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


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


def rightflux(m):
    a = minus(10, m)  # Reference Magnitude: 10
    b = divide(a, 2.5)
    c = log(15000, 10)  # Reference Flux: 15,000
    d = add(b, c)
    Flux = power(10, d)
    return Flux


def lmfit_extract(fit_report):
    if not isinstance(fit_report, str):
        raise ValueError("Input must be a string.")

    pattern = r"del_col:\s*([+-]?\d+\.?\d*)\s*\+/-\s*([+-]?\d+\.?\d*)|del_row:\s*([+-]?\d+\.?\d*)\s*\+/-\s*([+-]?\d+\.?\d*)"

    # Finding all matches
    matches = re.findall(pattern, fit_report)

    # Flattening the list and removing empty strings
    matches_flattened = [
        value for match in matches for value in match if value]

    # Extracting the values and uncertainties
    del_col_value, del_col_uncertainty, del_row_value, del_row_uncertainty = matches_flattened

    return del_col_value, del_col_uncertainty, del_row_value, del_row_uncertainty


def get_args():
    parser = argparse.ArgumentParser(
        description='Process TIC ID & Sector for delta col and row calculation.')
    parser.add_argument('TIC_ID', type=int, help='TIC ID to process.')
    parser.add_argument('Sector', type=int, help='Sector number to process.')
    return parser.parse_args()


def delta_col_row(ID, Sector):

    filename = glob.glob(
        f'/shared_data/vanir/TESS/TP/sector{Sector:02}/*{ID}*')[0]
    tpf = lk.TessTargetPixelFile(filename)

    inaperture = np.where(tpf.pipeline_mask)

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
    ##########################################################################

    gaia_catalog = 'I/345/gaia2'

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

    ##########################################################################
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

    ##########################################################################

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

    ##########################################################################
    goodframes = np.where((tpf.quality == 0) &
                          np.isfinite(tpf.pos_corr1) &
                          np.isfinite(tpf.pos_corr2))[0]
    np.random.seed(tpf.targetid)  # Setting the random number seed
    randomid = np.random.choice(goodframes, 100, replace=False)

    FLUX = np.array(tpf[randomid].flux.value)
    FLUX_ERR = np.array(tpf[randomid].flux_err.value)
    posI = np.array(tpf.pos_corr1[randomid])
    posII = np.array(tpf.pos_corr2[randomid])
    ##########################################################################

    def star(stampsize, col, row, flux):
        star = flux*prf.locate(col, row, stampsize=tpf.shape[1:])
        return star

    def offset_stars(del_col, del_row, star_table, stampsize, pos1, pos2):
        model_images = np.zeros(stampsize)
        for i in range(len(star_table)):
            model_images += star(stampsize,
                                 star_table.iloc[i].x + del_col - pos1,
                                 star_table.iloc[i].y + del_row - pos2,
                                 star_table.iloc[i].Flux)
        return model_images

    def stack_images(del_col, del_row, star_table, stampsize, pos1_arr, pos2_arr):
        stack = np.zeros((len(pos1_arr), stampsize[0], stampsize[1]))
        for i in range(len(pos1_arr)):
            stack[i, :, :] = offset_stars(del_col, del_row, star_table, stampsize,
                                          pos1_arr[i], pos2_arr[i])
        return stack

    def residuals(params, star_table, pos1_arr, pos2_arr, flux, flux_err):
        stampsize = flux.shape[1:]
        del_col = params["del_col"]
        del_row = params["del_row"]
        model = stack_images(del_col, del_row, star_table,
                             stampsize, pos1_arr, pos2_arr)
        residuals = ((flux - model) / flux_err).flatten()
        return residuals
    ##########################################################################
    params = Parameters()
    params.add("del_col", value=0)
    params.add("del_row", value=0)
    output = minimize(residuals, params, args=(
        Merged, posI, posII, FLUX, FLUX_ERR))
    print(fit_report(output))

    lmfit_dataframe = dict(
        del_col_value=output.params["del_col"].value,
        del_row_value=output.params["del_row"].value
    )

    del_col_row_df = pd.DataFrame([lmfit_dataframe])

    # del_col_row_df.to_pickle("C:\\Users\\swany\\TIC " +
    # str(ID) + "\\TIC_" + str(ID) + "_" + str(sector) + "_del_col_row_df.pkl")

    del_col_row_df.to_pickle("~/mendel-nas1/NEW_PROJECT_Del_Col_Row" +
                             "/TIC_" + str(ID) + "_" + str(sector) + "_del_col_row_df.pkl")
    
    #del_col_row_df.to_pickle("~/mendel-nas1/NEW_PROJECT_Del_Col_Row/Trial_1_Del_Col_Row.pkl")

    return del_col_row_df


if __name__ == "__main__":
    args = get_args()
    delta_col_row(args.TIC_ID, args.Sector)
