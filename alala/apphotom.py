#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Sun Nov 24 21:42:55 2019
.. @author: Nicholas Vieira
.. @apphotom.py

CONTENTS:
    - :func:`image2xy`: Make a list of sources with `astrometry.net`
    - :func:`imsegm_make_source_mask`: Make a mask of sources using image 
      segmentation
    - :func:`error_array`: Compute the combination of the Gaussian + 
      Poissonian error in the image
    - :func:`aperture_photom`: Perform aperture photometry on an image 
    - :func:`limiting_magnitude`: Get the limiting magnitude of an image 
    
NON-STANDARD PYTHON DEPENDENCIES:
    - `astropy`
    - `photutils`

NON-PYTHON DEPENDENCIES:
    - `astrometry.net`

"""

import re
from subprocess import run
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats, SigmaClip

from photutils import (Background2D, MMMBackground,
                       make_source_mask, detect_sources,
                       SkyCircularAperture, aperture_photometry,
                       SkyCircularAnnulus)
from photutils.utils import calc_total_error

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)




def image2xy(image_file, astrom_sigma=5.0, psf_sigma=5.0, alim=10000,
             write=False, output=None):
    """Use astrometry.net's `image2xy` to detect sources in an image.
    
    Arguments
    ---------
    image_file : str
        Filename for **NOT background-subtracted** image
    astrom_sigma : float, optional
        Sigma threshold for astrometry.net source detection image (default 5.0)
    psf_sigma : float, optional
        Sigma of the Gaussian PSF of the image (default 5.0)
    alim : int, optional
        Maximum allowed source area, in pixels**2, beyond which sources will 
        be deblended (default 10000)
    write : bool, optional
        Write the output list of sources? (default False)
    output : str, optional
        Name for output source list file (default set by function)
        
    Returns
    -------
    astropy.io.fits.fitsrec.FITS_rec
        List of sources
        
    Notes
    -----
    Can write sources to a list of sources and/or a file to be used as a 
    source mask in aperture photometry.

    """

    if not(output):
        output = image_file.replace(".fits", ".xy.fits")    
    
    # -O --> overwrite
    # -p <astrom_sigma> --> signficance
    # -w <psf_sigma> --> estimated PSF sigma 
    # -m <alim> --> max object size for deblending is <alim> 
    # -o <output> --> name for output source list
    options = f" -O -p {astrom_sigma} -w {psf_sigma}"
    options = f"{options} -m {alim} -o {output}"
    run(f"image2xy {options} {image_file}", shell=True) 
    
    # get the results
    source_list = fits.getdata(output)
    
    # remove files if desired
    if not(write):
        run(f"rm {output}", shell=True) 

    # helpful print
    print(f'\nFound {len(source_list)} stars at >{astrom_sigma}'+
          f' sigma image {re.sub(".*/", "", image_file)}'+
          ' with astrometry.net')
    
    return source_list


def imsegm_make_source_mask(image_file, mask_file=None, sigma=3.0, 
                            write=False, output=None):
    """Make a mask which contains sources in the image using image 
    segmentation. 
    
    Arguments
    ---------
    image_file : str
        Filename for **NOT background-subtracted** image
    mask_file : str, optional
        Filename for a bad pixel mask (default None)
    sigma : float, optional
        Detection sigma to use in image segmentation (default 3.0)
    write : bool, optional
        Write the source mask to a file? (default False)
    output : str, optional
        Name for output file (default set by function)
        
    Returns
    -------
    np.ndarray
        Source mask where 0 = background, 1 = source

    Notes
    -----
    Use crude image segmentation to make a proto source mask, use the mask to
    get the background, and then perform proper image segmentation on the 
    background-subtracted image data. The resulting segmentation image is a 
    source mask, to be used in other steps in aperture photometry. The 
    output source mask also flags bad pixels.

    """    
    data = fits.getdata(image_file)
    hdr = fits.getheader(image_file)
    
    ## set the threshold for image segmentation
    # use *crude* image segmentation to find sources above SNR=3, build a 
    # source mask, and estimate the background RMS 
    if mask_file: # load a bad pixel mask if one is present 
        bp_mask = fits.getdata(mask_file)
        bp_mask = bp_mask.astype(bool)
        try:
            source_mask = make_source_mask(data, nsigma=3, npixels=5, 
                                           dilate_size=15, mask=bp_mask)
        except TypeError: # in older version of photutils, nsigma was snr
            source_mask = make_source_mask(data, snr=3, npixels=5, 
                                           dilate_size=15, mask=bp_mask)            
        # combine the bad pixel mask and source mask 
        rough_mask = np.logical_or(bp_mask,source_mask)
    else: 
        try:
            source_mask = make_source_mask(data, nsigma=3, npixels=5, 
                                           dilate_size=15)
        except TypeError: # in older version of photutils, nsigma was snr
            source_mask = make_source_mask(data, snr=3, npixels=5, 
                                           dilate_size=15)
        rough_mask = source_mask
    
    # estimate the background standard deviation
    try:
        sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
    except TypeError: # in older versions of astropy, "maxiters" was "iters"
        sigma_clip = SigmaClip(sigma=3, iters=5)
    bkg_estimator = MMMBackground()
    
    bkg = Background2D(data, (20,20), filter_size=(5,5), 
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
                       mask=rough_mask)
    bkg_rms = bkg.background_rms
    threshold = sigma*bkg_rms # threshold for proper image segmentation

    ## proper image segmentation to get a source mask    
    segm = detect_sources(data-bkg.background, threshold, npixels=5).data
    segm[segm>0] = 1 # sources have value 1, background has value 0 
    
    if write:       
        if not(output):
            output = image_file.replace(".fits", "_sourcemask.fits")
        segm_hdu = fits.PrimaryHDU(segm, hdr)   
        segm_hdu.writeto(output, overwrite=True, output_verify="ignore")
        
    return segm


def error_array(image_file, source_mask=None, sigma=3.0, write=True, 
                output=None):
    """Make an array of the error in an image, including root-mean-squared 
    deviation on the background and Poisson noise for detected sources.

    Arguments
    ---------
    image_file : str
        Filename for **NOT background-subtracted** image
    source_mask : np.ndarray, str, optional
        Filename for a source mask or the mask data itself (default None; in 
        which case a source mask is made)
    sigma : float, optional
        Detection sigma to use in image segmentation, *if* a source mask is to 
        be made (default 3.0; only relevant if `source_mask = None`)
    write : bool, optional
        Write the error array to a file? (default False)
    output : str, optional
        Name for output file (default set by function)
    
    Returns
    -------
    np.ndarray
        Image error array
    
    Notes
    -----
    Computes the error on the background-only image as the RMS deviation 
    of the background, and then computes the total image error including 
    the contribution of the Poisson noise for detected sources. Necessary 
    for error propagation in aperture photometry. 

    """

    
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    
    try:
        if "WIRCam" in image_header["INSTRUME"]:
            eff_gain = 3.8 # effective gain (e-/ADU) for WIRCam
        else: 
            eff_gain = image_header["GAIN"] # effective gain for MegaPrime
    except KeyError:
        eff_gain = image_header["HIERARCH CELL.GAIN"] # for PS1

    # mask out sources and convert to bool for background estimation
    if type(source_mask) == str:
        source_mask = fits.getdata(source_mask)
        source_mask = np.logical_not(source_mask).astype(bool)
    elif type(source_mask) == np.ndarray:
        source_mask = source_mask.astype(bool)
    else: 
        print("\nSince no mask was provided, a source mask will be obtained "+
              "now...", flush=True)
        source_mask = imsegm_make_source_mask(image_file, sigma=sigma)
        source_mask = source_mask.astype(bool)
    
    # estimate background 
    bkg_est = MMMBackground()
    bkg = Background2D(image_data, (20,20), filter_size=(5,5), 
                       bkg_estimator=bkg_est, mask=source_mask)
    
    # compute sum of Poisson error and background error  
    ### currently, this seems to overestimate unless the input data is 
    ### background-subtracted
    err = calc_total_error(image_data-bkg.background, 
                           bkg.background_rms, eff_gain)
    
    if write:
        if not(output):
            output = image_file.replace(".fits", "_error.fits")
        err_hdu = fits.PrimaryHDU(data=err, header=image_header)
        err_hdu.writeto(output, overwrite=True, output_verify="ignore")
            
    return err



###############################################################################
### PHOTOMETRY ################################################################

def aperture_photom(image_file, 
                    ra, dec, 
                    mask=None, error=None, 
                    sigma=None, bkgsub_verify=True,
                    
                    ap_radius=1.2, r1=2.0, r2=5.0,
                    
                    thresh_sigma=3.0, 
                    
                    plot_annulus=True, 
                    plot_aperture=True, 
                    ann_output=None, 
                    ap_output=None, 
                    write=False, 
                    output=None, 
                    cmap="bone"):
    """Perform aperture photometry on some image.
    
    Arguments
    ---------
    image_file : str
        Filename for **NOT background-subtracted** image
    ra, dec : float, list
        Floats **or** lists of RA, Dec of interest
    mask : np.ndarray, str, optional
        Filename for a source mask **or** the mask data itself (default None; 
        in which case a source mask is made)
    error : np.ndarray, str, optional
        Filename for image error **or** the image error itself (default None; 
        in which case an error array is made)
    sigma : float, optional
        Sigma below which to reject a source (default None --> do not reject 
        any sources)
    bkgsub_verify : bool, optional
        Verify that the background-subtracted flux is non-negative? (default 
        True)
    ap_radius : float, optional
        Aperture radius, in arcseconds (default 1.2")
    r1, r2 : float, optional
        Inner and outer radius of the annulus, in arcseconds (default 2.0", 
        5.0")
    thresh_sigma : float, optional
        Detection sigma to use in image segmentation (default 3.0; only 
        relevant if mask and/or error are not given)
    plot_annulus : bool, optional
        Plot the annulus? (default True)
    plot_aperture : bool, optional
        Plot the aperture? (default True)
    ann_output : str, optional
        Name for output annulus plot (default None; set by function if 
        `plot_annulus = True`)
    ap_output : str, optional
        Name for output aperture plot (default None; set by function if 
        `plot_aperture = True`)
    write : bool, optional
        Write the resultant table? (default False)
    output : str, optional
        Name for the output table file (default None; set by function if 
        `write = True`)
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colormap for plots (default "bone")
        
    Returns
    -------
    TYPE
        Table containing the results of aperture photometry
    
    Notes
    -----
    Finds the total flux in a defined aperture, computes the background in an 
    annulus around this aperture, and computes the background-subtracted flux 
    of the "source" defined by the aperture. Will compute aperture photometry 
    for multiple sources if a list of RA/Decs is given. 
    
    If the background-subtracted flux at some location is negative, make sure 
    that no sources remain in the annulus of the data, or consider getting a 
    **limiting magnitude** at the RA, Dec of interest instead with 
    :func:`limiting_magnitude`

    """        
    
    ## load in data 
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    # check that ZP is present
    try:
        zp_mean = image_header["ZP_MEAN"]
        zp_std = image_header["ZP_STD"]
    except KeyError:
        print("\nZP has not yet been obtained for the input image, so"+
              " calibrated magnitudes cannot be obtained. Exiting.", 
              flush=True)
        return    
    try: filt = image_header["FILTER"][0] 
    except KeyError: filt = image_header["HIERARCH FPA.FILTER"][0] # for PS1
    try: t_MJD = image_header["MJDATE"] 
    except KeyError: t_MJD = image_header["MJD-OBS"] # for PS1
    
    ## get the source mask
    if type(mask) == str:
        source_mask = fits.getdata(mask)
        source_mask = source_mask.astype(bool)
    elif type(mask) == np.ndarray:
        source_mask = mask.astype(bool)
    else: 
        print("\nSince no mask was provided, a source mask will be obtained "+
              "now...", flush=True)
        source_mask = imsegm_make_source_mask(image_file, sigma=thresh_sigma,
                                              write=False)
        source_mask = source_mask.astype(bool)           

    ## get the error array 
    if type(error) == str:
        image_error = fits.getdata(error)
    elif type(error) == np.ndarray:
        image_error = error.astype(bool)
    else:
        print("\nSince no image error array was provided, an image error "+
              "array will be obained now...", flush=True)
        image_error = error_array(image_file, source_mask, thresh_sigma, 
                                  write=False)
    image_error = np.abs(image_error) # take abs val
        
    # initialize table of sources found by aperture photometry if needed
    cols = ["xcenter","ycenter", "ra","dec", "aperture_sum", 
            "aperture_sum_err", "aper_r", "annulus_r1", "annulus_r2",
            "annulus_median", "aper_bkg", "aper_bkg_std", 
            "aper_sum_bkgsub", "aper_sum_bkgsub_err", "mag_fit", 
            "mag_unc", "mag_calib", "mag_calib_unc", "sigma"]
    aperture_sources = Table(names=cols)
    filt_col = Column([], "filter", dtype='S2') # specify
    mjd_col = Column([], "MJD")
    aperture_sources.add_column(filt_col)
    aperture_sources.add_column(mjd_col)
            
    # convert to lists if needed 
    if (type(ra) in [float, int]):
        ra = [ra]
    if (type(dec) in [float, int]):
        dec = [dec]
    
    # compute background-subtracted flux for the input aperture(s) 
    # add these to the list of sources found by aperture photometry 
    print("\nAttempting to perform aperture photometry...", flush=True)
    for i in range(0, len(ra)):

        phot_table = __drop_aperture(image_data=image_data, 
                                     image_error=image_error, 
                                     image_header=image_header,
                                     mask=source_mask, 
                                     ra=ra[i], dec=dec[i],
                                     ap_radius=ap_radius, r1=r1, r2=r2, 
                                     plot_annulus=plot_annulus,
                                     ann_output=ann_output,
                                     plot_aperture=plot_aperture,
                                     ap_output=ap_output,
                                     bkgsub_verify=bkgsub_verify,
                                     cmap=cmap)
        if phot_table: # if a valid flux (non-negative) is found
            
            # compute error on bkg-subtracted aperture sum 
            # dominated by aperture_sum_err
            phot_table["aper_sum_bkgsub_err"] = np.sqrt(
                    phot_table["aperture_sum_err"]**2+
                    phot_table["aper_bkg_std"]**2)
            
            # compute instrumental magnitude
            flux = phot_table["aper_sum_bkgsub"]
            phot_table["mag_fit"] = -2.5*np.log10(flux)
            
            # compute error on instrumental magnitude 
            phot_table["mag_unc"] = 2.5/(phot_table['aper_sum_bkgsub']*
                                        np.log(10))
            phot_table["mag_unc"] *= phot_table['aper_sum_bkgsub_err']
            
            # obtain calibrated magnitudes, propagate errors
            mag_calib = phot_table['mag_fit'] + zp_mean
            mag_calib.name = 'mag_calib'
            mag_calib_unc = np.sqrt(phot_table['mag_unc']**2 + zp_std**2)
            mag_calib_unc.name = 'mag_calib_unc'
            phot_table['mag_calib'] = mag_calib
            phot_table['mag_calib_unc'] = mag_calib_unc
             
            # compute sigma 
            phot_table["sigma"] = phot_table['aper_sum_bkgsub']
            phot_table["sigma"] /= phot_table['aper_sum_bkgsub_err']
            
            # other useful columns 
            col_filt = Column(filt, "filter", dtype = np.dtype("U2"))
            col_mjd = Column(t_MJD, "MJD")
            phot_table["filter"] = col_filt
            phot_table["MJD"] = col_mjd
            phot_table.remove_column("id") # id is not accurate 
            
            if sigma and (phot_table["sigma"] >= sigma):
                aperture_sources.add_row(phot_table[0])
                
            elif sigma and (phot_table["sigma"] < sigma):
                print("\nA source was detected, but below the requested "+
                      f"{sigma} sigma level. The source is therefore "+
                      "rejected.", flush=True)
                return
            else:
                aperture_sources.add_row(phot_table[0])
        
            try: # print calibrated mags
                a = phot_table[0]
                s = f'\n{a["filter"]} = {a["mag_calib"]:.2f} +/- '
                s += f'{a["mag_calib_unc"]:.2f}, {a["sigma"]:.1f} sigma\n'
                print(s, flush=True)
            except KeyError: # if ZP was not present before
                pass
        
        else: # if __drop_aperture fails
            return
        
    if write:
        if not(output):
            output = image_file.replace(".fits", "_apphotom.fits")
        aperture_sources.write(output, format="ascii", overwrite=True)
        
    return aperture_sources



def limiting_magnitude(image_file, 
                       ra, dec, 
                       
                       source_list=None, 
                       mask=None, error=None,
                       sigma=5.0,
                       
                       astrom_sigma=5.0, psf_sigma=5.0, 
                       
                       alim=10000, 
                       
                       ap_radius=1.2, r1=2.0, r2=10.0,
                       
                       thresh_sigma=3.0,
                       
                       plot_annulus=True, 
                       plot_aperture=True, 
                       ann_output=None, 
                       ap_output=None, 
                       write=False, 
                       output=None,
                       cmap="bone"):
    """Obtain the limiting magnitude of some image.
    
    Arguments
    ---------
    image_file : str
        Filename for **NOT background-subtracted** image
    ra, dec : float, list
        Floats **or** lists of RA, Dec of interest
    source_list : str, astropy.io.fits.fitsrec.FITS_rec, optional
        Filename for source list **or** the source list fits bintable itself 
        (default None --> make a list)
    mask : np.ndarray, str, optional
        Filename for a source mask **or** the mask data itself (default None; 
        in which case a source mask is made)
    error : np.ndarray, str, optional
        Filename for image error **or** the image error itself (default None; 
        in which case an error array is made)
    sigma : float, optional
        Sigma to use when computing limiting magnitude (default 5.0)
    astrom_sigma : float, optional
        Sigma threshold for astrometry.net source detection image (default 5.0)    
    psf_sigma : float, optional
        Sigma of the Gaussian PSF of the image to be used in 
        astrometry.net(default 5.0)
    alim : int, optional
        Maximum allowed source area, in pixels**2, beyond which sources will 
        be deblended in astrometry.net (default 10000)
    ap_radius : float, optional
        Aperture radius, in arcseconds (default 1.2")
    r1, r2 : float, optional
        Inner and outer radius of the annulus, in arcseconds (default 2.0", 
        10.0")
    thresh_sigma : float, optional
        Detection sigma to use in image segmentation (default 3.0; only 
        relevant if mask and/or error are not given)
    plot_annulus : bool, optional
        Plot the annulus? (default True)
    plot_aperture : bool, optional
        Plot the aperture? (default True)
    ann_output : str, optional
        Name for output annulus plot (default None; set by function if 
        `plot_annulus = True`)
    ap_output : str, optional
        Name for output aperture plot (default None; set by function if 
        `plot_aperture = True`)
    write : bool, optional
        Write the resultant table? (default False)
    output : str, optional
        Name for the output table file (default None; set by function if 
        `write = True`)
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colormap for plots (default "bone")    

    Returns
    -------
    TYPE
        Table containing RA, Dec, calibrated magnitude, filter used, and 
        MJD for the limiting magnitude
        
    Notes
    -----
    For a given RA, Dec, finds the limiting magnitude at its location. If 
    a source was previously detected <= 3" away from the given coords by 
    astrometry.net, the aperture will randomly move about until a valid
    RA, Dec is found. 

    """
 
    ## load in data 
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    # check that ZP is present
    try:
        zp_mean = image_header["ZP_MEAN"]
    except KeyError:
        print("\nZP has not yet been obtained for the input image, so"+
              " calibrated magnitudes cannot be obtained. Exiting.", 
              flush=True)
        return    
    try: filt = image_header["FILTER"][0] 
    except KeyError: filt = image_header["HIERARCH FPA.FILTER"][0] # for PS1
    try: t_MJD = image_header["MJDATE"] 
    except KeyError: t_MJD = image_header["MJD-OBS"] # for PS1

    ## get the list of sources 
    if not(source_list == None):
        source_list = fits.getdata(source_list)        
    if not(source_list):
        print("\nSince no list of sources was provided, a source list will be"+
              " obtained now...", flush=True)
        source_list = image2xy(image_file, astrom_sigma, psf_sigma, alim, 
                               write=False)   
        
    ## get the source mask
    if type(mask) == str:
        source_mask = fits.getdata(mask)
        source_mask = source_mask.astype(bool)
    elif type(mask) == np.ndarray:
        source_mask = mask.astype(bool)
    else: 
        print("\nSince no mask was provided, a source mask will be obtained "+
              "now...", flush=True)
        source_mask = imsegm_make_source_mask(image_file, sigma=sigma, 
                                              write=False)
        source_mask = source_mask.astype(bool)          

    ## get the error array 
    if type(error) == str:
        image_error = fits.getdata(error)
    elif type(error) == np.ndarray:
        image_error = error.astype(bool)
    else:
        print("\nSince no image error array was provided, an image error "+
              "array will be obained now...", flush=True)
        image_error = error_array(image_file, source_mask, thresh_sigma, 
                                  write=False)
    image_error = np.abs(image_error) # take abs val

    # get coords for all sources in the source list
    w = wcs.WCS(image_header)
    coords = w.all_pix2world(source_list["X"], source_list["Y"], 1)
    coords = SkyCoord(coords[0]*u.deg, coords[1]*u.deg, 1)
    target = SkyCoord(ra*u.deg, dec*u.deg)
    
    # find the smallest separation between target and all sources
    smallest_sep = np.min(target.separation(coords).value)*3600.0 
    while smallest_sep < 3.0: # while closest star is less than 3" away
        print('\nastrometry.net previously found a source < 3.0" away '+
              'from the target. The target aperture will be randomly '+
              'moved until it does not sit on top of a source...', 
              flush=True)
  
        # randomly move           
        new_ra = target.ra - u.arcsec*np.random.randint(-5, 5)
        new_dec = target.dec - u.arcsec*np.random.randint(-5, 5)
        
        # make sure it hasn't gone too far, return to center if so
        x, y = w.all_world2pix(new_ra.value, new_dec.value, 1)
        if (x<0.1*image_data.shape[1]) or (y<0.1*image_data.shape[0]) or (
                x>0.9*image_data.shape[1]) or (y>0.9*image_data.shape[0]):
            new = w.all_pix2world(image_data.shape[1]//2, 
                                  image_data.shape[0]//2, 1)
            new_ra, new_dec = new[0]*u.deg, new[1]*u.deg
        
        target = SkyCoord(new_ra, new_dec)
        smallest_sep = np.min(target.separation(coords).value)*3600.0
        
    ra, dec = target.ra.value, target.dec.value

    print("\nFinding the limiting magnitude at (RA, Dec) = "+
          f"({ra:.4f}, {dec:.4f})", flush=True)   
    
    # do aperture photometry on region of interest with large annulus
    phot_table = __drop_aperture(image_data=image_data, 
                                 image_error=image_error, 
                                 image_header=image_header,
                                 mask=source_mask, 
                                 ra=ra, dec=dec, 
                                 ap_radius=ap_radius, r1=r1, r2=r2, 
                                 plot_annulus=plot_annulus,
                                 ann_output=ann_output,
                                 plot_aperture=plot_aperture,
                                 ap_output=ap_output,
                                 bkgsub_verify=False,
                                 cmap=cmap)
 
    phot_table["aper_sum_bkgsub_err"] = np.sqrt(
            phot_table["aperture_sum_err"]**2 +
            phot_table["aper_bkg_std"]**2)
    
    # compute limit below which we can't make a detection
    limit = sigma*phot_table["aper_sum_bkgsub_err"][0]    
    limiting_mag = -2.5*np.log10(limit) + zp_mean
    print(f"\n{filt} > {limiting_mag:.1f} ({int(sigma):d} sigma)\n", 
          flush=True)
    
    lim_table = Table(data=[[ra], [dec], [limiting_mag], [filt], [t_MJD]],
                      names=["ra","dec","mag_calib","filter","MJD"])
    
    if write:
        if not(output):
            output = image_file.replace(".fits",
                                        f"_limmag_RA{ra:.5f}_DEC{dec:.5f}"+
                                        ".fits")
        lim_table.write(output, format="ascii", overwrite=True)
    
    return lim_table



def ellipse_photom(image_file, 
                   ra, dec, 
                   
                   mask=None, #error=None, 
                   
                   guess_sma=0.9, guess_eps=0.3, guess_pa=60.0,
                   sma_min=0.2, sma_max=2.5,
                   
                   r1=4.0, r2=12.0, 
                   
                   thresh_sigma=3.0,
                   
                   flux_sub=0,
                   
                   plot_annulus=True, 
                   plot_ellipses=True, 
                   ann_output=None, 
                   ell_output=None, 
                   write=False, 
                   output=None, 
                   cmap="bone"):
    """Like :func:`aperture_photom`, but use an elliptical aperture.
    
    Arguments
    ---------
    image_file : str
        Filename for **NOT background-subtracted** image
    ra, dec : float, list
        Floats **or** lists of RA, Dec of interest
    mask : np.ndarray, str, optional
        Filename for a source mask **or** the mask data itself (default None; 
        in which case a source mask is made)
    guess_sma : float, optional
        Initial guess for semi-major of ellipse, in arcseconds (default 1.0)
    guess_eps : float, optional
        Initial guess for ellipticity (default 0.3)
    guess_pa : float, optional
        Initial guess for position angle of ellipse, in degrees (default 60.0)
    sma_min, sma_max : float, optional
        Minimum/maximum allowed semi-major axis of ellipse, in arcseconds 
        (default 0.2)
    r1, r2 : float, optional
        Inner and outer radius of the annulus, in arcseconds (default 4.0", 
        12.0")
    thresh_sigma : float, optional
        Detection sigma to use in image segmentation (default 3.0)
    flux_sub : float, optional
        Additional flux to subtract, if, e.g., want to subtract the flux of 
        some galaxy (default 0)
    plot_annulus : bool, optional
        Plot the annulus? (default True)
    plot_ellipses : bool, optional
        Plot the elliptical isotophotes? (default True)
    ann_output : str, optional
        Name for output annulus plot (default None; set by function if 
        `plot_annulus = True`)
    ell_output : str, optional
        Name for output ellipses plot (default None; set by function if 
        `plot_ellipses = True`)
    write : bool, optional
        Write the resultant table? (default False)
    output : str, optional
        Name for the output table file (default None; set by function if 
        `write = True`)
    cmap : str, matplotlib.colors.ListedColormap, optional
        Colormap for plots (default "bone")

    Returns
    -------
    TYPE
        Table containing the results of "ellipse" photometry
        
    """
    
    
    from photutils.isophote import Ellipse, EllipseGeometry    
    
    ## load in data 
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file)
    w = wcs.WCS(image_header)
    # check that ZP is present
    try:
        zp_mean = image_header["ZP_MEAN"]
        zp_std = image_header["ZP_STD"]
    except KeyError:
        print("\nZP has not yet been obtained for the input image, so"+
              " calibrated magnitudes cannot be obtained. Exiting.", 
              flush=True)
        return    
    # other useful headers
    try: filt = image_header["FILTER"][0] 
    except KeyError: filt = image_header["HIERARCH FPA.FILTER"][0] # for PS1
    try: t_MJD = image_header["MJDATE"] 
    except KeyError: t_MJD = image_header["MJD-OBS"] # for PS1
    pixscale = image_header["PIXSCAL1"]
    
    ## get the source mask
    if type(mask) == str:
        source_mask = fits.getdata(mask)
        source_mask = source_mask.astype(bool)
    elif type(mask) == np.ndarray:
        source_mask = mask.astype(bool)
    else: 
        print("\nSince no mask was provided, a source mask will be obtained "+
              "now...", flush=True)
        source_mask = imsegm_make_source_mask(image_file, sigma=thresh_sigma,
                                              write=False)
        source_mask = source_mask.astype(bool)           

#    ## get the error array 
#    if type(error) == str:
#        image_error = fits.getdata(error)
#    elif type(error) == np.ndarray:
#        image_error = error.astype(bool)
#    else:
#        print("\nSince no image error array was provided, an image error "+
#              "array will be obained now...", flush=True)
#        image_error = error_array(image_file, source_mask, thresh_sigma, 
#                                  write=False)
#    image_error = np.abs(image_error) # take abs val
       
    ## initial guesses for ellipse parameters
    x0, y0 = w.all_world2pix(ra, dec, 1) # centers of isophote [pixels]
    print(f"\n ra = {ra}, dec = {dec}", flush=True)
    print(f"x0 = {x0:.3f}, y0 = {y0:.3f}", flush=True)
    sma = guess_sma/pixscale # semimajor axis length [pixels]
    eps = 0.3 # ellipticity 
    pa = guess_pa*np.pi/180.0 # position angle [radians]
    ellip_geom = EllipseGeometry(x0, y0, sma, eps, pa)
    ellip_geom.find_center(image_data) # update the center coords
    ellipse = Ellipse(image_data, geometry=ellip_geom) # build the ellipse
    
    ## get a list of isophotes, take outermost ellipse
    isolist = ellipse.fit_image(minsma=sma_min/pixscale, 
                                maxsma=sma_max/pixscale,
                                integrmode='median', 
                                sclip=3.0, nclip=3, fflag=0.3)
    tbl = isolist.to_table() # convert to table
    
    if len(tbl) == 0:
        print("\nNo ellipse could be fit. Exiting.\n", flush=True)
        return 
    tbl["tflux_circle"] = isolist.tflux_c # extra col: total flux in circle
    tbl["tflux_ellipse"] = isolist.tflux_e # extra col: total flux in ellipse
    
    phot_table = tbl[-1:] # take outermost ellipse
    ra_col = Column([ra], "ra") # add in ra, dec of source
    dec_col = Column([dec], "dec") # SHOULD THESE COORDS BE UPDATED FIRST?
    phot_table.add_column(ra_col)
    phot_table.add_column(dec_col)
    
    ## build a bad pixel mask which excludes negative pixels
    bp_mask = np.logical_or(image_data<=0, np.isnan(image_data))
    # add the pixels to the input mask (just in case)
    mask = np.logical_or(source_mask, bp_mask)   
    
    ## lay down the annulus 
    position = SkyCoord(ra, dec, unit="deg", frame="icrs") # source posn
    annulus_apertures = SkyCircularAnnulus(position, r_in=r1*u.arcsec, 
                                           r_out=r2*u.arcsec)
    annulus_apertures = annulus_apertures.to_pixel(w)
    annulus_masks = annulus_apertures.to_mask(method='center')
    
    # mask out bad pixels AND sources
    image_data_masked = np.ma.masked_where(mask, image_data)
    image_data_masked.fill_value = 0 # set to zero
    image_data_masked = image_data_masked.filled()
    annulus_data = annulus_masks.multiply(image_data_masked)
    
    try: # mask invalid data (nans) if present
        mask = np.isnan(annulus_data)
    except TypeError: # if annulus_data is None
        print("\nThere is no annulus data at this aperture. Either the "+
              "input target is out of bounds or the entire annulus is "+
              "filled with sources. Consider using a different radius "+
              "for the aperture/annuli. Exiting.", flush=True)
        return      
    annulus_data = np.ma.masked_where(mask, annulus_data)
    
    # estimate background as median in the annulus, ignoring data <= 0
    annulus_data_1d = annulus_data[annulus_data>0]
    bkg_mean, bkg_med, bkg_std = sigma_clipped_stats(annulus_data_1d,
                                                     mask_value=0.0)
    bkg_total = bkg_med*isolist[-1].npix_e # multiply by area of ellipse

    # add annuli radii, background (median), total background in ellipse, and 
    # stdev in background to table 
    # subtract total background in ellipse from total flux in ellipse 
    phot_table['annulus_r1'] = r1
    phot_table['annulus_r2'] = r2
    phot_table['annulus_median'] = bkg_med
    phot_table['ellipse_bkg'] = bkg_total
    phot_table['ellipse_bkg_std'] = np.std(annulus_data_1d) #NOT sigma-clipped
    # subtract some additional flux (manually input) if desired
    phot_table['tflux_ellipse_bkgsub'] = (
            phot_table['tflux_ellipse']-phot_table['ellipse_bkg']-flux_sub)

    ## get the magnitude from the total intensity 
    # compute instrumental magnitude
    flux = phot_table["tflux_ellipse_bkgsub"]
    phot_table["mag_fit"] = -2.5*np.log10(flux)
    
    # compute error on instrumental magnitude 
    phot_table["mag_unc"] = 2.5/(flux*np.log(10))
    phot_table["mag_unc"] *= phot_table['intens_err']
    
    # obtain calibrated magnitudes, propagate errors
    mag_calib = phot_table['mag_fit'] + zp_mean
    mag_calib.name = 'mag_calib'
    mag_calib_unc = np.sqrt(phot_table['mag_unc']**2 + zp_std**2)
    mag_calib_unc.name = 'mag_calib_unc'
    phot_table['mag_calib'] = mag_calib
    phot_table['mag_calib_unc'] = mag_calib_unc
    
    # other useful columns 
    col_filt = Column(filt, "filter", dtype = np.dtype("U2"))
    col_mjd = Column(t_MJD, "MJD")
    phot_table["filter"] = col_filt
    phot_table["MJD"] = col_mjd
    
    # print 
    a = phot_table[0]
    s = f'\n{a["filter"]} = {a["mag_calib"]:.2f} +/- '
    s += f'{a["mag_calib_unc"]:.2f}\n'
    print(s, flush=True)

    ## plotting 
    if plot_annulus:
        __plot_annulus(image_header, annulus_data, ra, dec, r1, r2, ann_output)
    if plot_ellipses:
        __plot_ellipses(image_data, image_header, annulus_apertures, ra, dec, 
                        isolist, r1, r2, ell_output, cmap)

    # write the table 
    if write:
        if not(output):
            output = image_file.replace(".fits",
                                        "_ellip_photom_RA"+
                                        f"{ra:.5f}_DEC{dec:.5f}.fits")
        phot_table.write(output, format="ascii", overwrite=True)
    
    return phot_table



###############################################################################

def __drop_aperture(image_data, image_error, image_header, mask, 
                    ra, dec, ap_radius, r1, r2, 
                    plot_annulus, ann_output,
                    plot_aperture, ap_output, bkgsub_verify,
                    cmap):
    """Find the total flux in a defined aperture, compute the background in an 
    annulus around this aperture, and compute the background-subtracted flux 
    of the "source" defined by the aperture.
    
    Returns a table containing the pixel coords, RA, Dec aperture flux, 
    aperture radius, annulus inner and outer radii, median background, total 
    background in aperture, standard deviation in this background, and 
    background-subtracted aperture flux"""
            
    # wcs object
    w = wcs.WCS(image_header)
    
    # lay down the aperture 
    position = SkyCoord(ra, dec, unit="deg", frame="icrs") # source posn
    ap = SkyCircularAperture(position, r=ap_radius*u.arcsec) # aperture 
    ap_pix = ap.to_pixel(w) # aperture in pix
    
    # build a bad pixel mask which excludes negative pixels
    bp_mask = np.logical_or(image_data<=0, np.isnan(image_data))
    # add the pixels to the input mask (just in case)
    mask = np.logical_or(mask, bp_mask)
    
    # table of the source's x, y, and total flux in aperture
    phot_table = aperture_photometry(image_data, ap_pix, error=image_error,
                                     mask=bp_mask)
    # ra, dec of source
    ra_col = Column([ra], "ra")
    dec_col = Column([dec], "dec")
    phot_table.add_column(ra_col, 3)
    phot_table.add_column(dec_col, 4)
    
    # lay down the annulus 
    annulus_apertures = SkyCircularAnnulus(position, r_in=r1*u.arcsec, 
                                           r_out=r2*u.arcsec)
    annulus_apertures = annulus_apertures.to_pixel(w)
    annulus_masks = annulus_apertures.to_mask(method='center')
    
    # mask out bad pixels AND sources
    image_data_masked = np.ma.masked_where(mask, image_data)
    image_data_masked.fill_value = 0 # set to zero
    image_data_masked = image_data_masked.filled()
    annulus_data = annulus_masks.multiply(image_data_masked)
    
    try: # mask invalid data (nans) if present
        mask = np.isnan(annulus_data)
    except TypeError: # if annulus_data is None
        print("\nThere is no annulus data at this aperture. Either the "+
              "input target is out of bounds or the entire annulus is "+
              "filled with sources. Consider using a different radius "+
              "for the aperture/annuli. Exiting.", flush=True)
        return      
    annulus_data = np.ma.masked_where(mask, annulus_data)
    
    # estimate background as median in the annulus, ignoring data <= 0
    annulus_data_1d = annulus_data[annulus_data>0]
    bkg_mean, bkg_med, bkg_std = sigma_clipped_stats(annulus_data_1d,
                                                     mask_value=0.0)
    bkg_total = bkg_med*ap_pix.area # was ap_pix.area() before
    
    # add aperture radius, annuli radii, background (median), total 
    # background in aperture, and stdev in background to table 
    # subtract total background in aperture from total flux in aperture 
    phot_table['aper_r'] = ap_radius
    phot_table['annulus_r1'] = r1
    phot_table['annulus_r2'] = r2
    phot_table['annulus_median'] = bkg_med
    phot_table['aper_bkg'] = bkg_total
    ### SHOULD THE NEXT LINE BE MULTIPLIED BY THE AREA?
    phot_table['aper_bkg_std'] = np.std(annulus_data_1d) #NOT sigma-clipped
    phot_table['aper_sum_bkgsub'] = (
            phot_table['aperture_sum']-phot_table['aper_bkg'])
    
    if (phot_table['aper_sum_bkgsub'] < 0) and bkgsub_verify:
        print("Warning: the background-subtracted flux at this aperture "+
              "is negative. It cannot be used to compute a magnitude. "+
              "Consider using a different radius for the aperture/annuli "+
              "and make sure that no sources remain in the annulus. "+
              "Alternatively, get a limiting magnitude at these coordinates "+
              "instead. Exiting.", flush=True)
        return         
    
    if plot_annulus:
        __plot_annulus(image_header, annulus_data, ra, dec, r1, r2, ann_output)   
    if plot_aperture:
        __plot_aperture(image_data, image_header, annulus_apertures, ra, dec, 
                        ap_pix, r1, r2, ap_output)  
        
    return phot_table


def __plot_annulus(image_header, annulus_data, ra, dec, r1, r2, ann_output,
                   cmap="bone"):
    """Plot an image of the pixels in the annulus drawn around a source of 
    interest for aperture photometry"""   
    
    pixscale = image_header["PIXSCAL1"]
    
    # plotting
    fig, ax = plt.subplots(figsize=(10,10)) 
    plt.imshow(annulus_data, origin="lower", cmap=cmap)
    cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cb.set_label(label="ADU",fontsize=15)
    plt.title(f'Annulus around {ra:.5f}, {dec:.5f} (1 pixel = {pixscale}")',
              fontsize=15)
    plt.xlabel("Pixels", fontsize=15)
    plt.ylabel("Pixels", fontsize=15)
    
    # textbox indicating inner/outer radii of annulus 
    textstr = r'$r_{\mathrm{in}} = %.1f$"'%r1+'\n'
    textstr += r'$r_{\mathrm{out}} = %.1f$"'%r2
    box = dict(boxstyle="square", facecolor="white", 
       alpha=0.6)
    plt.text(0.81, 0.91, transform=ax.transAxes, s=textstr, bbox=box,
             fontsize=14)
    
    if not(ann_output):
        ann_output = f"annulus_RA{ra:.5f}_DEC{dec:.5f}.png"
        
    plt.savefig(ann_output, bbox_inches="tight")
    plt.close()
        
        
def __plot_aperture(image_data, image_header, annulus_pix, ra, dec, ap_pix, r1, 
                    r2, ap_output, cmap="bone"):
    """Plot an image of the aperture and annuli drawn around a source of 
    interest for aperture photometry"""
    
    pixscale = image_header["PIXSCAL1"]
    # wcs object
    w = wcs.WCS(image_header)

    # update wcs object and image to span a box around the aperture
    # was .positions[0] before
    xpix, ypix = ap_pix.positions # pix coords of aper. centre 
    boxsize = int(annulus_pix.r_out)+5 # size of box around aperture 
    idx_x = [int(xpix-boxsize), int(xpix+boxsize)]
    idx_y = [int(ypix-boxsize), int(ypix+boxsize)]
    w.wcs.crpix = w.wcs.crpix - [idx_x[0], idx_y[0]] 
    image_data_temp = image_data[idx_y[0]:idx_y[1], idx_x[0]:idx_x[1]] 
    
    # update aperture/annuli positions 
    ap_pix.positions -= [idx_x[0], idx_y[0]] 
    annulus_pix.positions -= [idx_x[0], idx_y[0]] 
    
    # plotting
    plt.figure(figsize=(10,10))
    ax = plt.subplot(projection=w) # show wcs 
    plt.imshow(image_data_temp, origin="lower", cmap=cmap)
    ap_pix.plot(color='#c20078', lw=2) # aperture as dark pink circle
    annulus_pix.plot(color='#ed0dd9', lw=2) # annuli as fuschia circles 
    cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cb.set_label(label="ADU", fontsize=15)
    plt.title("Aperture photometry around %.5f, %.5f"%(ra, dec), 
              fontsize=15)
    textstr = r'$r_{\mathrm{aper}} = %.1f$"'%(ap_pix.r*pixscale)+'\n'
    textstr += r'$r_{\mathrm{in}} = %.1f$"'%r1+'\n'
    textstr += r'$r_{\mathrm{out}} = %.1f$"'%r2
    box = dict(boxstyle="square", facecolor="white", alpha=0.6)
    plt.text(0.83, 0.88, transform=ax.transAxes, s=textstr, bbox=box, 
             fontsize=14)
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(ap_output):
        ap_output = f"aperture_RA{ra:.5f}_DEC{dec:.5f}.png"
    
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    plt.savefig(ap_output, bbox_inches="tight")
    plt.close()


def __plot_ellipses(image_data, image_header, annulus_pix, ra, dec, 
                    isolist, r1, r2, ell_output, cmap="bone"):
    """Plot an image of the elliptical isophotes and annuli drawn around a 
    source of interest for elliptical aperture photometry"""
    
    pixscale = image_header["PIXSCAL1"]
    # wcs object
    w = wcs.WCS(image_header)

    # update wcs object and image to span a box around the aperture
    xpix, ypix = annulus_pix.positions # pix coords of aper. centre 
    boxsize = int(annulus_pix.r_out)+5 # size of box around aperture 
    idx_x = [int(xpix-boxsize), int(xpix+boxsize)]
    idx_y = [int(ypix-boxsize), int(ypix+boxsize)]
    w.wcs.crpix = w.wcs.crpix - [idx_x[0], idx_y[0]] 
    image_data_temp = image_data[idx_y[0]:idx_y[1], idx_x[0]:idx_x[1]] 
    
    # update aperture/annuli positions 
    annulus_pix.positions -= [idx_x[0], idx_y[0]] 
    
    # plotting
    plt.figure(figsize=(10,10))
    ax = plt.subplot(projection=w) # show wcs 
    plt.imshow(image_data_temp, origin="lower", cmap=cmap)
    # make sure smallest/largest ellipse is plotted, and plot 1/4 of others 
    iso2plot = [isolist[0]] + [isolist[-1]] 
    iso2plot += [isolist[i] for i in range(1,len(isolist)-1) if i%4==0]
    for iso in iso2plot:
        x, y = iso.sampled_coordinates()
        x -= idx_x[0]; y -= idx_y[0]
        plt.plot(x, y, color='#c20078', lw=2) # ellipses as dark pink circles
    annulus_pix.plot(color='#ed0dd9', lw=2) # annuli as fuschia circles 
    cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cb.set_label(label="ADU", fontsize=15)
    plt.title("Elliptical aperture photometry around %.5f, %.5f"%(ra, dec), 
              fontsize=15)
    textstr = r'$a_{\mathrm{max}} = %.1f$"'%(isolist[-1].sma*pixscale)+'\n'
    textstr += r'$r_{\mathrm{in}} = %.1f$"'%r1+'\n'
    textstr += r'$r_{\mathrm{out}} = %.1f$"'%r2
    box = dict(boxstyle="square", facecolor="white", alpha=0.6)
    plt.text(0.83, 0.88, transform=ax.transAxes, s=textstr, bbox=box, 
             fontsize=14)
    plt.xlabel("RA (J2000)", fontsize=16)
    plt.ylabel("Dec (J2000)", fontsize=16)
    
    if not(ell_output):
        ell_output = f"ellipse_RA{ra:.5f}_DEC{dec:.5f}.png"
    
    ax.coords["ra"].set_ticklabel(size=15)
    ax.coords["dec"].set_ticklabel(size=15)
    
    plt.savefig(ell_output, bbox_inches="tight")
    plt.close()
