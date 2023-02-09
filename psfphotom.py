#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Wed Sep 18 13:46:08 2019
.. @author: Nicholas Vieira
.. @psfphotom.py

CONTENTS:
    - :func:`PSF_photometry`: Get the effective Point Spread Function (ePSF) of 
      an image, fit all (or as many as desired) stars in the image with the 
      ePSF, get instrumental magnitudes, and then crossmatch with external 
      catalogue to get **calibrated** magnitudes
    - :func:`ePSF_FWHM`: Get the full width at half-max (FWHM) of an ePSF 
    - :func:`copy_zero_point`: Copy zero point headers from one image to 
      another 
    
NON-STANDARD PYTHON DEPENDENCIES:
    - `astropy`
    - `photutils`
    - `astroquery`

NON-PYTHON DEPENDENCIES:
    - `astrometry.net`
    
"""

import re
from subprocess import run
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.nddata import NDData
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter

from astroquery.vizier import Vizier

from photutils.psf import (extract_stars, BasicPSFPhotometry, DAOGroup)
from photutils import (make_source_mask, detect_sources, 
                       #source_properties,
                       SourceCatalog,
                       EPSFBuilder)

# disable annoying warnings
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)



###############################################################################

def PSF_photometry(image_file, mask_file=None, 
                   
                   nstars=40,
                   
                   thresh_sigma=5.0, 
                   pixelmin=20, 
                   elongation_lim=1.4,
                   area_max=500, 
                   
                   cutout=35, 
                   
                   astrom_sigma=5.0, 
                   psf_sigma=5.0, 
                   alim=10000,                    
                   write_xy=False,
                   
                   nsources=None, 
                   cat_num=None, 
                   sep_max=2.0, 
                   
                   write_ePSF=False, ePSF_output=None, 
                   plot_ePSF=True, ePSF_plot_output=None, 
                   plot_resids=False, resids_output=None,
                   plot_corr=False, corr_output=None,
                   plot_source_offs=False, source_offs_output=None, 
                   plot_field_offs=False, field_offs_output=None,
                   gaussian_blur_sigma=30.0, 
                   
                   write=False, output=None):
    """
    
    Arguments
    ---------
    image_file : str
        Filename for **background-subtracted** image
    mask_file : str, optional
        Filename for a bad pixel mask (default None)
    nstars : int, optional
        Maximum number of stars to use in **constructing** the ePSF (default 
        40; set to None to impose no limit)
    thresh_sigma : float, optional
        Detection sigma to use in image segmentation (default 5.0)
    pixelmin : int, optional
        *Minimum* allowed number of pixels in isophotes to be used in 
        building the ePSF (default 20)
    elongation_lim : float, optional
        *Maximum* allowed elongation of isophotes to used be in building the 
        ePSF (default 1.4)
    area_max : float, optional
        *Maximum* alllowed area for sources to be used in building the ePSF, 
        in square pixels (default 500)
    cutout : int, optional
        Size of cutout around each star, in pixels (default 35; **must be 
        odd, will be rounded down if even**)
    astrom_sigma : float, optional
        Sigma threshold for astrometry.net source detection image (default 5.0)    
    psf_sigma : float, optional
        Sigma of the Gaussian PSF of the image to be used by 
        astrometry.net (default 5.0)
    alim : int, optional
        Maximum allowed source area, in pixels**2, beyond which sources will 
        be deblended in astrometry.net (default 10000)
    write_xy : bool, optional
        Keep the `.xy.fits` source list produced by astrometry.net? (default 
        False)
    nsources : int, optional
        Limit on number of sources to fit with ePSF for obtaining the zero 
        point (default None --> impose no limit)
    cat_num : str, optional
        Vizier catalog number to determine which catalog to cross-match (
        default None --> set to PanStarrs 1, SDSS DR12, or 2MASS depending on 
        filter used during observations)
    sep_max : float, optional
        *Maximum* allowed separation between sources when cross-matching to 
        catalog, in pixels (default 2.0 ~ 0.6 arcseconds for WIRCam, 0.37 
        arcseconds for MegaCam)
    write_ePSF: bool, optional
        Write the derived ePSF to a .fits file? (default False)
    output_ePSF : str, optional
        Filename for output ePSF fits file (default None --> set by function)
    plot_ePSF : bool, optional
        Plot the ePSF? (default True)
    ePSF_plot_output : str, optional
        Filename for output ePSF plot (default None --> set by function)
    plot_resids : bool, optional
        Plot the residuals of the iterative ePSF fitting? (default False)
    resids_output : str, optional
        Filename for output residuals plot (default None --> set by function)
    plot_corr : bool, optional
        Plot the instrumental magnitude versus castalogue magnitude 
        correlation used to obtain the zero point (default False)
    corr_output : str, optional
        Filename for output correlation plot (default None --> set by function)
    plot_source_offs : bool, optional
        Plot offsets between image WCS and catalog WCS (default False)
    source_offs_output : str, optional
        Filename for output source offsets plot (default None --> set by 
        function)
    plot_field_offs : bool, optional
        Plot offsets across the field, with a Gaussian blur to visualize 
        large-scale structure in the offsets, if any
    field_offs_output : str, optional
        Filename for output field offsets plot (default None --> set by 
        function)
    gaussian_blur_sigma : float, optional
        Sigma of the Gaussian blur to use in the field offsets plot, if 
        plotting (default 30.0)
    write : bool, optional
        Write the table of calibrated sources with PSF photometry to a file? 
        (default False)
    output : str, optional
        Name for output fits table of sources (default None --> set by 
        function)
    
    Returns
    -------
    TYPE
        Table of all PSF-fit sources, with their calibrated magnitudes
    
    Notes
    -----
    Using image segmentation, find as many sources as possible in the image 
    with some constraints on number of pixels, elongation, and area. Use these 
    sources to build an empirical effective PSF (ePSF). Use a list of sources 
    found by astrometry.net to fit the ePSF to all of those sources. Compute 
    the instrumental magnitude of all of these sources. Query the correct 
    online catalogue for the given filter to crossmatch sources in the image 
    with those in the catalogue (e.g., Pan-STARRS 1). Find the zero point 
    which satisfies AB_mag = ZP + instrumental_mag and gets the calibrated 
    AB mags for all PSF-fit sources. 

    """
    
    psf_sources = __fit_PSF(image_file, mask_file, 
                            nstars, 
                            thresh_sigma, 
                            pixelmin, elongation_lim, area_max, cutout,
                            
                            astrom_sigma, psf_sigma, alim, write_xy, 
                            
                            nsources, 
                            
                            write_ePSF, ePSF_output, 
                            
                            plot_ePSF, ePSF_plot_output, 
                            plot_resids, resids_output)
    
    psf_sources = __zero_point(image_file, psf_sources, sep_max,
                               
                               plot_corr, corr_output, 
                               plot_source_offs, source_offs_output, 
                               plot_field_offs, field_offs_output, 
                               gaussian_blur_sigma, 
                               cat_num, 
                               write, output)

    return psf_sources


def __fit_PSF(image_file, mask_file=None, nstars=40,                
              thresh_sigma=5.0, pixelmin=20, elongation_lim=1.4, area_max=500,             
              cutout=35, 
              astrom_sigma=5.0, psf_sigma=5.0, alim=10000, write_xy=False, 
              nsources=None, 
              write_ePSF=False, ePSF_output=None, 
              plot_ePSF=True, ePSF_plot_output=None, 
              plot_resids=False, resids_output=None):
    """Fit several sources in an image to obtain the Point Spread Function 
    (PSF).
    
    Notes
    -----
    Use image segmentation to obtain a list of sources in the image with their 
    x, y coordinates. Use `EPSFBuilder` to empirically obtain the ePSF of 
    these stars. Optionally write and/or plot the obtaind ePSF. Finally, use
    astrometry.net to find all sources in the image, and fit them with the 
    empirically obtained ePSF.
    
    **The ePSF obtained here should NOT be used in convolutions. Instead, it 
    can serve as a tool for estimating the seeing of an image.**

    """

    # load in data 
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file) 
    try:
        instrument = image_header["INSTRUME"]
    except KeyError:
        instrument = "Unknown"
    pixscale = image_header["PIXSCAL1"]
    
    ### SOURCE DETECTION

    ### use image segmentation to find sources with an area > pixelmin pix**2 
    ### which are above the threshold sigma*std 
    image_data = fits.getdata(image_file) # subfile data
    image_data = np.ma.masked_where(image_data==0.0, 
                                    image_data) # mask bad pixels
    
    ## build an actual mask
    mask = (image_data==0)
    if mask_file:
        mask = np.logical_or(mask, fits.getdata(mask_file))

    ## set detection standard deviation
    try:
        std = image_header["BKGSTD"] # header written by amakihi.bkgsub fn
    except KeyError:
        # make crude source mask, get standard deviation of background
        source_mask = make_source_mask(image_data, snr=3, npixels=5, 
                                       dilate_size=15, mask=mask)
        final_mask = np.logical_or(mask, source_mask)
        std = np.std(np.ma.masked_where(final_mask, image_data))
    
    ## use the segmentation image to get the source properties 
    # use <mask>, which does not mask sources
    segm = detect_sources(image_data, thresh_sigma*std, npixels=pixelmin,
                          mask=mask) 
    #cat = source_properties(image_data, segm, mask=mask) # photutils 0.8
    cat = SourceCatalog(data=image_data, segment_image=segm,
                        mask=mask) # photutils >=1.1

    ## get the catalog and coordinates for sources
    tbl = cat.to_table()
    
    # restrict elongation and area to obtain only unsaturated stars 
    tbl = tbl[(tbl["elongation"] <= elongation_lim)]
    tbl = tbl[(tbl["area"].value <= area_max)]

    sources = Table() # build a table 
    sources['x'] = tbl['xcentroid'] # for EPSFBuilder 
    sources['y'] = tbl['ycentroid']
    sources['flux'] = tbl['source_sum'].data/tbl["area"].data   
    sources.sort("flux")
    sources.reverse()
    
    if nstars:
        sources = sources[:min(nstars, len(sources))]

    ## setup: get WCS coords for all sources 
    w = wcs.WCS(image_header)
    sources["ra"], sources["dec"] = w.all_pix2world(sources["x"],
                                                    sources["y"], 1)
     
    ## mask out edge sources: 
    # a bounding circle for WIRCam, rectangle for MegaPrime
    xsize = image_data.shape[1]
    ysize = image_data.shape[0]
    if "WIRCam" in instrument:
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((sources['x']-xsize/2.0)**2 + 
                                 (sources['y']-ysize/2.0)**2)
        dmask = dist_to_center <= rad_limit
        sources = sources[dmask]
    else: 
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        dmask = (sources['x']>x_lims[0]) & (sources['x']<x_lims[1]) & (
                 sources['y']>y_lims[0]) & (sources['y']<y_lims[1])
        sources = sources[dmask]
        
    ## empirically obtain the effective Point Spread Function (ePSF)  
    nddata = NDData(image_data) # NDData object
    if mask_file: # supply a mask if needed 
        nddata.mask = fits.getdata(mask_file)
    if cutout%2 == 0: # if cutout even, subtract 1
        cutout -= 1
    stars = extract_stars(nddata, sources, size=cutout) # extract stars

    ## build the ePSF
    nstars_epsf = len(stars.all_stars) # no. of stars used in ePSF building
    
    if nstars_epsf == 0:
        raise ValueError("Found no valid sources to build the ePSF with "+
                         "the given conditions")
    
    print(f"\n{nstars_epsf} stars used in building the ePSF", flush=True)
        
    start = timer()
    epsf_builder = EPSFBuilder(oversampling=1, maxiters=7, # build it
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    epsf_data = epsf.data
    
    end = timer() # timing 
    time_elaps = end-start
    
    # print ePSF FWHM, if desired
    print(f"Finished building ePSF in {time_elaps:.2f} s\n")
    _ = ePSF_FWHM(epsf_data)

    epsf_hdu = fits.PrimaryHDU(data=epsf_data)
    if write_ePSF: # write, if desired
        if not(ePSF_output):
            ePSF_output = image_file.replace(".fits", "_ePSF.fits")
            
        epsf_hdu.writeto(ePSF_output, overwrite=True, output_verify="ignore")
    
    psf_model = epsf # set the model
    psf_model.x_0.fixed = True # fix centroids (known beforehand) 
    psf_model.y_0.fixed = True
 
    ### USE ASTROMETRY.NET TO FIND SOURCES TO FIT  
    # -b --> no background-subtraction
    # -O --> overwrite
    # -p <astrom_sigma> --> signficance
    # -w <psf_sigma> --> estimated PSF sigma 
    # -m <alim> --> max object size for deblending is <alim>      
    options = f"-O -b -p {astrom_sigma} -w {psf_sigma}"
    options += f" -m {alim}"
    run(f"image2xy {options} {image_file}", shell=True)
    image_sources_file = image_file.replace(".fits", ".xy.fits")
    image_sources = fits.getdata(image_sources_file)
    if not write_xy:
        run(f"rm {image_sources_file}", shell=True) # this file is not needed

    print(f'\n{len(image_sources)} stars at >{astrom_sigma}'+
          f' sigma found in image {re.sub(".*/", "", image_file)}'+
          ' with astrometry.net', flush=True)   

    astrom_sources = Table() # build a table 
    astrom_sources['x_mean'] = image_sources['X'] # for BasicPSFPhotometry
    astrom_sources['y_mean'] = image_sources['Y']
    astrom_sources['flux'] = image_sources['FLUX']
    
    # initial guesses for centroids, fluxes
    pos = Table(names=['x_0', 'y_0','flux_0'], 
                data=[astrom_sources['x_mean'], astrom_sources['y_mean'], 
                      astrom_sources['flux']]) 

    ### FIT THE ePSF TO ALL DETECTED SOURCES 
    start = timer() # timing the fit 
    
    # sources separated by less than this critical separation are grouped 
    # together when fitting the PSF via the DAOGROUP algorithm
    sigma_psf = 2.0 # 2 pix
    crit_sep = 2.0*sigma_psf*gaussian_sigma_to_fwhm  # twice the PSF FWHM
    daogroup = DAOGroup(crit_sep) 

    # an astropy fitter, does Levenberg-Marquardt least-squares fitting
    fitter_tool = LevMarLSQFitter()
    
    # if we have a limit on the number of sources to fit
    if not(type(nsources) == type(None)):
        try: 
            import random # pick a given no. of random sources 
            source_rows = random.choices(astrom_sources, k=nsources)
            astrom_sources = Table(names=['x_mean', 'y_mean', 'flux'], 
                                   rows=source_rows)
            pos = Table(names=['x_0', 'y_0','flux_0'], 
                        data=[astrom_sources['x_mean'], 
                              astrom_sources['y_mean'], 
                              astrom_sources['flux']])
            
            
        except IndexError:
            print("The input source limit exceeds the number of sources"+
                  " detected by astrometry, so no limit is imposed.\n", 
                  flush=True)
    
    photometry = BasicPSFPhotometry(group_maker=daogroup,
                            bkg_estimator=None, # bg subtract already done
                            psf_model=psf_model,
                            fitter=fitter_tool,
                            fitshape=(11,11))
    
    result_tab = photometry(image=image_data, init_guesses=pos) # results
    residual_image = photometry.get_residual_image() # residuals of PSF fit
    residual_image = np.ma.masked_where(mask, residual_image)
    residual_image.fill_value = 0 # set to zero
    residual_image = residual_image.filled()

    
    end = timer() # timing 
    time_elaps = end - start
    print(f"Finished fitting ePSF to all sources in {time_elaps:.2f} s\n", 
          flush=True)
    
    # include WCS coordinates
    pos["ra"], pos["dec"] = w.all_pix2world(pos["x_0"], pos["y_0"], 1)
    result_tab.add_column(pos['ra'])
    result_tab.add_column(pos['dec'])
    
    # mask out negative flux_fit values in the results 
    mask_flux = (result_tab['flux_fit'] >= 0.0)
    psf_sources = result_tab[mask_flux] # PSF-fit sources 
    
    # compute magnitudes and their errors and add to the table
    # error = (2.5/(ln(10)*flux_fit))*flux_unc
    mag_fit = -2.5*np.log10(psf_sources['flux_fit']) # instrumental mags
    mag_fit.name = 'mag_fit'
    mag_unc = 2.5/(psf_sources['flux_fit']*np.log(10))
    mag_unc *= psf_sources['flux_unc']
    mag_unc.name = 'mag_unc' 
    psf_sources['mag_fit'] = mag_fit
    psf_sources['mag_unc'] = mag_unc
    
    # mask entries with large magnitude uncertainties 
    mask_unc = psf_sources['mag_unc'] < 0.4
    psf_sources = psf_sources[mask_unc]
    
    if plot_ePSF: # if we wish to see the ePSF
        plt.figure(figsize=(10,9))
        plt.imshow(epsf_data, origin='lower', aspect=1, cmap='magma',
                   interpolation="nearest")
        plt.xlabel("Pixels", fontsize=16)
        plt.ylabel("Pixels", fontsize=16)
        plt.title("Effective Point-Spread Function (1 pixel = "
                                                    +str(pixscale)+
                                                    '")', fontsize=16)
        plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
        plt.rc("xtick",labelsize=16) # not working?
        plt.rc("ytick",labelsize=16)
        
        if not(type(ePSF_plot_output) == type(None)):
            ePSF_plot_output = image_file.replace(".fits", "_ePSF.png")
        plt.savefig(ePSF_plot_output, bbox_inches="tight")
        plt.close()
    
    if plot_resids: # if we wish to see a plot of the residuals
        if "WIRCam" in instrument:
            plt.figure(figsize=(10,9))
        else:
            plt.figure(figsize=(12,14))
        ax = plt.subplot(projection=w)
        plt.imshow(residual_image, cmap='magma', aspect=1, 
                   interpolation='nearest', origin='lower')
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title("PSF residuals", fontsize=16)
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
        cb.set_label(label="ADU", fontsize=16)
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not(type(resids_output) == type(None)):
            resids_output = image_file.replace(".fits", "_ePSFresiduals.png")
        plt.savefig(resids_output, bbox_inches="tight")
        plt.close()
    
    return psf_sources     
    
    
def __zero_point(image_file, psf_sources, sep_max=2.0,
                 
                 plot_corr=False, corr_plotname=None,
                 plot_source_offs=False, source_offs_plotname=None,
                 plot_field_offs=False, field_offs_plotname=None, 
                 gaussian_blur_sigma=30.0, 
                 cat_num=None, 
                 write=False, output=None):
    
    """Compute the zero point of an image, for photometric calibration.
    
    Notes
    -----
    Use `astroquery` to query an online catalog through Vizier. Find sources 
    which match those previously detected and fit for their ePSF. Compute the 
    difference between the apparent and instrumental magnitudes of the queried 
    sources for photometric calibration. Compute the mean, median and 
    standard deviation of this difference.
    
    """   

    # load in data 
    image_data = fits.getdata(image_file)
    image_header = fits.getheader(image_file) 
    
    # don't necessarily need:
    try:
        instrument = image_header["INSTRUME"]
    except KeyError:
        instrument = "Unknown"        
    # mandatory:
    pixscale = image_header["PIXSCAL1"]    
    try: filt = image_header["FILTER"][0] 
    except KeyError: filt = image_header["HIERARCH FPA.FILTER"][0] # for PS1
    try: t_MJD = image_header["MJDATE"] 
    except KeyError: t_MJD = image_header["MJD-OBS"]
    
    # determine the catalog to compare to for photometry
    if cat_num: # if a Vizier catalog number is given 
        ref_cat = cat_num
        ref_cat_name = cat_num
    else:  
        if filt in ['g','r','i','z','Y']:
            zp_filter = (filt).lower() # lowercase needed for PS1
            ref_cat = "II/349/ps1" # PanStarrs 1
            ref_cat_name = "PS1" 
        elif filt == 'u':
            zp_filter = 'u' # closest option right now 
            ref_cat = "V/147" 
            ref_cat_name = "SDSS DR12"
        else: 
            zp_filter = filt[0] # Ks must be K for 2MASS 
            ref_cat = "II/246/out" # 2MASS
            ref_cat_name = "2MASS"
        
    w = wcs.WCS(image_header) # WCS object and coords of centre 
    xsize = image_data.shape[1]
    ysize = image_data.shape[0]          
    wcs_centre = np.array(w.all_pix2world(xsize/2.0, ysize/2.0, 1)) 

    ra_centre = wcs_centre[0]
    dec_centre = wcs_centre[1]
    radius = pixscale*np.max([xsize,ysize])/60.0 #arcmins
    minmag = 13.0 # magnitude minimum
    maxmag = 20.0 # magnitude maximum
    max_emag = 0.4 # maximum allowed error 
    nd = 5 # minimum no. of detections for a source (across all filters)
     
    # actual querying (internet connection needed)
    print(f"\nQuerying Vizier {ref_cat} ({ref_cat_name}) "+
          f"around RA {ra_centre:.4f}, Dec {dec_centre:.4f} "+
          f"with a radius of {radius:.4f} arcmin", flush=True)
    
    v = Vizier(columns=["*"], column_filters={
            zp_filter+"mag":str(minmag)+".."+str(maxmag),
            "e_"+zp_filter+"mag":"<"+str(max_emag),
            "Nd":">"+str(nd)}, row_limit=-1) # no row limit 
    Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                        unit=(u.deg, u.deg)), radius=f'{radius}m', 
                        catalog=ref_cat, cache=False)

    if len(Q) == 0: # if no matches
        raise ValueError(f"\nFound no matches in {ref_cat_name}; requested "+
                         "region may be outside footprint of catalogue or "+
                         "inputs are too strict")
        
    
    # pixel coords of found sources
    cat_coords = w.all_world2pix(Q[0]['RAJ2000'], Q[0]['DEJ2000'], 1)
    
    # mask out edge sources
    # a bounding circle for WIRCam, rectangle for MegaPrime/other instruments
    if "WIRCam" in instrument:
        rad_limit = xsize/2.0
        dist_to_center = np.sqrt((cat_coords[0]-xsize/2.0)**2 + 
                                 (cat_coords[1]-ysize/2.0)**2)
        mask = dist_to_center <= rad_limit
        good_cat_sources = Q[0][mask]
    else:
        x_lims = [int(0.05*xsize), int(0.95*xsize)] 
        y_lims = [int(0.05*ysize), int(0.95*ysize)]
        mask = (cat_coords[0] > x_lims[0]) & (
                cat_coords[0] < x_lims[1]) & (
                cat_coords[1] > y_lims[0]) & (
                cat_coords[1] < y_lims[1])
        good_cat_sources = Q[0][mask] 
    
    # cross-matching coords of sources found by astrometry
    source_coords = SkyCoord(ra=psf_sources['ra'], 
                             dec=psf_sources['dec'], 
                             frame='icrs', unit='degree')
    # and coords of valid sources in the queried catalog 
    cat_source_coords = SkyCoord(ra=good_cat_sources['RAJ2000'], 
                                 dec=good_cat_sources['DEJ2000'], 
                                 frame='icrs', unit='degree')
    
    # indices of matching sources (within <sep_max> pixels of each other) 
    idx_image, idx_cat, d2d, d3d = cat_source_coords.search_around_sky(
            source_coords, sep_max*pixscale*u.arcsec)

    if len(idx_image) <= 3:
        raise ValueError(f"Found {len(idx_image)} matches between image but "+
                         f"{ref_cat_name} and >3 matches are required")
        return
   
    nmatches = len(idx_image) # store number of matches 
    sep_mean = np.mean(d2d.value*3600.0) # store mean separation in "
    print(f'\nFound {nmatches:d} sources in {ref_cat_name} within '+
          f'{sep_max} pix of sources detected by astrometry, with average '+
          f'separation {sep_mean:.3f}" ', flush=True)
    
    # get coords for sources which were matched
    source_matches = source_coords[idx_image]
    cat_matches = cat_source_coords[idx_cat]
    source_matches_ra = [i.ra.value for i in source_matches]
    cat_matches_ra = [i.ra.value for i in cat_matches]
    source_matches_dec = [i.dec.value for i in source_matches]
    cat_matches_dec = [i.dec.value for i in cat_matches]
    # compute offsets 
    ra_offsets = np.subtract(source_matches_ra, cat_matches_ra)*3600.0 # arcsec
    dec_offsets = np.subtract(source_matches_dec, cat_matches_dec)*3600.0
    ra_offsets_mean = np.mean(ra_offsets)
    dec_offsets_mean = np.mean(dec_offsets)

    # plot the correlation
    if plot_corr:
        # fit a straight line to the correlation
        from scipy.optimize import curve_fit
        def f(x, m, b):
            return b + m*x
        
        xdata = good_cat_sources[zp_filter+'mag'][idx_cat] # catalog
        xdata = [float(x) for x in xdata]
        ydata = psf_sources['mag_fit'][idx_image] # instrumental 
        ydata = [float(y) for y in ydata]
        popt, pcov = curve_fit(f, xdata, ydata) # obtain fit
        m, b = popt # fit parameters
        perr = np.sqrt(np.diag(pcov))
        m_err, b_err = perr # errors on parameters 
        fitdata = [m*x + b for x in xdata] # plug fit into data 
        
        # plot correlation
        fig, ax = plt.subplots(figsize=(10,10))
        ax.errorbar(good_cat_sources[zp_filter+'mag'][idx_cat], 
                 psf_sources['mag_fit'][idx_image], 
                 psf_sources['mag_unc'][idx_image],
                 marker='.', mec="#fc5a50", mfc="#fc5a50", ls="", color='k', 
                 markersize=12, label=f"Data [{filt}]", zorder=1) 
        ax.plot(xdata, fitdata, color="blue", 
                 label=r"$y = mx + b $"+"\n"+r"$ m=$%.3f$\pm$%.3f, $b=$%.3f$\pm$%.3f"%(
                         m, m_err, b, b_err), zorder=2) # the linear fit 
        ax.set_xlabel(f"Catalog magnitude [{ref_cat_name}]", fontsize=15)
        ax.set_ylabel("Instrumental PSF-fit magnitude", fontsize=15)
        ax.set_title("PSF Photometry", fontsize=15)
        ax.legend(loc="upper left", fontsize=15, framealpha=0.5)
        
        if not(corr_plotname):
            corr_plotname=image_file.replace(".fits", "_PSF_photometry.png")
        plt.savefig(corr_plotname, bbox_inches="tight")
        plt.close()        
    
    # plot the RA, Dec offset for each matched source 
    if plot_source_offs:             
        # plot
        plt.figure(figsize=(10,10))
        plt.plot(ra_offsets, dec_offsets, marker=".", linestyle="", 
                color="#ffa62b", mec="black", markersize=5)
        plt.xlabel('RA (J2000) offset ["]', fontsize=15)
        plt.ylabel('Dec (J2000) offset ["]', fontsize=15)
        plt.title(f"Source offsets from {ref_cat_name} catalog", fontsize=15)
        plt.axhline(0, color="k", linestyle="--", alpha=0.3) # (0,0)
        plt.axvline(0, color="k", linestyle="--", alpha=0.3)
        plt.plot(ra_offsets_mean, dec_offsets_mean, marker="X", 
                 color="blue", label = "Mean", linestyle="") # mean
        plt.legend(fontsize=15)
        plt.rc("xtick",labelsize=14)
        plt.rc("ytick",labelsize=14)
        
        if not(source_offs_plotname):
            source_offs_plotname = image_file.replace(".fits", 
                                       "_source_offsets_astrometry.png")
        plt.savefig(source_offs_plotname, bbox_inches="tight")        
        plt.close()
    
    # plot the overall offset across the field 
    if plot_field_offs:
        from scipy.ndimage import gaussian_filter
        # add offsets to a 2d array
        offsets_image = np.zeros(image_data.shape)
        for i in range(len(d2d)): 
            x = psf_sources[idx_image][i]["x_0"]
            y = psf_sources[idx_image][i]["y_0"]
            intx, inty = int(x), int(y)
            offsets_image[inty, intx] = d2d[i].value*3600.0    
        # apply a gaussian blur to visualize large-scale structure
        blur_sigma = gaussian_blur_sigma
        offsets_image_gaussian = gaussian_filter(offsets_image, blur_sigma)
        offsets_image_gaussian *= np.max(offsets_image)
        offsets_image_gaussian *= np.max(offsets_image_gaussian)
        
        # plot
        if "WIRCam" in instrument:
            plt.figure(figsize=(10,9))
        else:
            plt.figure(figsize=(9,13))                
        ax = plt.subplot(projection=w)
        plt.imshow(offsets_image_gaussian, cmap="magma", 
                   interpolation="nearest", origin="lower")
        # textbox indicating the gaussian blur and mean separation
        textstr = r"Gaussian blur: $\sigma = %.1f$"%blur_sigma+"\n"
        textstr += r'$\overline{offset} = %.3f$"'%sep_mean
        box = dict(boxstyle="square", facecolor="white", alpha=0.8)
        if "WIRCam" in instrument:
            plt.text(0.6, 0.91, transform=ax.transAxes, s=textstr, 
                     bbox=box, fontsize=15)
        else:
            plt.text(0.44, 0.935, transform=ax.transAxes, s=textstr, 
                     bbox=box, fontsize=15)    
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.title(f"Field offsets from {ref_cat_name} catalog", fontsize=15)
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not(field_offs_plotname):
            field_offs_plotname = image_file.replace(".fits", 
                                       "_field_offsets_astrometry.png")
            
        plt.savefig(field_offs_plotname, bbox_inches="tight")        
        plt.close()
    
    # compute magnitude differences and zero point mean, median and error
    mag_offsets = ma.array(good_cat_sources[zp_filter+'mag'][idx_cat] - 
                  psf_sources['mag_fit'][idx_image])

    zp_mean, zp_med, zp_std = sigma_clipped_stats(mag_offsets)
    
    # add these to the header of the image file 
    f = fits.open(image_file, mode="update")
    f[0].header["ZP_MEAN"] = zp_mean
    f[0].header["ZP_MED"] = zp_med
    f[0].header["ZP_STD"] = zp_std
    f.close()
    
    # add a mag_calib and mag_calib_unc column to psf_sources
    mag_calib = psf_sources['mag_fit'] + zp_mean
    mag_calib.name = 'mag_calib'
    # propagate errors 
    mag_calib_unc = np.sqrt(psf_sources['mag_unc']**2 + zp_std**2)
    mag_calib_unc.name = 'mag_calib_unc'
    psf_sources['mag_calib'] = mag_calib
    psf_sources['mag_calib_unc'] = mag_calib_unc
    
    # add flag indicating if source is in a catalog and which catalog 
    in_cat = []
    for i in range(len(psf_sources)):
        if i in idx_image:
            in_cat.append(True)
        else:
            in_cat.append(False)
    in_cat_col = Column(data=in_cat, name="in_catalog")
    psf_sources[f"in {ref_cat_name}"] = in_cat_col
    
    # add new columns 
    nstars = len(psf_sources)
    col_filt = Column([filt for i in range(nstars)], "filter",
                       dtype = np.dtype("U2"))
    col_mjd = Column([t_MJD for i in range(nstars)], "MJD")
    psf_sources["filter"] = col_filt
    psf_sources["MJD"] = col_mjd
    
    # compute magnitude differences between catalog and calibration 
    # diagnostic for quality of zero point determination 
    sources_mags = psf_sources[idx_image]["mag_calib"]
    cat_mags = good_cat_sources[idx_cat][zp_filter+"mag"]
    mag_diff_mean = np.mean(sources_mags - cat_mags)
    print("\nMean difference between calibrated magnitudes and "+
          f"{ref_cat_name} magnitudes = {mag_diff_mean}", flush=True)
    
    if write: # write the table of sources w calibrated mags, if desired
        if not(output):
            output = image_file.replace(".fits", "_PSF_photometry.fits")
        psf_sources.write(output, overwrite=True, format="ascii")    
        
    return psf_sources



###############################################################################
### UTILITIES #################################################################

def ePSF_FWHM(epsf_data):
    """Obtain the full width at half-max (FWHM) of an ePSF.
    
    Arguments
    ---------
    epsf_data : str, np.ndarray
        Fits file containing ePSF data **or** the data array itself
        
    Returns
    -------
    float
        FWHM of the ePSF
        
    """
    
    from scipy.ndimage import zoom
    
    if (type(epsf_data) == str): # if a filename, open it 
        epsf_data = fits.getdata(epsf_data)
    
    # enlarge the ePSF by a factor of 100 
    epsf_data = zoom(epsf_data, 10)
    
    # compute FWHM of ePSF 
    y, x = np.indices(epsf_data.shape)
    x_0 = epsf_data.shape[1]/2.0
    y_0 = epsf_data.shape[0]/2.0
    r = np.sqrt((x-x_0)**2 + (y-y_0)**2) # radial distances from source
    r = r.astype(np.int) # round to ints 
    
    # bin the data, obtain and normalize the radial profile 
    tbin = np.bincount(r.ravel(), epsf_data.ravel()) 
    norm = np.bincount(r.ravel())  
    profile = tbin/norm 
    
    # find radius at FWHM
    limit = np.min(profile) 
    limit += 0.5*(np.max(profile)-np.min(profile)) # limit: half of maximum
    for i in range(len(profile)):
        if profile[i] >= limit:
            continue
        else: # if below 50% of max 
            epsf_radius = i # radius in pixels 
            break

    print(f"ePSF FWHM = {epsf_radius*2.0/10.0} pix\n", flush=True)
        
    return epsf_radius*2.0/10.0


    
def copy_zero_point(source_file, target_file):
    """Copy the zero point headers from `source_file` to `target_file`
    
    Returns
    -------
    zp_mean : float
        Zero points mean
    zp_med : float
        Zero points median
    zp_std : float
        Zero points standard deviation

    Notes
    -----
    Copy the zero point obtained via PSF photometry from one fits header to 
    another. Useful when one uses a background-subtracted image to do *PSF 
    photometry*, obtains the zero point, and then wants to perform *aperture
    photometry* on the original, unsubtracted image. 

    """
    
    source_hdr = fits.getheader(source_file)
    target = fits.open(target_file, mode="update")

    target[0].header["ZP_MEAN"] = source_hdr["ZP_MEAN"]
    target[0].header["ZP_MED"] = source_hdr["ZP_MED"]    
    target[0].header["ZP_STD"] = source_hdr["ZP_STD"]
    target.close()
    
    return source_hdr["ZP_MEAN"], source_hdr["ZP_MED"], source_hdr["ZP_STD"]
