#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Wed May 29 16:43:27 2019
.. @author: Nicholas Vieira
.. @lightcurve.py

Combine aperture and/or PSF photometry generated with :obj:`apphotom` and/or
:obj:`psfphotom` modules, to generate light curves.

Light curves can include 3 kinds of data points:
    
1. Magnitudes: Measured magnitudes, including error bars
2. Limiting magnitudes: Limiting magnitudes, which do not have error bars
3. Reference magnitudes: Reference magnitudes obtained from some other work / 
   analysis, with (optional) error barts

"""

#import os 
import glob
#import sys
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from astropy.table import Table
#from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.patches as ptc

# currently hard-coding this 
#plt.switch_backend("agg")


###############################################################################
### CHANGING PLOTTING BACKEND (needs to be "agg" on remote servers) ###########

def plotting_to_agg():
    """Switch matplotlib backend to 'agg'"""
    plt.switch_backend("agg")
    
def plotting_to_Qt4Agg():
    """Switch matplotlib backend to 'Qt4Agg'"""
    plt.switch_backend('Qt4Agg')



###############################################################################
### CONSTANTS #################################################################

VALID_PHOT_FILTS = ["u", "g", "r", "i", "z", "Y", "J", "H", "K"]
"""Valid photometric filters."""


DEFAULT_PLOT_INSTRUCTIONS = {"u":["#be03fd","o"], 
                             "g":["#0165fc","o"],
                             "r":["#00ffff","o"],
                             "i":["#ff9408","o"],
                             "z":["#ff474c","o"],
                               
                             "Y":["#8e82fe","s"],
                             "J":["#029386","s"],
                             "H":["#fac205","s"],
                             "K":["#c04e01","s"]}
"""Default instructions for plotting magnitudes."""


DEFAULT_PLOT_INSTRUCTIONS_LIM_MAGS = {"u":["#be03fd","v"], 
                                      "g":["#0165fc","v"],
                                      "r":["#00ffff","v"],
                                      "i":["#ff9408","v"],
                                      "z":["#ff474c","v"],
                                        
                                      "Y":["#c65102","v"],
                                      "J":["#ff028d","v"],
                                      "H":["#fac205","v"],
                                      "K":["#c04e01","v"]}
"""Default instructions for plotting **limiting** magnitudes."""


DEFAULT_PLOT_INSTRUCTIONS_REF_MAGS = {"u":"#be03fd", 
                                      "g":"#0165fc",
                                      "r":"#00ffff",
                                      "i":"#ff9408",
                                      "z":"#ff474c",
                                   
                                      "Y":"#c65102",
                                      "J":"#ff028d",
                                      "H":"#fac205",
                                      "K":"#c04e01"}
"""Default instructions for plotting **reference** magnitudes."""



PLOT_INSTRUCTIONS_VIEIRA20 = {"g":["#76cd26","s"],
                              "i":["#0165fc","o"],
                              "z":["#ff474c","D"]}
"""Plotting instructions used in Vieira et al. (2020)."""


PLOT_INSTRUCTIONS_LIM_MAGS_VIEIRA20 = {"g":["#76cd26","v"],
                                       "i":["#0165fc","v"],
                                       "z":["#ff474c","v"]}
"""Plotting instructions for **limiting magnitudes** used in Vieira et al. 
(2020)."""                                 


PLOT_INSTRUCTIONS_REF_MAGS_VIEIRA20 = {"g":"#76cd26",
                                       "i":"#0165fc",
                                       "z":"#ff474c"} 
"""Plotting instructions for **reference magnitudes** used in Vieira et al. 
(2020)."""



###############################################################################
### BUILDING LIGHTCURVES FROM FILES, DIRECTORIES, OR POINTS ###################

def fromfile(readfile):
    """Read a single .fits table file containing either magnitudes or limiting 
    magnitudes 
    
    Returns
    -------
    LightCurve
        New :obj:`LightCurve` object
    
    Notes
    -----
    Does not work for reference magnitudes. Will assume magnitude is just
    a regular magnitude. Reference magnitudes must be added one file at a time
    (see :func:`add_ref_files` and :func:`add_ref_tables`).

    """
    
    tab = Table.read(readfile, format="ascii")
    if "mag_calib_unc" in tab.colnames: # if a magnitude
        return LightCurve(mag_tab=tab)
    else: # a limiting magnitude
        return LightCurve(lim_mag_tab=tab)


def fromdirectory(directory):
    """Search a directory for .fits files and read in magnitudes or limiting 
    magnitudes

    Returns
    -------
    LightCurve
        New :obj:`LightCurve` object
    
    Notes
    -----
    Assumes that all files in the directory contain magnitudes or limiting 
    magnitudes. Reference magnitudes must be added one file at a time (see 
    :func:`add_ref_files` and :func:`add_ref_tables`).

    """
    files = glob.glob(f"{directory}/*.fits")
    ret = LightCurve()
    ret.add_files(*(files)) 
    return ret


def frompoint(ra, dec, mag, mag_err, filt, mjd):
    """Generate a new light curve from a single specified data point
    
    Arguments
    ---------
    ra, dec : float
        RA and Dec of the source of interest
    mag : float
        Magnitude
    mag_err : float
        Error on the magnitude
    filt : str
        Photometric filter used during observation
    mjd : float
        MJD at time of observation
        
    Returns
    -------
    LightCurve
        New :obj:`LightCurve` object
    
    Notes
    -----
    Should not be used for a limiting or reference magnitude. 

    """
    
    tab = Table(names=["ra","dec","mag_calib","mag_calib_unc","filter","MJD"], 
                data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])
    return LightCurve(tab)



###############################################################################
### LIGHTCURVE CLASS ##########################################################

class LightCurve:
    def __init__(self, mag_tab=None, lim_mag_tab=None, ref_mag_tab=None):

        # useful lists 
        self.__mag_colnames = ["ra", "dec", "mag_calib", "mag_calib_unc", 
                             "filter", "MJD"]
        self.__limmag_colnames = ["ra", "dec", "mag_calib", "filter", "MJD"]
        
        # magnitudes, limiting magnitudes, reference magnitudes 
        
        self.__mags = Table(names=self.__mag_colnames,
                          dtype=[np.dtype(float), np.dtype(float), 
                                 np.dtype(float), np.dtype(float), 
                                 np.dtype(str), np.dtype(float)])
        self.__lim_mags = Table(names=self.__limmag_colnames,
                              dtype=[np.dtype(float), np.dtype(float),
                                     np.dtype(float), np.dtype(str),
                                     np.dtype(float)])
        self.__ref_mags = Table(names=self.__mag_colnames,
                              dtype=[np.dtype(float), np.dtype(float), 
                                     np.dtype(float), np.dtype(float), 
                                     np.dtype(str), np.dtype(float)])
        
        if mag_tab: # if a normal magnitude table is provided
            LightCurve.__mag_table_append(self, mag_tab)
            
        if lim_mag_tab: # if a LIMITING magnitude table is provided 
            LightCurve.__limmag_table_append(self, lim_mag_tab)      

        if ref_mag_tab: # if a REFERENCE magnitude table is provided 
            LightCurve.__refmag_table_append(self, ref_mag_tab) 
        
        # a dictionary containing instructions on how to plot the lightcurve 
        # based on the filter being used (marker color, marker style) 
        self.__plot_instructions = DEFAULT_PLOT_INSTRUCTIONS.copy()
                                        
        # instructions for LIMITING magnitudes
        self.__plot_instructions_lim_mags = DEFAULT_PLOT_INSTRUCTIONS_LIM_MAGS.copy()                             

        # instructions for REFERENCE magnitudes
        self.__plot_instructions_ref_mags = DEFAULT_PLOT_INSTRUCTIONS_REF_MAGS.copy()
                                           
        # useful lists 
        self.__mag_colnames = ["ra", "dec", "mag_calib", "mag_calib_unc", 
                             "filter", "MJD"]
        self.__limmag_colnames = ["ra", "dec", "mag_calib", "filter", "MJD"]


    def __str__(self):
        if len(self.__mags) > 0: 
            print("\n\nMAGS\n", flush=True)
            self.__mags.pprint()
        if len(self.__lim_mags) > 0: 
            print("\nLIMITING MAGS\n", flush=True)
            self.__lim_mags.pprint()
        if len(self.__ref_mags) > 0: 
            print("\nREFERENCE MAGS\n", flush=True)
            self.__ref_mags.pprint()            
        return ""


    def copy(self):
        """Return a (deep) copy of the object"""
        return deepcopy(self)


    ### GETTERS ###############################################################
    
    @property
    def mags(self):
        """Table of magnitudes"""
        return self.__mags
    
    @property
    def limmags(self):
        """Table of **limiting** magnitudes"""
        return self.__lim_mags
    
    @property
    def refmags(self):
        """Table of **reference** magnitudes"""
        return self.__ref_mags
    
    @property
    def plot_instructions(self):
        """Instructions for plotting magnitudes"""
        return self.__plot_instructions
        
    @property
    def plot_instructions_lim_mags(self):
        """Instructions for plotting **limiting** magnitudes"""
        return self.__plot_instructions_lim_mags

    @property
    def plot_instructions_ref_mags(self):
        """Instructions for plotting **reference** magnitudes"""
        return self.__plot_instructions_ref_mags
    
    @property
    def mag_colnames(self):
        """Columns to use as keys when searching for magnitudes in input 
        files/tables"""
        return self.__mag_colnames

    @property
    def limmag_colnames(self):
        """Columns to use as keys when searching for **limiting** magnitudes 
        in input files/tables"""
        return self.__limmag_colnames
    

    ### UTILITIES #############################################################
    ## adding magnitude data points from tables/files ##            
    def __mag_table_append(self, table_new):
        """Append to the object's existing magnitude table"""
        for r in table_new[self.__mag_colnames]:
            self.__mags.add_row(r)
        self.__mags.sort(['ra','dec','MJD'])


    def __mag_file_append(self, file):
        """Read in file and append to the object's existing magnitude table"""
        t = Table.read(file, format="ascii")
        LightCurve.__mag_table_append(self, t)
            

    ## adding LIMITING magnitude data points from tables/files ##
    def __limmag_table_append(self, table_new):
        """Append to the object's existing **limiting** magnitude table"""       
        for r in table_new[self.__limmag_colnames]:
            self.__lim_mags.add_row(r)
        self.__lim_mags.sort(['ra','dec','MJD'])


    def __limmag_file_append(self, file):
        """Read in file and append to the object's existing **limiting** 
        magnitude table"""
        t = Table.read(file, format="ascii")
        LightCurve.__limmag_table_append(self, t)


    ## adding REFERENCE magnitude data points from tables/files ##
    def __refmag_table_append(self, table_new):
        """Append to the object's existing **reference** magnitude table"""  
        if not "mag_calib_unc" in table_new.colnames:
            table_new["mag_calib_unc"] = [None for i in range(len(table_new))]
        
        for r in table_new[self.__mag_colnames]:
            self.__ref_mags.add_row(r)
        self.__ref_mags.sort(['ra','dec','MJD'])


    def __refmag_file_append(self, file):
        """Read in file and append to the object's existing **reference** 
        magnitude table"""
        t = Table.read(file, format="ascii")
        LightCurve.__refmag_table_append(self, t)



    ### READING, WRITING, COPYING #############################################
#    def read(self, LC_file):
#        """        
#        Input: LightCurve file (.lc.fits) to read table data from
#            
#        Reads two tables from a .fits file and uses it to build a table for the 
#        given LightCurve object. Overwrites the existing data.
#        
#        Output: 
#        """
#        
#        if not(".lc.fits" in LC_file):
#            print("The input file must have extension .lc.fits. Exiting.")
#            return
#        
#        try:
#            hdul = fits.open(LC_file)
#            mags, lim_mags = Table(hdul[1].data), Table(hdul[2].data)
#            self.__mags, self.__lim_mags = mags, lim_mags           
#            self.__mags.sort(['ra','dec','MJD'])
#            self.__lim_mags.sort(['ra','dec','MJD'])
#        except:
#            e = sys.exc_info()
#            print("\n"+str(e[0])+"\n"+str(e[1])+"\nExiting.")            
#
#    
#    def write(self, LC_file, overwrite=False):
#        """
#        Input: 
#            - file to write the LightCurve table data to (MUST be of form 
#              "[...].lc.fits")
#            - whether to overwrite if file already exists (optional; default 
#              False, in which case the new data is *appended* to the old)
#            
#        Output: None
#        """
#        if not(".lc.fits" in LC_file):
#            print("Output file must have extension .lc.fits. Exiting.")
#            return
#        
#        cwd = os.getcwd() # current working directory 
#        
#        # if file is already present and populated and overwrite=False
#        if not(overwrite) and (LC_file in os.listdir(cwd)) and (
#                os.stat(LC_file).st_size!=0):
#            print("\nFile exists and is non-empty. Set overwrite=True to "+
#                  "overwrite it.")
#            return       
#        
#        hdr = fits.Header() # blank header
#        primhdu = fits.PrimaryHDU(hdr) # wrap in Primary HDU, add headers
#        primhdu.header["CLASS"] = "lightcurve"
#        primhdu.header["TAB0"] = ("magnitudes", "table of magnitudes")
#        primhdu.header["TAB1"] = ("limiting magnitudes",
#                      "table of limiting magnitudes")
#        # build HDUList out of informational primary HDU and the magnitude and 
#        # limiting magnitude tables, each as BinTableHDU
#        hdul = fits.HDUList([primhdu, 
#                             fits.BinTableHDU(self.__mags),
#                             fits.BinTableHDU(self.__lim_mags)])    
#        hdul.writeto(LC_file) # write it
    

    ### ADDING TABLES/ADDING TABLES FROM FILES ################################
    
    ### MAGNITUDES / LIMITING MAGNITUDES
    
    def add_tables(self, *tables):
        """Add one or more new tables to the existing LightCurve's magnitude 
        table and/or limiting magnitude table
        
        Notes
        -----
        Changes the `LightCurve` in place. Assumes that the tables contain 
        either magnitudes or limiting magnitudes. For adding in reference 
        magnitudes, see :func:`add_ref_tables`.

        """
        
        for t in tables:
            # if table contains actual aperture magnitudes
            if "mag_calib_unc" in t.colnames:
                LightCurve.__mag_table_append(self, t.copy())
            # if table contains limiting magnitudes
            else: 
                LightCurve.__limmag_table_append(self, t.copy())


    def add_files(self, *files):
        """Read in from files and then append any new tables to the existing 
        LightCurve's magnitude table and/or limiting magnitude table

        Notes
        -----
        Changes the `LightCurve` in place. Assumes that the files contain 
        tables with either magnitudes or limiting magnitudes. For adding in 
        reference magnitudes, see :func:`add_ref_files`.
        
        """
        for f in files:
            # if file contains actual aperture magnitudes
            if "mag_calib_unc" in Table.read(f, format="ascii").colnames:
                LightCurve.__mag_file_append(self, f)
            # if table contains limiting magnitudes
            else:              
                LightCurve.__limmag_file_append(self, f)           


    ### REFERENCE MAGNITUDES
       
    def add_ref_tables(self, *tables):
        """Same as :func:`add_tables`, but for **reference** magnitudes"""
        for t in tables: LightCurve.__refmag_table_append(self, t.copy())


    def add_ref_files(self, *files):
        """Same as :func:`add_files`, but for **reference** magnitudes"""
        for f in files: LightCurve.__refmag_file_append(self, f)     



    ### ADDING DISCRETE MEASURED/LIMTIING/REFERENCE MAGNITUDES ################
    
    def add_mag(self, ra, dec, mag, mag_err, filt, mjd):
        """Manually add a single point to the light curve's magnitude table
        
        Arguments
        ---------
        ra, dec : float
            RA and Dec of the source of interest
        mag : float
            Magnitude
        mag_err : float
            Error on the magnitude
        filt : str
            Photometric filter used during observation
        mjd : float
            MJD at time of observation

        """
        
        pt = Table(names=self.__mag_colnames, 
                   data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])
        
        LightCurve.add_tables(self, pt)
     
        
    def add_limmag(self, ra, dec, mag, filt, mjd):
        """Manually add a single point to the light curve's **limiting** 
        magnitude table
        
        Arguments
        ---------
        ra, dec : float
            RA and Dec of the source of interest
        mag : float
            **Limiting** magnitude
        filt : str
            Photometric filter used during observation
        mjd : float
            MJD at time of observation
        
        """
        
        lm = Table(names=self.__limmag_colnames, 
                   data=[[ra],[dec],[mag],[filt],[mjd]])        

        self.__lim_mags.add_row(lm[0])
        
    
    def add_refmag(self, ra, dec, mag, filt, mjd, mag_err=None):
        """Manually add a single point to the light curve's **reference** 
        magnitude table
        
        Arguments
        ---------
        ra, dec : float
            RA and Dec of the source of interest
        mag : float
            **Reference** magnitude
        filt : str
            Photometric filter used during observation
        mjd : float
            MJD at time of observation
        mag_err : float, optional
            Error on the magnitude, if any (default None)

        """

        rm = Table(names=self.__mag_colnames, 
                   data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])    
            
        self.__ref_mags.add_row(rm[0])



    ### RETURN A NEW LIGHTCURVE OBJECT FOR ONLY ONE COORDINATE ################

    def coords_select(self, ra, dec, sep=1.0):
        """Given some light curve, select only data points for measurements 
        taken within `sep` arcseconds of RA `ra` and Dec `dec`
        
        Arguments
        ---------
        ra, dec : float
            RA, Dec of the source of interest
        sep : float, optional
            Maximum allowed separation from the RA, Dec, in arcseconds 
            (default 1.0)
            
        Returns
        -------
        LightCurve
            New :obj:`LightCurve` object containing only sources satisfying 
            the condition
        
        Notes
        -----
        Does **NOT** change the object in-place. Returns a new object.

        """
        # target of interest
        toi = SkyCoord(ra, dec, frame='icrs', unit='degree') 
        
        # coords of magnitude/limiting magnitude table
        LC_mag_coords = SkyCoord(ra=self.__mags['ra'], dec=self.__mags["dec"], 
                                 frame='icrs', unit='degree')
        LC_limmag_coords = SkyCoord(ra=self.__lim_mags['ra'], 
                                    dec=self.__lim_mags["dec"], 
                                    frame='icrs', unit='degree')
        LC_refmag_coords = SkyCoord(ra=self.__ref_mags['ra'], 
                                    dec=self.__ref_mags["dec"], 
                                    frame='icrs', unit='degree')
        
        mask = (toi.separation(LC_mag_coords) <= sep*u.arcsec)
        mags_match = self.__mags[mask]
        mask = (toi.separation(LC_limmag_coords) <= sep*u.arcsec) 
        limmags_match = self.__lim_mags[mask]
        mask = (toi.separation(LC_refmag_coords) <= sep*u.arcsec) 
        refmags_match = self.__ref_mags[mask]
        
        mags_match.pprint()
        limmags_match.pprint()
        refmags_match.pprint()
        
        return LightCurve(mag_tab=mags_match, lim_mag_tab=limmags_match,
                          ref_mag_tab=refmags_match)
        

    ### SETTING THE PLOTTING INSTRUCTIONS #####################################   
    def set_plot_instructions(self, filename):
        """Set the plotting instructions
        
        Arguments
        ---------
        filename : str
            Name of .csv file containing plotting instructions
            
        Notes
        -----
        The .csv should have no header, and should contain separate rows of 
        the form 
        
            "u","red","o"
            
            "g","#00ffff","s"
            
            ...
        
        Indicating the photometric filter, line/marker colour, and marker 
        style to use. 
        
        Will **not** change the marker style for limiting magnitudes, which 
        are fixed at "v" (downwards caret).
        
        """
        import pandas as pd
        df = pd.read_csv(filename, header=None) # read in csv
        
        if df.isnull().values.any(): # if any nans in any columns, reject
            raise ValueError("Found a nan/empty cell in the table; fill this "+
                             "cell and try again")
        
        # build dictionaries for mags, limiting mags, and reference mags
        magdict = dict(zip(df[0].values,
                           [[df[1].tolist()[i], df[2].tolist()[i]] for i in 
                             range(len(df))]))
        limdict = dict(zip(df[0].values,
                           [[df[1].tolist()[i], "v"] for i in range(len(df))]))
        refdict = dict(zip(df[0].values, df[1].values))
        
        # update the object
        self.__plot_instructions = magdict
        self.__plot_instructions_lim_mags = limdict
        self.__plot_instructions_ref_mags = refdict

    
    ### PLOTTING ##############################################################
    def plot(self, *filters, output="lightcurve.png", title=None, 
             tmerger=None, show_legend=True, connect=True, text=None,
             mag_min=None, mag_max=None, 
             limmag_min=None, limmag_max=None, 
             refmag_min=None, refmag_max=None):
        """Plot your light curve!
        
        Arguments
        ---------
        *filters : str, optional
            Filter(s) of choice, if we wish to plot only the points for these 
            filters (default None; valid options in :obj:`VALID_PHOT_FILTERS`)
        output : str, optional
            Filename for output figure (default "lightcurve.png"; set to None 
            to save no figure)
        title : str, optional
            Title for the plot (default None)
        tmerger : float, optional
            Time of the merger, in MJD, to plot in time elapsed since merger 
            (default None --> just plot MJD)
        show_legend : bool, optional
            Include a legend? (default True)
        connect : bool, optional
            Connect points of the same photometric filter with a line, for 
            legibility? (default True)
        text : tuple, optional
            Text to place in a text box, in the form (x, y, 'text you want 
            printed') (default None)
            
        mag_min, mag_max : float, optional
            Lower, upper limits on the magnitudes in the light curve; omit 
            datapoints outside bounds (default None)
        limmag_min, limmag_max : float, optional
            Lower, upper limits on the **limiting** magnitudes in the light 
            curve; omit datapoints outside bounds (default None)
        refmag_min, refmag_max : float, optional
            Lower, upper limits on the **reference** magnitudes in the light 
            curve; omit datapoints outside bounds (default None)

        Returns
        -------
        matplotlib.figure.Figure
            Figure object, after all plotting

        Notes
        -----
        "lower" and "upper" limits on the data points are understood in terms 
        of the magnitude system. I.e., m=26 is a lower limit, m=21 is an upper 
        limit.

        """
        
        if (len(self.__mags) == 0)  and (len(self.__lim_mags) == 0):
            raise ValueError("LightCurve object has no magnitude or limiting "+
                             "magnitude data points; cannot plot")
        
        plotted_filts = [] # keep track of filters which have been plot already
        
        self.__mags.sort("MJD")
        self.__lim_mags.sort("MJD")
        self.__ref_mags.sort("MJD")
        
        ## plot magnitudes and their errors
        if len(self.__mags) > 0:
            sources = self.__mags  
            
            if filters: # if a filters argument is given
                mask = (sources["filter"] == filters[0])
                for filt in filters[1:]:
                    mask += (sources["filter"] == filt)
                sources = sources[mask] # only use those filters   
            # if limits on magnitudes 
            if mag_min:
                sources = sources[sources["mag_calib"] < mag_min]
            if mag_max:
                sources = sources[sources["mag_calib"] > mag_max]              
            
            if tmerger: # if a t=0 is given
                t = [tim - tmerger for tim in sources["MJD"].data]
            else:
                t = sources["MJD"].data
                
            # magnitudes and their errors
            mag = sources["mag_calib"].data
            mag_err = sources["mag_calib_unc"].data
    
            # plot them          
            fig = plt.figure(figsize=(14,10))
            for i in range(len(t)): 
                filt = str(sources["filter"].data[i])
                color, form = self.__plot_instructions[filt]
                if filt in plotted_filts:
                    plt.errorbar(t[i], mag[i], mag_err[i], fmt=form, mfc=color, 
                                 mec="black", mew=2.0, ls="", color="black", 
                                 ms=18.0, zorder=4)
                else:
                    plt.errorbar(t[i], mag[i], mag_err[i], fmt=form, mfc=color, 
                                 mec="black", mew=2.0, ls="", color="black", 
                                 label=filt, ms=18.0, zorder=4)
                plotted_filts.append(filt)
            
            if connect:
                for f in plotted_filts:
                    mask = self.__mags["filter"] == f
                    color, __ = self.__plot_instructions[f]
                    trelevant = np.array(t)[mask]
                    magrelevant = np.array(mag)[mask]
                    plt.plot(trelevant, magrelevant, marker="", ls="-", 
                             lw=2.0, zorder=0, color=color, alpha=0.6)
        
        ## plot limiting magnitudes 
        if len(self.__lim_mags) > 0:
            lims = self.__lim_mags
            
            if filters: # if a filters argument is given
                mask = (lims["filter"] == filters[0])
                for filt in filters[1:]:
                    mask += (lims["filter"] == filt)
                lims = lims[mask] # only use those filters

            # if limits on limiting magnitudes 
            if limmag_min:
                lims = lims[lims["mag_calib"] < limmag_min]
            if limmag_max:
                lims = lims[lims["mag_calib"] > limmag_max] 

            if tmerger: # if a t=0 is given
                t = [tim - tmerger for tim in lims["MJD"].data]
            else:
                t = lims["MJD"].data
            
            # limiting magnitudes
            mag = lims["mag_calib"].data
            
            # plot them
            for i in range(len(lims)): 
                filt = str(lims["filter"].data[i])
                color, form = self.__plot_instructions_lim_mags[filt]
                if filt in plotted_filts:
                    plt.plot(t[i], mag[i], marker=form, mfc=color, mec="black", 
                             mew=2.0, ls="", ms=24.0, zorder=3)     
                else:
                    plt.plot(t[i], mag[i], marker=form, mfc=color, mec="black", 
                             mew=2.0, ls="", label=filt, ms=24.0, zorder=3)  
                plotted_filts.append(filt)    

        ## plot reference magnitudes 
        if len(self.__ref_mags) > 0:
            refs = self.__ref_mags
            
            if filters: # if a filters argument is given
                mask = (refs["filter"] == filters[0])
                for filt in filters[1:]:
                    mask += (refs["filter"] == filt)
                refs = refs[mask] # only use those filters

            # if limits on reference magnitudes 
            if refmag_min:
                refs = refs[refs["mag_calib"] < refmag_min]
            if refmag_max:
                refs = refs[refs["mag_calib"] > refmag_max] 
                
            if tmerger: # if a t=0 is given
                t = [tim - tmerger for tim in refs["MJD"].data]
            else:
                t = refs["MJD"].data
            
            # refence magnitudes and their errors
            mag = refs["mag_calib"].data
            mag_err = refs["mag_calib_unc"].data
            
            # plot them          
            for i in range(len(t)): 
                filt = str(refs["filter"].data[i])
                color = self.__plot_instructions_ref_mags[filt]
                if filt in plotted_filts:
                    plt.axhline(mag[i], color=color, ls="-", lw=3.0, zorder=2)
                else:
                    plt.axhline(mag[i], color=color, ls="-", lw=3.0,
                                label=filt, zorder=2)
                    
                if mag_err[i]:
                    x0, xf = plt.xlim()
                    rect = ptc.Rectangle([x0, mag[i]-mag_err[i]], 
                                         width=xf-x0, height=2.0*mag_err[i],
                                         alpha=0.2, color=color,
                                         zorder=1)
                    plt.gca().add_patch(rect)
                plotted_filts.append(filt)     
                


        ## show the legend
        if show_legend:
            # remove duplicate labels/handles 
            handles, labels = plt.gca().get_legend_handles_labels()
            temp_dict = dict(zip(labels, handles))
            # reorder labels from shortest wavelength to longest 
            valid_filts = VALID_PHOT_FILTS.copy()
            new_labels = []; new_handles = []
            while len(valid_filts) > 0:
                if valid_filts[0] in labels:
                    new_labels.append(valid_filts[0])
                    new_handles.append(temp_dict[valid_filts[0]])
                valid_filts = valid_filts[1:]      
            by_label = OrderedDict(zip(new_labels,new_handles))
            ax = plt.gca() # get current axes
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, 
                             box.width, box.height * 0.9])
            ax.legend(by_label.values(), by_label.keys(), 
                      #loc="lower center", 
                      #bbox_to_anchor=(0.5, -0.15), 
                      loc="center right",
                      bbox_to_anchor=(1.13, 0.5),
                      fontsize=18, 
                      ncol=1, fancybox=True)
        
        ## titles and axis labels 
        if title:  # set it
            plt.title(title, fontsize=18)
        if tmerger:
            plt.xlabel(r"$t - t_{\mathrm{merger}}$"+" [days]", fontsize=18)
        else:
            plt.xlabel("MJD", fontsize=18)        
        plt.ylabel("Magnitude (AB)", fontsize=18)
        plt.gca().invert_yaxis() # invert so brighter stars higher up  
        plt.grid()
        
        ## ticks
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.gca().xaxis.set_ticks_position("both") # ticks on both sides
        plt.gca().yaxis.set_ticks_position("both")
        
        ## text box
        if text:
            plt.text(text[0], text[1], text[2], fontsize=18)
        
        if output:
            plt.savefig(output, bbox_inches="tight")
        
        return fig      
