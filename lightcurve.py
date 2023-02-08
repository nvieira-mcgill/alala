#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Wed May 29 16:43:27 2019
.. @author: Nicholas Vieira
.. @lightcurve.py

Combine aperture and/or PSF photometry (from `apphotom` and `psfphotom`) to 
generate light curves.

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
D_OFF = False
import matplotlib.patches as ptc

# currently hard-coding this 
#plt.switch_backend("agg")

###############################################################################
### TURN THE DISPLAY OFF (e.g. when on a remote sever) ########################

def display_off():
    global D_OFF
    D_OFF = True
    #plt.switch_backend("agg")

###############################################################################
### BUILDING LIGHTCURVES FROM FILES, DIRECTORIES, OR POINTS ###################

def fromfile(readfile):
    """
    Input: a single .fits table file containing either magnitudes or limiting 
    magnitudes 
    
    NOTE: Does not work for reference magnitudes. Will assume magnitude is just
    a regular magnitude. Reference magnitudes must be added one file at a time
    (see add_ref_files() and add_ref_tables() below.)
    
    Output: a new LightCurve object
    """
    
    tab = Table.read(readfile, format="ascii")
    if "mag_calib_unc" in tab.colnames: # if a magnitude
        return LightCurve(mag_tab=tab)
    else: # a limiting magnitude
        return LightCurve(lim_mag_tab=tab)

def fromdirectory(directory):
    """
    Input: a single directory to look for .fits files and use all of them to
    build a LightCurve object
    
    NOTE: Assumes that all files in the directory are magnitudes or limiting 
    magnitudes. Reference magnitudes must be added one file at a time (see 
    add_ref_files() and add_ref_tables() below.)
    
    Output: a new LightCurve object
    """
    files = glob.glob(f"{directory}/*.fits")
    ret = LightCurve()
    ret.add_files(*(files)) 
    return ret


def frompoint(ra, dec, mag, mag_err, filt, mjd):
    """
    Input: 
        - RA and Dec
        - magnitude and its error
        - filter used
        - time of observation in MJD 
        
    Initialize a LightCurve from a single magnitude data point. This cannot be 
    done using a limiting or reference magnitude. 
        
    Output: a new LightCurve object
    """
    tab = Table(names=["ra","dec","mag_calib","mag_calib_unc","filter","MJD"], 
                data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])
    return LightCurve(tab)

###############################################################################
### LIGHTCURVE CLASS ##########################################################

class LightCurve:
    def __init__(self, mag_tab=None, lim_mag_tab=None, ref_mag_tab=None):

        # useful lists 
        self.mag_colnames = ["ra", "dec", "mag_calib", "mag_calib_unc", 
                             "filter", "MJD"]
        self.limmag_colnames = ["ra", "dec", "mag_calib", "filter", "MJD"]
        
        # magnitudes, limiting magnitudes, reference magnitudes 
        
        self.mags = Table(names=self.mag_colnames,
                          dtype=[np.dtype(float), np.dtype(float), 
                                 np.dtype(float), np.dtype(float), 
                                 np.dtype(str), np.dtype(float)])
        self.lim_mags = Table(names=self.limmag_colnames,
                              dtype=[np.dtype(float), np.dtype(float),
                                     np.dtype(float), np.dtype(str),
                                     np.dtype(float)])
        self.ref_mags = Table(names=self.mag_colnames,
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

        ## UPDATED
        self.plot_instructions = {"u":["#be03fd","o"], 
                                  "g":["#0165fc","o"],
                                  "r":["#00ffff","o"],
                                  "i":["#ff9408","o"],
                                  "z":["#ff474c","o"],
                                       
                                  "Y":["#8e82fe","s"],
                                  "J":["#029386","s"],
                                  "H":["#fac205","s"],
                                  "K":["#c04e01","s"]}
                                        
        # instructions for LIMITING magnitudes
        self.plot_instructions_lim_mags = {"u":["#be03fd","v"], 
                                           "g":["#0165fc","v"],
                                           "r":["#00ffff","v"],
                                           "i":["#ff9408","v"],
                                           "z":["#ff474c","v"],
                                                
                                           "Y":["#c65102","v"],
                                           "J":["#ff028d","v"],
                                           "H":["#fac205","v"],
                                           "K":["#c04e01","v"]}                                   

        # instructions for REFERENCE magnitudes
        self.plot_instructions_ref_mags = {"u":"#be03fd", 
                                           "g":"#0165fc",
                                           "r":"#00ffff",
                                           "i":"#ff9408",
                                           "z":"#ff474c",
                                           
                                           "Y":"#c65102",
                                           "J":"#ff028d",
                                           "H":"#fac205",
                                           "K":"#c04e01"} 
        
#        ## AT TIME OF GW190814 PAPER: ####
#        self.plot_instructions = {"g":["#76cd26","s"],
#                                  "i":["#0165fc","o"],
#                                  "z":["#ff474c","D"]}
#                                        
#        # instructions for LIMITING magnitudes
#        self.plot_instructions_lim_mags = {"g":["#76cd26","v"],
#                                           "i":["#0165fc","v"],
#                                           "z":["#ff474c","v"]}                                   
#
#        # instructions for REFERENCE magnitudes
#        self.plot_instructions_ref_mags = {"g":"#76cd26",
#                                           "i":"#0165fc",
#                                           "z":"#ff474c"} 
                                           
        # useful lists 
        self.mag_colnames = ["ra", "dec", "mag_calib", "mag_calib_unc", 
                             "filter", "MJD"]
        self.limmag_colnames = ["ra", "dec", "mag_calib", "filter", "MJD"]

    def __str__(self):
        """
        Input: None
        Printing function.
        Output: None
        """
        
        if len(self.mags) > 0: 
            print("\n\nMAGS\n")
            self.mags.pprint()
        if len(self.lim_mags) > 0: 
            print("\nLIMITING MAGS\n")
            self.lim_mags.pprint()
        if len(self.ref_mags) > 0: 
            print("\nREFERENCE MAGS\n")
            self.ref_mags.pprint()            
        return ""


    ## adding magnitude data points from tables/files ##            
    def __mag_table_append(self, table_new):
        """
        Input: table to append to the object's existing MAG table 

        Output: None 
        """        
        for r in table_new[self.mag_colnames]:
            self.mags.add_row(r)
        self.mags.sort(['ra','dec','MJD'])


    def __mag_file_append(self, file):
        """
        Input: file to read a table from to then append to the object's 
        existing MAG table             
        
        Output: None
        """
        t = Table.read(file, format="ascii")
        LightCurve.__mag_table_append(self, t)
            

    ## adding LIMITING magnitude data points from tables/files ##
    def __limmag_table_append(self, table_new):
        """        
        Input: table to append to the object's existing LIMITING mag table 

        Output: None 
        """        
        for r in table_new[self.limmag_colnames]:
            self.lim_mags.add_row(r)
        self.lim_mags.sort(['ra','dec','MJD'])


    def __limmag_file_append(self, file):
        """        
        Input: file to read a table from to then append to the object's 
        existing LIMITING mag table             
        
        Output: None
        """
        t = Table.read(file, format="ascii")
        LightCurve.__limmag_table_append(self, t)


    ## adding REFERENCE magnitude data points from tables/files ##
    def __refmag_table_append(self, table_new):
        """        
        Input: table to append to the object's existing REFERENCE mag table 

        Output: None 
        """    
        if not "mag_calib_unc" in table_new.colnames:
            table_new["mag_calib_unc"] = [None for i in range(len(table_new))]
        
        for r in table_new[self.mag_colnames]:
            self.ref_mags.add_row(r)
        self.ref_mags.sort(['ra','dec','MJD'])


    def __refmag_file_append(self, file):
        """        
        Input: file to read a table from to then append to the object's 
        existing REFERENCE mag table             
        
        Output: None
        """
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
#            self.mags, self.lim_mags = mags, lim_mags           
#            self.mags.sort(['ra','dec','MJD'])
#            self.lim_mags.sort(['ra','dec','MJD'])
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
#                             fits.BinTableHDU(self.mags),
#                             fits.BinTableHDU(self.lim_mags)])    
#        hdul.writeto(LC_file) # write it


    def copy(self):
        """
        Input: None        
        Produces a (deep) copy of the object.        
        Output: the copy 
        """
        return deepcopy(self)

    ### ADDING TABLES/ADDING TABLES FROM FILES ################################    
    ### FOR NORMAL/LIMITING MAGNITUDES       
    def add_tables(self, *tables):
        """
        Input: one or more tables to append to the end of the LightCurve's mag 
        table and/or limiting mag table
        
        Adds one or more new tables to the existing LightCurve's mag table 
        and/or limiting mag table. Changes the LightCurve in place.
        Always assumes that the tables are either pure magnitudes or limiting 
        magnitudes. For adding in reference magnitudes, see add_ref_tables()

        Output: None
        """
        
        for t in tables:
            # if table contains actual aperture magnitudes
            if "mag_calib_unc" in t.colnames:
                LightCurve.__mag_table_append(self, t.copy())
            # if table contains limiting magnitudes
            else: 
                LightCurve.__limmag_table_append(self, t.copy())


    def add_files(self, *files):
        """
        Input: one or more files from which to read tables and then append to 
        the end of the LightCurve's mag table and/or limiting mag table

        Reads in and then adds one or more new tables to the existing 
        LightCurve's mag table and/or limiting mag table. Changes the 
        LightCurve in place.
        Always assumes that the tables are either pure magnitudes or limiting 
        magnitudes. For adding in reference magnitudes, see add_ref_files()
              
        Output: None
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
        """
        Input: one or more tables to append to the end of the LightCurve's 
        REFERENCE mag table
        
        Adds one or more new tables to the existing LightCurve's REFERENCE mag 
        table. Changes the LightCurve in place.

        Output: None
        """        
        for t in tables: LightCurve.__refmag_table_append(self, t.copy())


    def add_ref_files(self, *files):
        """
        Input: one or more files from which to read tables and then append to 
        the end of the LightCurve's mag table and/or limiting mag table

        Reads in and then adds one or more new tables to the existing 
        LightCurve's REFERENCE mag table. Changes the LightCurve in place.
              
        Output: None
        """
        for f in files: LightCurve.__refmag_file_append(self, f)     


    ### ADDING DISCRETE NORMAL/LIMTIING/REFERENCE MAGNITUDES ##################
    def add_mag(self, ra, dec, mag, mag_err, filt, mjd):
        """
        Input: 
            - RA and Dec
            - magnitude and its error
            - filter used
            - time of observation in MJD 
            
        Manually adds a single point to the LightCurve's mag table. 
        
        Output: None
        """
        pt = Table(names=self.mag_colnames, 
                   data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])
        
        LightCurve.add_tables(self, pt)
     
        
    def add_limmag(self, ra, dec, mag, filt, mjd):
        """
        Input: 
            - RA and Dec
            - limiting magnitude
            - filter used
            - time of observation in MJD 
            
        Manually adds a single point to the LightCurve's LIMITING mag table. 
        
        Output: None
        """
        lm = Table(names=self.limmag_colnames, 
                   data=[[ra],[dec],[mag],[filt],[mjd]])        

        self.lim_mags.add_row(lm[0])
        
    
    def add_refmag(self, ra, dec, mag, filt, mjd, mag_err=None):
        """
        Input: 
            - RA and Dec
            - reference magnitude
            - filter used
            - time of observation in MJD 
            - error on magnitude (optional; default None)
            
        Manually adds a single point to the LightCurve's REFERENCE mag table. 
        
        Output: None
        """

        rm = Table(names=self.mag_colnames, 
                   data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])    
            
        self.ref_mags.add_row(rm[0])

    ### RETURN A NEW LIGHTCURVE OBJECT FOR ONLY ONE COORDINATE ################
    def coords_select(self, ra, dec, sep=1.0):
        """
        Input:
            - ra, dec of interest
            - separation from the ra, dec to probe (in arcsec; optional; 
              default 1.0")
            
        Output: a new LightCurve object containing only sources within <sep> 
        arcsec of the input ra, dec
        """
        # target of interest
        toi = SkyCoord(ra, dec, frame='icrs', unit='degree') 
        
        # coords of magnitude/limiting magnitude table
        LC_mag_coords = SkyCoord(ra=self.mags['ra'], dec=self.mags["dec"], 
                                 frame='icrs', unit='degree')
        LC_limmag_coords = SkyCoord(ra=self.lim_mags['ra'], 
                                    dec=self.lim_mags["dec"], 
                                    frame='icrs', unit='degree')
        LC_refmag_coords = SkyCoord(ra=self.ref_mags['ra'], 
                                    dec=self.ref_mags["dec"], 
                                    frame='icrs', unit='degree')
        
        mask = (toi.separation(LC_mag_coords) <= sep*u.arcsec)
        mags_match = self.mags[mask]
        mask = (toi.separation(LC_limmag_coords) <= sep*u.arcsec) 
        limmags_match = self.lim_mags[mask]
        mask = (toi.separation(LC_refmag_coords) <= sep*u.arcsec) 
        refmags_match = self.ref_mags[mask]
        
        mags_match.pprint()
        limmags_match.pprint()
        refmags_match.pprint()
        
        return LightCurve(mag_tab=mags_match, lim_mag_tab=limmags_match,
                          ref_mag_tab=refmags_match)
        

    ### SETTING THE PLOTTING INSTRUCTIONS #####################################   
    def set_plot_instructions(self, filename):
        """
        Input: filename for a .csv which contains columns (WITHOUT headers) of 
               the form 
        
               u    red    o
               g    #00ffff s
               ...
        
               indicating the filter, marker colour, and marker style to use. 
               Will NOT change the marker style of limiting magnitudes, which 
               is fixed at "v"(downwards caret).
               
        
        Output: None
        """
        import pandas as pd
        df = pd.read_csv(filename, header=None) # read in csv
        
        if df.isnull().values.any(): # if any nans in any columns, reject
            print("\nFound a nan/empty cell in the table. Please fill this "+
                  "cell and try again. Exiting.")
            return
        
        # build dictionaries for mags, limiting mags, and reference mags
        magdict = dict(zip(df[0].values,
                           [[df[1].tolist()[i], df[2].tolist()[i]] for i in 
                             range(len(df))]))
        limdict = dict(zip(df[0].values,
                           [[df[1].tolist()[i], "v"] for i in range(len(df))]))
        refdict = dict(zip(df[0].values, df[1].values))
        
        # update the object
        self.plot_instructions = magdict
        self.plot_instructions_lim_mags = limdict
        self.plot_instructions_ref_mags = refdict

    
    ### PLOTTING ##############################################################
    def plot(self, *filters, output="lightcurve.png", title=None, 
             tmerger=None, show_legend=True, connect=True, text=None,
             mag_min=None, mag_max=None, 
             limmag_min=None, limmag_max=None, 
             refmag_min=None, refmag_max=None):
        """
        Input: 
            - filter/filters of choice to plot the lightcurve for only these 
              filters (optional; default is to plot for all filters; options 
              are "u", "g", "r", "i", "z", "Y", "J", "H", "K")
            - name for the file to be saved (optional; default lightcurve.png;
              can be set to None to not save anything)
            - title for the plot (optional; default no title)
            - value in MJD for the time of merger (or some general reference 
              time which should be considered t=0) to plot time elapsed since 
              t=0 (optional; default None, which just plots MJD)
            - whether to show the legend (optional; default True)
            - whether to connect points of the same filter (optional;
              default True)
            - text to place in a textbox (optional; default None; must be a 
              tuple or list of the form (x, y, 'text you want printed'))
            - lower, upper limits on the magnitudes in the light curve 
              (optional; default None)
            - lower, upper limits on the limiting magnitudes in the light 
              curve (optional; default None)
            - lower, upper limits on the reference magnitudes in the light 
              curve (optional; default None)
              
        NOTE: "lower" and "upper" limits understood in terms of the magnitude 
        system. i.e., m=26 is a lower limit, m=21 is an upper limit.
              
        Output: None
        """
        if D_OFF: plt.switch_backend('agg')
        
        if (len(self.mags) == 0)  and (len(self.lim_mags) == 0):
            print("\nLightCurve object has no magnitude or limiting magnitude"+
                  " data points. Cannot plot. Exiting.")
            return
        
        plotted_filts = [] # keep track of filters which have been plot already
        
        self.mags.sort("MJD")
        self.lim_mags.sort("MJD")
        self.ref_mags.sort("MJD")
        
        ## plot magnitudes and their errors
        if len(self.mags) > 0:
            sources = self.mags  
            
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
                color, form = self.plot_instructions[filt]
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
                    mask = self.mags["filter"] == f
                    color, __ = self.plot_instructions[f]
                    trelevant = np.array(t)[mask]
                    magrelevant = np.array(mag)[mask]
                    plt.plot(trelevant, magrelevant, marker="", ls="-", 
                             lw=2.0, zorder=0, color=color, alpha=0.6)
        
        ## plot limiting magnitudes 
        if len(self.lim_mags) > 0:
            lims = self.lim_mags
            
            if filters: # if a filters argument is given
                mask = (lims["filter"] == filters[0])
                for filt in filters[1:]:
                    mask += (lims["filter"] == filt)
                lims = lims[mask] # only use those filters

            # if limits on limiting magnitudes 
            if limmag_min:
                print(lims["mag_calib"] < limmag_min)
                lims = lims[lims["mag_calib"] < limmag_min]
                print(lims)
            if limmag_max:
                print(lims["mag_calib"] > limmag_max)
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
                color, form = self.plot_instructions_lim_mags[filt]
                if filt in plotted_filts:
                    plt.plot(t[i], mag[i], marker=form, mfc=color, mec="black", 
                             mew=2.0, ls="", ms=24.0, zorder=3)     
                else:
                    plt.plot(t[i], mag[i], marker=form, mfc=color, mec="black", 
                             mew=2.0, ls="", label=filt, ms=24.0, zorder=3)  
                plotted_filts.append(filt)    

        ## plot reference magnitudes 
        if len(self.ref_mags) > 0:
            refs = self.ref_mags
            
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
                color = self.plot_instructions_ref_mags[filt]
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
            valid_filts = ["u","g","r","i","z","Y","J","H","K"]
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
        
        #if D_OFF: plt.close()
        #else: plt.show() 
        plt.close()
        
        return fig
   
##### testing
#a = fromdirectory("1_tmp/")
#a.add_refmag(14.070877361773102, -26.5402713165586, 21.11611255086921, "u", 
#             58713.546265)
#a.add_limmag(14.070877361773102, -26.5402713165586, 23.51611255086921, "u", 
#             58713.546265)
#a.add_refmag(14.070877361773102, -26.5402713165586, 21.11611255086921, "u", 
#             58713.546265, 0.69)
#a.add_limmag(14.070877361773102, -26.5402713165586, 19.11611255086921, "J", 
#             58719.546265)
#a.add_refmag(14.070877361773102, -26.5402713165586, 23.51611255086921, "g", 
#             58713.546265, 0.75)
#a.add_mag(14.070877361773102, -26.5402713165586, 17.51611255086921, 0.35, "Ks", 
#          58716.546265)
#a.add_mag(14.070877361773102, -26.5402713165586, 17.51611255086921, 0.35, "g", 
#          58715.546265)
#a.add_refmag(14.070877361773102, -26.5402713165586, 21.51611255086921, "r", 
#             58713.546265, 0.11)
#a.add_ref_files(*glob.glob("1_tmp/*.fits"))
#a.plot(tmerger=58709.88239578551)        
