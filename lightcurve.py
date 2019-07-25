#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:43:27 2019

@author: nvieira
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import OrderedDict
from astropy.table import Table, Column

def fromfile(readfile):
    """
    Input: a single file from which to build a LightCurve object
    Output: the new LightCurve object
    """
    tab = Table.read(readfile, format="ascii.ecsv")
    return LightCurve(tab)

def fromdirectory(directory):
    """
    Input: a single directory to look for .fits files and use all of them to
    build a LightCurve object
    Output: the new LightCurve object
    """
    all_files = np.array(os.listdir(directory))
    files = tuple([directory+"/"+f for f in all_files if ".fits" in f])
    ret = fromfile(files[0]) # load in first
    ret.add_points_from_files(*(files[1:])) # load in the rest 
    return ret

def frompoint(ra, dec, mag, mag_err, filt, mjd):
    """
    Input: the RA and Dec, magnitude and its error, the filter used, and 
    the time of observation in MJD for a point to initialize a LightCurve
    Output: the new LightCurve object
    """
    tab = Table(names=["ra","dec","mag_calib","mag_calib_unc","filter","MJD"], 
               data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])
    return LightCurve(tab)

class LightCurve:
    def __init__(self, source_table=Table()):
        self.source_table = source_table
        # a dictionary containing instructions on how to plot the lightcurve 
        # based on the filter being used (marker color, marker style) 
        self.plot_instructions = {"u":["#9a0eea","o"], 
                                  "g":["green","o"],
                                  "r":["#ff2100","o"],
                                  "i":["#ffb07c","o"],
                                  "z":["#ff796c","o"],
                                  "Y":["#c65102","o"],
                                  "J":["#ff028d","*"],
                                  "H":["#fac205","*"],
                                  "Ks":["#0652ff","*"]}
        self.plot_instructions_lim_mags = {"u":["#9a0eea",7], 
                                           "g":["green",7],
                                           "r":["#ff2100",7],
                                           "i":["#ffb07c",7],
                                           "z":["#ff796c",7],
                                           "Y":["#c65102",7],
                                           "J":["#ff028d",7],
                                           "H":["#fac205",7],
                                           "Ks":["#0652ff",7]}
        self.limiting_mags = Table(names=["ra", "dec", "mag", "filter", "MJD"],
                                   dtype=[np.dtype(float),np.dtype(float),
                                          np.dtype(float),np.dtype(str),
                                          np.dtype(float)])
        self.exist_limiting_mags = False                               
    
    
    def read(self, readfile):
        """
        Input: a file to read table data from
        Reads a table from a .fits file and sets this table to be the source 
        table of the given LightCurve object. 
        Output: None
        """
        read_table = Table.read(readfile, format="ascii.ecsv")
        self.source_table = read_table

    
    def write(self, datafile, overwrite=False):
        """
        Input: a datafile to write the table data to, and a bool indicating 
        whether or not to overwrite if the file already exists (default False,
        in which case the new data is just appended to the old)
        Output: None
        """
        cwd = os.getcwd() # current working directory 
        
        # check if file is already present and populated
        if (datafile in os.listdir(cwd)) and (os.stat(datafile).st_size!=0):
            if overwrite: # if we want to overwrite
                print("\nFile exists. Overwriting as desired.")
                self.source_table.write(datafile, overwrite=True, 
                                        format="ascii.ecsv")
            else: # if we want to append 
                print("\nFile exists. Appending new data. To overwrite "+
                      "instead, use overwrite=True flag when calling the "+
                      "write function.")
                self.source_table = Table.read(datafile, format="ascii.ecsv")
                LightCurve.__table_append(self, self.source_table) # append 
                self.source_table.write(datafile, overwrite=True, 
                                        format="ascii.ecsv")
              
        # if not present or present and empty 
        else: 
            self.source_table.write(datafile, overwrite=True, 
                                    format="ascii.ecsv")
        
                
    def __table_append(self, table_new):
        """
        Input: an astropy table table_new to append to the object's current 
        source table. Tables must be of format ascii.ecsv to prevent loss of 
        information. Changes the object's table in place.
        Output: None 
        """
        for r in table_new:
            self.source_table.add_row(r)

    def __file_append(self, file):
        """
        Input: a file to read a table from to then append to the object's 
        existing table
        Output: None
        """
        t = Table.read(file, format="ascii.ecsv")
        LightCurve.__table_append(self, t)
           
            
    def add_points_from_tables(self, *tables):
        """
        Input: one or more tables to append to the end of the the object's 
        table of data. 
        Output: None
        """
        for t in tables:
            LightCurve.__table_append(self, t)


    def add_points_from_files(self, *files):
        """
        Input: one or more files to read from, appending the tables in each 
        file to the end of the object's existing table 
        Output: None
        """
        for f in files:
            LightCurve.__file_append(self, f)
            
            
    def add_point(self, ra, dec, mag, mag_err, filt, mjd):
        """
        Input: the RA and Dec, magnitude and its error, the filter used, and 
        the time of observation in MJD for a point to be manually added to the 
        light curve
        Output: None
        """
        pt = Table(names=["ra","dec","mag_calib","mag_calib_unc","filter","MJD"], 
                   data=[[ra],[dec],[mag],[mag_err],[filt],[mjd]])
        
        LightCurve.add_points_from_tables(self, pt)
        
    def add_limiting_magnitude(self, ra, dec, mag, filt, mjd):
        """
        Input: the RA and Dec, magnitude, filter used, and time of observation 
        in MJD for a point which is a limiting magnitude measurement
        Output: None
        """
        lm = Table(names=["ra","dec","mag_calib","filter","MJD"], 
                   data=[[ra],[dec],[mag],[filt],[mjd]])        
        
        #### NOT WORKING YET ####
        print(lm)
        self.limiting_mags.add_row(lm[0])
        self.limiting_mags[-1]["filter"] = filt
        self.exist_limiting_mags = True
        
        
    def plot(self, filename, *filters):
        """
        Input: a filename to which the plot is saved and a filter/filters of 
        choice to plot the lightcurve for only these filters (optional; 
        default is to plot for all filters)
        Output: None
        """
        sources = self.source_table
        ra_to_avg = sources["ra"].data
        dec_to_avg = sources["dec"].data
        
        if filters: # if a filters argument is given
            mask = sources["filter"] == filters[0]
            for filt in filters[1:]:
                mask += sources["filter"] == filt
            sources = sources[mask]
        
        t = sources["MJD"].data
        mag = sources["mag_calib"].data
        mag_err = sources["mag_calib_unc"].data
        
        plt.figure()
        for i in range(len(t)):
            filt = str(sources["filter"].data[i])
            color, form = self.plot_instructions[filt]
            plt.errorbar(t[i], mag[i], mag_err[i], fmt=form, mfc=color, 
                         mec=color, ls="", color="black", label=filt)
            
        if self.exist_limiting_mags:
            lims = self.limiting_mags
            t = lims["MJD"].data
            mag = lims["mag"].data
            
            for i in range(len(lims)):
                filt = str(lims["filter"].data[i])
                color, form = self.plot_instructions_lim_mags[filt]
                plt.plot(t[i], mag[i], ls="", mfc=color, mec=color, ms=10.0,
                         marker=form, label=filt)
            
            ra_to_avg = np.append(ra_to_avg, self.limiting_mags["ra"])
            dec_to_avg = np.append(dec_to_avg, self.limiting_mags["dec"])        

        ra = np.mean(sources["ra"].data)
        dec = np.mean(sources["dec"].data)
        
        # remove duplicate labels/handles 
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels,handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper left")
        
        plt.xlabel("MJD", fontsize=14)
        plt.ylabel("Magnitude", fontsize=14)
        plt.gca().invert_yaxis()
        plt.title("Source at RA=%.5f, Dec=%.5f"%(ra,dec))
        plt.savefig(filename)
                
        