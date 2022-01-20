#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Thu May 9 11:08:26 2019
.. @author: Nicholas Vieira
.. @alala.py

SECTIONS:
    - `RawData` class 
        - Basic functions
        - Image diagnostics + pre-solving before stacking
        - Locating coordinates among raw data + writing extensions 
        - Combining/dividing (WIRCam) cubes
        - Cropping images
        - Stacking and stack preparation (bad pixel masks)
        
    - `Stack` class 
        - Bad pixel masks/source masks/error arrays
        - Making images (plots)
        - Astrometry
        - PSF photometry
        - Aperture photometry
        - PSF/aperture photometry comparison
        - Source selection

"""

import os
import glob
from subprocess import run
import re
from timeit import default_timer as timer
from copy import deepcopy

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from scipy.ndimage import zoom

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.stats import (gaussian_sigma_to_fwhm, sigma_clipped_stats, 
                           SigmaClip)
from astropy.visualization import simple_norm
from photutils import (Background2D, MMMBackground, 
                       make_source_mask, detect_sources, source_properties, 
                       SkyCircularAperture, SkyCircularAnnulus,
                       aperture_photometry)

###############################################################################
### ERRORS ####################################################################

class NoDataError(Exception):
    """Raise this error when a RawData object has no data."""
    pass


class TooFewMatchesError(Exception):
    """Raise this error when, during PSF photometry, the no. of sources in the 
    image which match those in some external catalog is less than 3."""
    pass


###############################################################################
### CONSTANTS #################################################################

VALID_DATA_EXT = ('fits', 'fits.fz', 'flt')
"""Valid file types for the image data."""

VALID_PLOT_EXT = ('pdf', 'jpg', 'bmp', 'png')
"""Valid file types for any plots which are produced."""


###############################################################################
### RawData CLASS #############################################################

class RawData:
    def __init__(self, data_dir, stack_dir=None, qso_grade_limit=None,
                 fmt="fits", plot_ext="png"):
        """Basic class for doing image diagnostics, making stacks, doing 
        photometry, etc.
        
        Arguments
        ---------
        data_dir : str
            Directory containing the raw data
        stack_dir : str, optional
            Directory in which to store stacked images (default None --> no 
            stacks)
        qso_grade_limit : int, optional
            Impose a limit on the queue service observer (QSO) grade applied 
            to the observations, where 1=good and 5=unusable (default None -->
            impose no limit; only relevant for WIRCam)
        fmt : {'fits', 'fits.fz', 'flt'}, optional
            Format of the data files (default 'fits')
        plot_ext : {'pdf', 'jpg', 'bmp', 'png'}, optional
            File format for any output plots (default 'png')
            
        Returns
        -------
        alala.RawData
            A RawData object for the input data, containing either CFHT 
            WIRCam or MegaPrime data

        """

        # fix directory names, if needed
        if data_dir[-1] == "/":
            data_dir = data_dir[:-1]
        if not(type(stack_dir) == type(None)):
            if stack_dir[-1] == "/":
                stack_dir = stack_dir[:-1]

        # check if the directories exist
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Directory '{data_dir}' does not exist")
        if not(type(stack_dir) == type(None)) and not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Directory '{stack_dir}' does not exist")            
        
        # directories
        self.__data_dir = data_dir # location of data
        self.__stack_dir = stack_dir # location of stacks to be produced

        
        # file format of the data
        if not fmt in VALID_DATA_EXT:
            raise ValueError(f"Expected fmt one of {VALID_DATA_EXT}"+
                             f"got '{fmt}'")
        self.__fmt = fmt
        
        # get the files 
        self.__files = (os.listdir(self.data_dir)) # all files
        (self.__files).sort()
        self.__files = [f for f in self.__files if fmt in f] # fits only
        # check if any are present
        if len(self.__files) == 0:
            raise NoDataError(
                    f"Directory {self.data_dir} does not contain any files "+
                    f"with extension '.{fmt}'")    

        # file extension for plots?
        if not plot_ext in VALID_PLOT_EXT:
            raise ValueError(f"Expected plot_ext one of {VALID_PLOT_EXT},"+
                             f"got '{plot_ext}'")
        self.__plot_ext = plot_ext        
                    
        # ***assume all extensions and files have same instrument/nextend:
        file = self.files[0]
        hdu = fits.open(f"{self.data_dir}/{file}")[0]
        self.__instrument = hdu.header["INSTRUME"] # instrument in use
        try:
             self.__nextend = int(hdu.header["NEXTEND"]) # no. of extensions
        except KeyError:
            self.__nextend = 0
        
        # check the QSO grade of the files and remove those with a grade below 
        # the limit; this DOES NOT delete the files
        self.__qso_grade_limit = qso_grade_limit
        if "WIRCam" in self.instrument and type(qso_grade_limit) == int:
            temp = []
            for f in self.files:
                if self.__nextend == 0: # if a single image
                    hdr = fits.open(f"{self.data_dir}/{f}")[0].header
                else: # if a cube
                    hdr = fits.getheader(f"{self.data_dir}/{f}")
                if int(hdr["QSOGRADE"]) <= qso_grade_limit: 
                    temp.append(f)
            self.__files = temp
                
        # check if data spans a single date or multiple dates
        # lazy check: only checking first and last files 
        # date is of format YYYY-MM-DDTHH:mm:ss, e.g. 2011-09-12T08:36:19
        # first 10 chars only and remove all - chars
        date_start = (
            fits.open(f"{self.data_dir}/{self.files[0]}")[0]).header["DATE"]
        date_start = (date_start[0:10]).replace("-","")
        date_end = (
            fits.open(f"{self.data_dir}/{self.files[-1]}")[0]).header["DATE"]
        date_end = (date_end[0:10]).replace("-","")
        
        if date_start == date_end: # same day month year 
            self.__date = (hdu.header["DATE"][0:10]).replace("-","")
        elif date_start[0:6] == date_end[0:6]: # if same month and year
                self.__date = date_start[0:6] 
        elif date_start[0:4] == date_end[0:4]: # same year
            self.__date  = date_start[0:4]
        else:            
            self.__date = "multiyear"
        self.__dates_init() 
            
        # Modified Julian Date (MJD) at start of observations for *first* file 
        # in directory
        self.__mjdate = (fits.open(
                f"{self.data_dir}/{self.files[0]}")[0]).header["MJDATE"]
        
        # choose potential filters based on instrument
        if "WIRCam" in self.instrument:
            # broadband filters:
            self.__J = [] # 1253 +/- 79
            self.__H = [] # 1631 +/- 144.5
            self.__Ks = [] # 2146 +/- 162.5
            # narrow-band filters:
            self.__Y = [] # 1020 +/- 50
            #self.__OH_1 = []
            #self.__OH_2 = []
            #self.__CH4_on = []
            #self.__CH4_off = []
            #self.__W = []
            #self.__H2 = []
            #self.__K_cont = []
            #self.__bracket_gamma = []
            #self.__CO = []
            self.__filters=['Y','J','H','Ks'] 
            #self.__filters=["Y","J","H","Ks","OH-1","OH-2","CH4_on","CH4_off",
            #              "W","H2","K_cont","bracket_gamma","CO"]
        
        elif "MegaPrime" in self.instrument:
            if self.mjdate > 57023: # if after 1 January 2015
                ### new filters system
                self.__u = [] # 355 +/- 43
                self.__g = [] # 475 +/- 77
                self.__r = [] # 640 +/- 74
                self.__i = [] # 776 +/- 77.5
                self.__z = [] # 925 +/- 76.5
                # since 2015A, old filters denoted with trailing S
                # they were retired in 2017, but for a brief period, PIs could
                # use both the old and the new 
                self.__uS = [] # 375 +/- 37
                self.__gS = [] # 487 +/- 71.5
                self.__rS = [] # 630 +/- 62
                self.__iS = [] # 770 +/- 79.5
                self.__zS = [] # N/A, 827 to ...
                self.__filters = ["u","g","r","i","z","uS","gS","rS","iS","zS"]
            else:  
                ### old filters system
                self.__u = [] # 375 +/- 37
                self.__g = [] # 487 +/- 71.5
                self.__r = [] # 630 +/- 62
                self.__i = [] # 770 +/- 79.5
                self.__z = [] # N/A, 827 to ...
                self.__filters = ['u','g','r','i','z']
              
        # construct lists of files for each filter as well as a dictionary
        self.__filter_init()
        
        # for later
        self.__stack_made = False
        
        
        
        
    def __filter_init(self):
        """Records which filters are used in each of the image data files. 
        Constructs a dictionary of the form {filter --> [file1,file2,...]}, 
        e.g., {'i' --> ['file1','file2','file3']}
        """

        # assume all extensions have same filter for a given file
        if "WIRCam" in self.instrument:  # if WIRCam data
            # broadband filters:
            self.__J = [] # 1253 +/- 79
            self.__H = [] # 1631 +/- 144.5
            self.__Ks = [] # 2146 +/- 162.5
            # narrow-band filters:
            self.__Y = [] # 1020 +/- 50
            #self.__OH_1 = []
            #self.__OH_2 = []
            #self.__CH4_on = []
            #self.__CH4_off = []
            #self.__W = []
            #self.__H2 = []
            #self.__K_cont = []
            #self.__bracket_gamma = []
            #self.__CO = []
            self.__filters=['Y','J','H','Ks'] 
            #self.__filters=["Y","J","H","Ks","OH-1","OH-2","CH4_on","CH4_off",
            #              "W","H2","K_cont","bracket_gamma","CO"]
            
            for f in self.files:
                hdu_temp = fits.open(f"{self.data_dir}/{f}")
                hdu = hdu_temp[0]
                if 'Y' in hdu.header["FILTER"]:
                    self.__Y.append(f)
                elif 'J' in hdu.header["FILTER"]:
                    self.__J.append(f)
                elif 'H' in hdu.header["FILTER"]:
                    self.__H.append(f)     
                elif 'Ks' in hdu.header["FILTER"]:
                    self.__Ks.append(f)
                hdu_temp.close()
                
            filter_vals = [self.__Y, self.__J, self.__H, self.__Ks]
                    
        else: # if MegaPrime data
            self.__u = [] # 355 +/- 43
            self.__g = [] # 475 +/- 77
            self.__r = [] # 640 +/- 74
            self.__i = [] # 776 +/- 77.5
            self.__z = [] # 925 +/- 76.5
            # since 2015A, old filters denoted with trailing S
            # they were retired in 2017, but for a brief period, PIs could
            # use both the old and the new 
            self.__uS = [] # 375 +/- 37
            self.__gS = [] # 487 +/- 71.5
            self.__rS = [] # 630 +/- 62
            self.__iS = [] # 770 +/- 79.5
            self.__zS = [] # N/A, 827 to ...
            
            for f in self.files:
                hdu_temp = fits.open(f"{self.data_dir}/{f}")
                hdu = fits.open(f"{self.data_dir}/{f}")[0]
                if 'u' in hdu.header["FILTER"]:
                    self.__u.append(f)
                elif 'g' in hdu.header["FILTER"]:
                    self.__g.append(f)
                elif 'r' in hdu.header["FILTER"]:
                    self.__r.append(f)     
                elif 'i' in hdu.header["FILTER"]:
                    self.__i.append(f)
                elif 'z' in hdu.header["FILTER"]:
                    self.__z.append(f)
                elif 'uS' in hdu.header["FILTER"]:
                    self.__uS.append(f)
                elif 'gS' in hdu.header["FILTER"]:
                    self.__gS.append(f)
                elif 'rS' in hdu.header["FILTER"]:
                    self.__rS.append(f)
                elif 'iS' in hdu.header["FILTER"]:
                    self.__iS.append(f)
                elif 'zS' in hdu.header["FILTER"]:
                    self.__zS.append(f)
                hdu_temp.close()
            
            if self.mjdate > 57023: # if after 1 January 2015
                self.__filters = ["u", "g", "r", "i", "z",
                                  "uS", "gS", "rS", "iS", "zS"]
                filter_vals = [self.__u, 
                               self.__g, 
                               self.__r, 
                               self.__i, 
                               self.__z, 
                               self.__uS, 
                               self.__gS, 
                               self.__rS, 
                               self.__iS, 
                               self.__zS]
            else: 
                self.__filters = ["u", "g", "r", "i", "z"]
                filter_vals = [self.__u, 
                               self.__g, 
                               self.__r, 
                               self.__i, 
                               self.__z]
        
        # make a dictionary
        self.__filters_dict = dict(zip(self.filters, filter_vals))
        
        # get rid of unecessary filters in the dict/list
        all_filters = list(self.filters) # make a copy 
        for fil in all_filters:
            if len(self.filters_dict[fil]) == 0: # if no files for a filter
                del self.__filters_dict[fil]
                delattr(self, fil)
                self.__filters.remove(fil)

                
    def __dates_init(self):
        """Records which dates are spanned by each of the image data files. 
        Constructs a dictionary of the form {date --> [file1, file2, file3]}, 
        e.g., {'20190816' --> ['file1', 'file2', 'file3']} 
        """
        self.__dates = []
        self.__dates_dict = {}
        for f in self.files:
            hdu_temp = fits.open(f"{self.data_dir}/{f}")
            hdu = hdu_temp[0]
            date = (hdu.header["DATE"][0:10]).replace("-","")
            hdu_temp.close()
            if not (date in(self.dates)):
                self.__dates.append(date) # add to list
                self.__dates_dict[date] = [] # add to dict
            self.__dates_dict[date].append(f)        
            

###############################################################################
### GETTERS ###################################################################

    @property
    def data_dir(self):
        """Directory containing the raw data"""
        return self.__data_dir
    
    @property
    def stack_dir(self):
        """Directory in which to store stacked images, if stacking"""
        return self.__stack_dir
    
    @property
    def fmt(self):
        """Format (file type/extension) of the data files"""
        return self.__fmt

    @property
    def files(self):
        """All of the data files"""
        return self.__files
    
    @property
    def plot_ext(self):
        """Format (file type/extension) of any plots made/to be made"""
        return self.__plot_ext

    @property
    def instrument(self):
        """Instrument (WIRCam, MegaCam, or MegaPrime) used to acquire data""" 
        return self.__instrument
    
    @property
    def nextend(self):
        """Number of extensions in the image data files"""
        return self.__nextend
    
    @property
    def qso_grade_limit(self):
        """Queue Service Observer (QSO) grade limit; only relevant if using 
        WIRCam data"""
        return self.__qso_grade_limit
    
    @property
    def date(self):
        """Date (YYYYMMDD) **at the start** of observations""" 
        return self.__date
    
    @property
    def mjdate(self):
        """Modified Julian Date (MJD) **at the start** of observations"""
        return self.__mjdate
    
    @property
    def dates(self):
        """All date(s) (YYYYMMDD) spanned by the data"""
        return self.__dates
    
    @property
    def dates_dict(self):
        """Dictionary with entries {date --> [file1,file2,...]}`, e.g., 
        '20190816' --> ['file1','file2','file3']"""
        return self.__dates_dict
    
    @property
    def filters(self):
        """Filters (bands) used in the observations which make up the data"""
        return self.__filters

    @property
    def filters_dict(self):
        """Dictionary with entries {filter --> [file1,file2,...]}`, e.g., 
        'i' --> ['file1','file2','file3']"""
        return self.__filters
    
    @property
    def J(self):
        """Filenames for images acquired in the J-band"""
        return self.__J
    
    @property
    def H(self):
        """Filenames for images acquired in the H-band"""
        return self.__H
    
    @property
    def Ks(self):
        """Filenames for images acquired in the Ks-band"""
        return self.__Ks
    
    @property 
    def Y(self):
        """Filenames for images acquired in the Y-band"""
        return self.__Y
    
    @property 
    def u(self):
        """Filenames for images acquired in the u-band"""
        return self.__u
    
    @property
    def g(self):
        """Filenames for images acquired in the g-band"""
        return self.__g
    
    @property
    def r(self):
        """Filenames for images acquired in the r-band"""
        return self.__r
    
    @property
    def i(self):
        """Filenames for images acquired in the i-band"""
        return self.__i
    
    @property
    def z(self):
        """Filenames for images acquired in the z-band"""
        return self.__z

    @property 
    def uS(self):
        """Filenames for images acquired in the uS-band"""
        return self.__uS
    
    @property
    def gS(self):
        """Filenames for images acquired in the gS-band"""
        return self.__gS
    
    @property
    def rS(self):
        """Filenames for images acquired in the rS-band"""
        return self.__rS
    
    @property
    def iS(self):
        """Filenames for images acquired in the iS-band"""
        return self.__iS
    
    @property
    def zS(self):
        """Filenames for images acquired in the zS-band"""
        return self.__zS    
        
    @property
    def stack_made(self):
        """Whether a stack has already been made for the object"""
        return self.__stack_made


###############################################################################
### SETTERS ###################################################################
            
    def set_stack_dir(self, stack_dir):
        """Set the name of the directory in which to store any stacked 
        images."""
        if stack_dir[-1] == "/": # get rid of / at end if needed 
            stack_dir = stack_dir[:-1]
            
        if not os.path.isdir(stack_dir): # check if it exists
            raise FileNotFoundError(f"Directory '{stack_dir}' does not exist") 

        self.__stack_dir = stack_dir
        

    def set_plot_ext(self, plot_ext):
        """Set the file extension of any plots which are produced."""        
        if plot_ext in VALID_PLOT_EXT:
            self.__plot_ext = plot_ext
        else:
            raise ValueError(f"Expected plot_ext one of {VALID_PLOT_EXT},"+
                             f"got '{plot_ext}'")


###############################################################################
### FILTER BY DATE, TARGET OBJECT... ##########################################
        
    def exclude_date(self, date):
        """Exclude a specific date from the data, by parsing through the image
        files' fits headers.
        
        Arguments
        ---------
        date : str
            A date of the format 'YYYYMMDD' or 'YYYYMM' or 'YYYY' to exclude 
            from the raw data
        
        """
        all_files = self.files # make a copy 
        new_files = []
        for f in all_files: # get data for every file 
            hdu_temp = fits.open(f"{self.data_dir}/{f}")
            hdu = hdu_temp[0]
            d = (hdu.header["DATE"][0:10]).replace("-","")
            hdu_temp.close()
            
            if not(date in d): # if file is NOT from the input date 
                new_files.append(f)

        if len(new_files) == 0:
            raise NoDataError("After exclusion, RawData object would have "+
                              "no remaining data") 
        
        self.__files = new_files
        self.__dates_init() # rebuild list/dict of dates
        self.__filter_init() # rebuild list/dict of filters        

    
    def exclude_object(self, obj):
        """Exclude a specific target object from the data, by parsing through 
        the image files' fits headers.
        
        Arguments
        ---------
        obj : str
            Name of a target object to exclude, e.g., 'NGC457'
            
        """
        
        all_files = self.files # make a copy 

        new_files = []
        for f in all_files: # get data for every file 
            hdu_temp = fits.open(f"{self.data_dir}/{f}")
            hdu = hdu_temp[0]
            o = hdu.header["OBJECT"]
            hdu_temp.close()
            
            if not(obj in o): # if file is NOT of the input pointing 
                new_files.append(f)

        if len(new_files) == 0:
            raise NoDataError("After exclusion, RawData object would have "+
                              "no remaining data") 
           
        self.__files = new_files
        self.__dates_init() # rebuild list/dict of dates
        self.__filter_init() # rebuild list/dict of filters

    
    def only_date(self, date):
        """Exclude **all but one** specific date in the data, by parsing 
        through the image files' fits headers.
        
        Arguments
        ---------
        date : str
            A date of the format 'YYYYMMDD' or 'YYYYMM' or 'YYYY' to select in 
            the raw data
            
        """
        
        self.__files = []
        for d in self.dates: # look at all dates 
            if date in d: # if input date is in the list of dates 
                self.__files.append(self.dates_dict[d])
        
        self.__files = self.__files[0]
        RawData.__dates_init(self) # rebuild list/dict of dates
        RawData.__filter_init(self) # rebuild list/dict of filters

        if len(self.files) == 0:
            print(f"Warning: {date} was not spanned by any of the raw data "+
                  "files. After this operation, the RawData object has no "+
                  "remaining data.")
        else:
            self.__date = date # update the date 

    
    def only_object(self, obj):
        """
        Excude **all but one** specific target object in the data to be used by 
        parsing through the image files' fits headers.
        
        Arguments
        ---------
        obj : str
            Name of the object, e.g., 'NGC457'
            
        """
        all_files = self.files # make a copy 
        self.__files = []
        for f in all_files: # get data for every file 
            hdu_temp = fits.open(f"{self.data_dir}/{f}")
            hdu = hdu_temp[0]
            o = hdu.header["OBJECT"]
            hdu_temp.close()
            
            if obj in o: # if file IS of the input pointing 
                self.__files.append(f)
        
        self.files = self.files[0]
        RawData.__dates_init(self) # rebuild list/dict of dates
        RawData.__filter_init(self) # rebuild list/dict of filters

        if len(self.files) == 0:
            print(f"Warning: {obj} was not targeted in any of the raw data "+
                  "files. After this operation, the RawData object has no "+
                  "remaining data.")
        
        
    def copy(self):
        """Produces a (deep) copy of the object."""
        return deepcopy(self)

            
###############################################################################
### IMAGE DIAGNOSTICS ######################################################### 
    
    def print_headers(self, ext, *headers):
        """Print particular fits header(s) for data files. 
        
        Arguments
        ---------
        ext : {'fits', 'fits.fz', 'flt'}
            Extension of interest for the image data to be parsed
        headers : str
            Name(s) of any header(s) of interest
            
        """
        headers = list(headers) # convert tuple to list 

        # check file type of interest
        if not ext in VALID_DATA_EXT:
            raise ValueError(f"Expected ext one of {VALID_DATA_EXT}, got "+
                             f"{ext}")
        
        # alert user that some headers are not present and remove them
        # buggy - can not handle when multiple headers are not present 
        testfile = self.files[0]
        hdu_test_temp = fits.open(f"{self.data_dir}/{testfile}")
        hdu_test = hdu_test_temp[ext]
        for h in headers:
            try:
                test = hdu_test.header[h]
                del test
            except KeyError:
                print(f"Header '{str(h).upper()}' not found.\n")
                headers.remove(h)
                continue                
        
        if len(headers) == 0:
            return # if no headers left, quit
        
        headers_string = ""
        for h in headers:
            headers_string = f"{headers_string}{h}\t"
        toprint = f"FILE\t\t\t{headers_string}"
        print(toprint)
        # print the desired headers in readable format for all raw data files
        for f in self.files:
            toprint = f"{f}\t"
            hdu = fits.open(f"{self.data_dir}/{f}")[ext]
            for h in headers:
                toprint += f"{str(hdu.header[h])}\t"
            print(toprint)
        hdu_test_temp.close()


    def value_at(self, ra, dec):
        """Get the value at some RA and Dec, for all images.
        
        Arguments
        ---------
        ra, dec : float
            Right Ascension (RA) and Declination of interest 


        For all of the files contained in the RawData object, prints the ADU 
        value at the given RA and Dec, **if** these coordinates are within the 
        image's bounds. Can be used to see if, e.g., the ADU was set to 0 
        over a source of interest during image de-trending.
        
        """
        for f in self.files:
            data = fits.getdata(f"{self.data_dir}/{f}")
            hdr = fits.getheader(f"{self.data_dir}/{f}")
            w = wcs.WCS(hdr)
            xpix, ypix = w.all_world2pix(ra, dec, 1)
            xpix = int(xpix)
            ypix = int(ypix)
            if (0 < xpix < 2048.0) and (0 < ypix < 2048.0):
                print(f"ADU at ({xpix:d}, {ypix:d}) = {data[ypix][xpix]:.2f}")
                
                
    def background(self):
        """Estimate the background of the images. 
                
        Returns
        -------
        list
            List containing the background level of each file


        Naively estimates the background as the median of the image's ADU 
        for every raw image in order to see how the data varies. 

        Notes
        -----
        Does not mask sources, but this is not important for the purpose of 
        this function: to see if any data is dubious 

        """
        bg_levels = []
        for f in self.files:
            data = fits.getdata(f"{self.data_dir}/{f}")
            bg_levels.append(np.median(data))
        return bg_levels
        
    
    def radial_PSFs(self, ra, dec, solved=True, adu_min=4000, adu_max=66000):
        """Produce radial PSFs for all images. 
        
        Arguments
        ---------
        ra, dec : float
            Right Ascension (RA) and Declination of interest         
        solved : bool, optional
            Have the images already been solved with astrometry? (default 
            True)
        adu_min, adu_max : float, optional
            Minimum and maximum allowed ADU in the image (default 4000, 66000)


        For all of the files in the raw data directory, produces and saves a 
        figure of the radial profile around the input RA and Dec. If 
        `solved=False`, first solves the image with astrometry.net. (A refined 
        astrometric solution is required to get an accurate radial profile.) A 
        crude technique for estimating the PSF of an image.
            
        Notes
        -----
        Best to pick a source which is bright, but not one which saturates the
        detectors.

        """
        topfile = re.sub(".*/", "", self.data_dir) # for a file /a/b/c, extract the "c"
        plots_dir = os.path.abspath(f"{self.data_dir}/..")
        plots_dir = f"{plots_dir}/profs_RA{ra:.3f}_DEC{dec:.3f}_{topfile}"
        run(f"mkdir -p {plots_dir}", shell=True)
        
        if not solved: # if astrometry hasn't been done yet
            solved_dir = f'{os.path.abspath("{l}/..")}/solved_{topfile}'
            print("A refined astrometric solution is required for this "+
                  "function to work. Using astrometry.net to solve the "+
                  "images now. Solved .fits files will be saved in "+
                  f"{solved_dir}\n")
            RawData.solve_all(self)
            files = os.listdir(
                    f'{os.path.abspath("{l}/..")}/solved_{topfile}')
            
        else: # if astrometry has already been done
            files = self.files
            solved_dir = self.data_dir
        
        # the radial PSF 
        for f in files: 
            image_data = fits.getdata(f"{solved_dir}/{f}")
            image_header = fits.getheader(f"{solved_dir}/{f}")
            w = wcs.WCS(image_header) # wcs object
            pix_x, pix_y = w.all_world2pix(ra, dec, 1) # pix coords of source
            
            y, x = np.indices(image_data.shape)
            r = np.sqrt((x-pix_x)**2 + (y-pix_y)**2) # radial dists from source
            r = r.astype(np.int) # round to ints 
            
            # not sure about this part yet
            # ravel flattens an array 
            tbin = np.bincount(r.ravel(), image_data.ravel()) # points per bin
            norm = np.bincount(r.ravel()) # total no. of points [?]    
            profile = tbin/norm
            
            # plot
            plt.figure(figsize=(7,7))
            plt.plot(range(len(profile)), profile, 'k.', markersize=15)
            plt.xlabel('Radial distance [pixels]', fontsize=15)
            plt.ylabel('Amplitude [ADU]', fontsize=15)
            plt.xlim(-2,20)
            plt.ylim(adu_min, adu_max) 
            plt.rc("xtick",labelsize=14)
            plt.rc("ytick",labelsize=14)
            plt.title(f"Radial profile around {ra:.5f}, {dec:.5f}",
                      fontsize=15)
            
            # annotate with date and time; filename; filter in use
            obs_date = image_header["DATE"]
            filt = image_header["FILTER"]
            box = dict(boxstyle="square", facecolor="white", alpha=0.8)
            box_y = adu_min + 0.85*(adu_max-adu_min)
            txt = f"{obs_date}\n{f}\n{filt}" 
            output_fig = f.replace(f".{self.fmt}", 
                                   f"_prof_RA{ra:.3f}_DEC{dec:.3f}"+
                                   f".{self.plot_ext}")
            plt.text(3, box_y, s=txt, bbox=box,fontsize=14)
            plt.savefig(f"{plots_dir}/{output_fig}")
            plt.close()
              

    def solve_all(self, solved_dir=None, depth=None):
        """Find the astrometric solution for all images.

        Arguments
        ---------
        solved_dir : str, optional
            Directory in which to store the solved images (default None --> 
            set below)
        depth : int, optional
            Number of stars to use in solving with astrometry.net, e.g., 
            `depth=100` will use only the 100 brightest stars (default None 
            --> no limit, use all stars)
        

        Using astrometry.net, solve all files, and put them in a new directory. 
        This is necessary when the astrometric solution obtained by CFHT is 
        inaccurate and requires refining, or if you wish to plot the radial 
        PSFs of some point in the raw data images.
        
        Notes
        -----
        For MegaCam images, detectors **(0, 1, ..., 17)**, **36**, **37** 
        (top half of camera) are oriented with North ANTI-parallel to the 
        y-axis and East parallel to the x-axis. Detectors 
        **(18, 19, ..., 35)**, **38**, **39** (bottom half of the camera) are 
        oriented with North parallel to the y-axis and East ANTI-parallel to 
        the x-axis. When trying to make a stack from images in both the top 
        and bottom halves, IRAF sometimes gets confused.
        
        """
        script_dir = os.getcwd()
        os.chdir(self.data_dir)
        
        image_header = fits.getheader(f"{self.data_dir}/{self.files[0]}")
        image_data = fits.getdata(f"{self.data_dir}/{self.files[0]}")
  
        pixscale = image_header["PIXSCAL1"] # pixel scale
        pixmin = pixscale-0.005
        pixmax = pixscale+0.005

        cent = [i//2 for i in image_data.shape]
        centy, centx = cent
        w = wcs.WCS(image_header)
        ra, dec = w.all_pix2world(centx, centy, 1) 
        radius = 0.5 # look in a radius of 0.5 degrees
        
        options = "--no-verify --overwrite --no-plot --fits-image"
        options = f"{options} --scale-low {pixmin} --scale-high {pixmax}"
        options = f"{options} --scale-units app"
        options = f"{options} --ra {ra} --dec {dec} --radius {radius}"

        if type(depth) in [float, int]: # if a float or int, convert
            options = f"{options} --depth {int(depth)}"
        elif depth:
            options = f"{options} --depth {depth}"

        options = f'{options} --match "none" --solved "none" --rdls "none"'
        options = f'{options} --corr "none" --wcs "none"'
        
        for f in self.files: # astrometry on each file 
#            options = "--no-verify --overwrite --no-plot --fits-image"
            foptions = options+" --new-fits "
            foptions += f.replace(f".{self.fmt}", "_solved.fits")
            
            # options to speed up astrometry: pixscale and rough, RA, Dec
#            options += " --scale-low "+str(pixmin)
#            options += " --scale-high "+str(pixmax)
#            options += " --scale-units app"
#            options += " --ra "+str(ra)+" --dec "+str(dec)
#            options += " --radius "+str(radius)
            
            # don't bother producing these files 
#            options += ' --match "none" --solved "none" --rdls "none"'
#            options += ' --corr "none" --wcs "none"'
            
            # stop astrometry when the solved fits file is produced
            foptions += " --cancel "+f.replace(f".{self.fmt}", "_solved.fits")
            
#            if type(depth) in [float, int]:
#                options += " --depth "+str(int(depth))
#            elif depth:
#                options += " --depth "+depth
            
            # run astrometry 
            run(f"solve-field {foptions} {f}", shell=True)
        
        # get rid of unneeded files
        run("rm *.axy", shell=True)
        run("rm *.xyls", shell=True)  
         

        # make a list of solved files, move them to a new directory, 
        # and make a list of unsolved files 
        if type(solved_dir) == type(None):
            topfile = re.sub(".*/", "", self.data_dir)
            solved_dir = f'{os.path.abspath(self.data_dir+"/..")}/solved_{topfile}'
        run(f"mkdir -p {solved_dir}", shell=True)
        run(f"rm -f {solved_dir}/*.fits", shell=True) # empty existing dir
        run(f"rm -f {solved_dir}/*.txt", shell=True) # empty existing dir
        
        solved = []
        unsolved = []
        files = [f.replace(f".{self.fmt}", "_solved.fits") for f in self.files]
        for f in files: 
            if os.path.exists(f"{self.data_dir}/{f}"):
                solved.append(f.replace("_solved.fits", f".{self.fmt}"))
                run(f"mv {self.data_dir}/{f} {solved_dir}", shell=True)
            else:
                unsolved.append(f.replace("_solved.fits", f".{self.fmt}"))
        
        # save a text file w list of unsolved files, if necessary
        if len(unsolved) != 0:
            np.savetxt(f"{solved_dir}/unsolved.txt", unsolved, fmt="%s")
            print("\nThe following images could not be solved:")
            for f in unsolved:
                print(f)
            print("\nThese filenames have been recorded in a file "+
                  f"{solved_dir}/unsolved.txt")
        
        if len(solved) != 0:
            print(f"\nSolved the following images from {self.instrument} on "+
                  f"{self.date}:")
            for f in solved:
                print(f)
            print("\nThese have been written to new solved .fits files in "+
                  solved_dir)
            
        os.chdir(script_dir)
    
    
###############################################################################
### WCS LOCATING & EXTENSION WRITING ##########################################

    def WCS_check(self, ra, dec, frac=1.0, verbose=True, checkall=True):
        """Check if the input coordinates are spanned by any of the data.
        
        Arguments            
        ---------
        ra, dec : float
            Right Ascension (RA) and Declination of interest
        frac : float, optional
            Fraction of the image to consider valid, e.g., 0.9 will 
            exclude the outermost 10% of the image (default 1.0 --> whole 
            image)
        verbose : bool, optional
            Print lots of details? (default True)
        checkall : bool, optional
            Check all files, or, just return the first matching file? 
            (default True --> check all files)
        
        Returns
        -------
        list
            List of files which span the given coordinates
        
        """
        good_files = []
        
        # set x and y limits of each detector
        if "WIRCam" in self.instrument: 
            x_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
            y_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
        else:
            x_lim = [32.0+(2048.0-32.0)*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
            y_lim = [0.0+4612.0*((1.0-frac)/2), 4612.0*((1.0+frac)/2)]
        
        if self.nextend == 0: # if images already divided up by detector/CCD
            for f in self.files: 
                hdr = fits.getheader(f"{self.data_dir}/{f}")
                w = wcs.WCS(hdr)
                naxis = int(hdr["NAXIS"])

                if naxis == 3: # if a cube
                    pix_coords = np.array(w.all_world2pix(ra,dec,1,1))
                else : # if just one image 
                    pix_coords = np.array(w.all_world2pix(ra,dec,1))
            
                # check if located in detector  
                if (x_lim[0]<pix_coords[0]<x_lim[1]) and (
                    y_lim[0]<pix_coords[1]<y_lim[1]):
                    good_files.append(f"{self.data_dir}/{f}")
                    if verbose:
                        print(f"{self.data_dir}/{f}")
                    if not(checkall): # if we only want the first file
                        return good_files
                    
        else: # if multiple detectors/CCDs
            for f in self.files: 
                n = RawData.locate_WCS(self, ra, dec)
                if n:
                    good_files.append(f"{self.data_dir}/{f}")
                    if verbose:
                        print(f"{self.data_dir}/{f} [detector {n}]")
                    if not(checkall): # if we only want the first file
                        return good_files
                        
                        
        if len(good_files) != 0:
            return good_files
        else:
            return
 
    
    def locate_WCS(self, ra, dec, frac=1.0):
        """Given some coordinates and a multi-detector image, find the number 
        ID of the detector where the coordinates are located.

        Arguments
        ---------
        ra, dec : float
            Right Ascension (RA) and Declination of interest
        frac : float, optional
            Fraction of the image to consider valid, e.g., 0.9 will 
            exclude the outermost 10% of the image (default 1.0 --> whole 
            image)

        Returns
        -------
        int
            Number ID of the detector where the coordinates are located

        """
        
        if self.nextend == 0:
            print("Cannot call locate_WCS() on a file without extensions.")
            return
        
        # assume the instruments do not drift and WCS does not significantly 
        # change from one image to another 
        testfile = self.files[0]
        hdu_list_test = fits.open(f"{self.data_dir}/{testfile}")
        
        for n in range(self.nextend): # for all extensions
            w = wcs.WCS(hdu_list_test[n+1].header) # WCS object
            
            # set x and y limits of each detector
            if "WIRCam" in self.instrument: 
                x_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
                y_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
            
                
                naxis = int(hdu_list_test[n+1].header["NAXIS"])
                if naxis == 3: # if a cube
                    pix_coords = np.array(w.all_world2pix(ra,dec,1,1))
                else : # if just one image 
                    pix_coords = np.array(w.all_world2pix(ra,dec,1))
            
            else:
                x_lim = [32.0+(2048.0-32.0)*((1.0-frac)/2), 
                         2048.0*((1.0+frac)/2)]
                y_lim = [0.0+4612.0*((1.0-frac)/2), 4612.0*((1.0+frac)/2)]
                
                pix_coords = np.array(w.all_world2pix(ra,dec,1))
            
            # check if located in detector n 
            if (x_lim[0]<pix_coords[0]<x_lim[1]) and (
                    y_lim[0]<pix_coords[1]<y_lim[1]):
                #print("Source is located in extension "+str(n+1))
                return n+1  
            
            return # nothing found 

            
    def __get_extension(self, fits_file, n_ext):
        """Extract the image data and header of a fits file's `n_ext`th 
        extension."""
        
        # WIRCam: 1 to 4
        # MegaPrime: 1 to 36 pre-2015A, 1 to 40 post-2015A 
        
        new_hdu_list = fits.open(f"{self.data_dir}/{fits_file}")
        new_hdu = new_hdu_list[n_ext] # compressed
        
        new_hdr = new_hdu.header
        new_data = new_hdu.data
        extension = fits.PrimaryHDU(data=new_data, header=new_hdr)
        new_hdu_list.close()
        
        return extension  

    
    def write_extensions(self, n_ext, exten_dir=None):
        """Write a specific extension of each multi-header fits file to 
        individual files.
        
        Arguments
        ---------
        n_ext : int
            Extension number of interest (should be at least 1, n_ext=1 --> 
            extract the 1st extension in the file which corresponds to the 0th 
            detector/CCD)
        exten_dir : str, optional
            Name for the directory which will hold the written extensions 
            (default None --> set below)

        
        Gets the header and image data for the given extension and writes them
        to a new .fits file. Does so for all raw data files. Stores them in a 
        new subdirectory. Used to extract image data for one of many CCDs / 
        detectors on either MegaPrime/WIRCam. 
        
        Notes
        -----
        - For a given dataset, once `locate_WCS()` has been used to find the 
          extension containing the WCS of interest, run this function once to 
          extract that specific extension. Can then use this newly made folder 
          for stacking.
        
        - If the extensions are themselves cubes (sometimes the case for 
          WIRCam), see `combine_WIRCam()` or `divide_WIRCam()`.
        
        - Currently only works for cubes of the form fits.fz, and produces
          files of form fits.
        
        """
        
        if n_ext < 1:
            raise ValueError(f"Expected n_ext >= 1, got {n_ext}")
        elif n_ext < 11:
            det_name = f"0{n_ext-1}"
        else:
            det_name = str(n_ext-1)
        
        if type(exten_dir) == type(None):
            # exten_dir encodes the detector number, instrument, and date
            exten_dir = f'{os.path.abspath(self.data_dir+"/..")}/det{det_name}_'
            exten_dir = f'{exten_dir}{self.instrument}_{self.date}'
        run(f"mkdir -p {exten_dir}", shell=True) # make exten_dir
        
        for f in self.files: 
            exten = RawData.__get_extension(self, f, n_ext)
            new_f = f.replace(".fits.fz", f"_det{det_name}.fits")
            exten.writeto(f"{exten_dir}/{new_f}", overwrite=True, 
                          output_verify="ignore") # write them
            
        print(f"Extracted headers/images for detector {det_name} "+
              f"of {self.instrument} on {self.date}")
        print(f"Written to new .fits files in {exten_dir}")
        
        
    def write_extensions_all(self, all_exten_dir=None):
        """Write each extension of each multi-header fits file to individual 
        files.
             
        Arguments
        ---------
        all_exten_dir : str, optional
            Name for the directory which will hold the written extensions 
            (default None --> set below)

        
        Gets the header and image data for **all** extensions of a 
        multi-header fits file and writes each extension to a new .fits file. 
        Useful for MegaPrime data where the scope moves a lot and the 
        detectors are all calibrated to the same ADU level.         
        
        Notes
        -----
        Currently only works for cubes of the form fits.fz, and produces 
        files of form fits.

        """
        
        if type(all_exten_dir) == type(None):
            # all_exten_dir encodes the instrument and date            
            all_exten_dir = f'{os.path.abspath(self.data_dir+"/..")}/dets_ALL_'
            all_exten_dir = f'{all_exten_dir}{self.instrument}_{self.date}'            
        run(f"mkdir -p {all_exten_dir}", shell=True) # make all_exten_dir

        for f in self.files:
            for n in range(self.nextend):
                if n < 10:
                    det_name = f"0{n}"
                else:
                    det_name = str(n)
                exten = RawData.__get_extension(self, f, n+1)

                new_f = f.replace(".fits.fz", f"_det{det_name}.fits")
                exten.writeto(f"{all_exten_dir}/{new_f}", overwrite=True, 
                              output_verify="ignore") # write them
            
        print("Extracted headers/images for all detectors of "+
              f"{self.instrument} on {self.date}")
        print(f"Written to new .fits files in {all_exten_dir}")
    
    
    def write_extensions_by_WCS(self, ra, dec, frac=1.0, wcs_exten_dir=None):
        """Same as `write_extensions()`, but select the extension using an RA 
        and Dec.
        
        Arguments
        ---------
        ra, dec : float
            Right Ascension (RA) and Declination of interest
        frac : float, optional
            Fraction of the image to consider valid, e.g., 0.9 will 
            exclude the outermost 10% of the image (default 1.0 --> whole 
            image)
        wcs_exten_dir : str, optional
            Name for the directory which will hold the written extensions 
            (default None --> set below)

        
        For a directory full of multi-extension fits files, gets the extensions
        which contain the input RA, Dec and writes them to a new file. 
        
        Notes
        -----
        Currently only works for cubes of the form fits.fz, and produces files 
        of form fits.
        
        """
        
        if not(wcs_exten_dir):
            # wcs_exten_dir encodes the wcs of interest, instrument, and date
            wcs_exten_dir = os.path.abspath(f"{self.data_dir}/..")
            wcs_exten_dir = f"{wcs_exten_dir}/dets_RA{ra:.3f}_DEC{dec:.3f}_"
            wcs_exten_dir = f"{wcs_exten_dir}{self.instrument}_{self.date}"
        
        # set x and y limits of each detector
        if "WIRCam" in self.instrument: 
            x_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
            y_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
        else:
            x_lim = [32.0+(2048.0-32.0)*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
            y_lim = [0.0+4612.0*((1.0-frac)/2), 4612.0*((1.0+frac)/2)]
        
        for f in self.files:
            for n in range(self.nextend):
                if n < 10:
                    det_name = f"0{n}"
                else:
                    det_name = str(n)
                exten = self.__get_extension(f, n+1)
                w = wcs.WCS(exten.header) # WCS object
                 
                naxis = int(exten.header["NAXIS"])
                if naxis == 3: # if a cube
                    pix_coords = np.array(w.all_world2pix(ra,dec,1,1))
                else : # if just one image 
                    pix_coords = np.array(w.all_world2pix(ra,dec,1))
                
                # check if located in detector n 
                if not((x_lim[0]<pix_coords[0]<x_lim[1]) and (
                        y_lim[0]<pix_coords[1]<y_lim[1])):
                    continue # continue to next extension if not 
                else: 
                    new_f = f.replace(".fits.fz", f"_det{det_name}.fits")
                    exten.writeto(f"{wcs_exten_dir}/{new_f}", overwrite=True, 
                                  output_verify="ignore") # write them
                    break # exit this for loop
                    
        print("Extracted headers/images for detectors which contain "+
              f"RA {ra:.3f}, Dec {dec:.3f} for data from "+
              f"{self.instrument} on {self.date}")
        print(f"Written to new .fits files in {wcs_exten_dir}")
        
        
    def write_source(self, name, ra, dec, frac=1.0):
        """Same as `write_extensions_by_WCS()`, but write the files to a new 
        directory with name `name`.
        
        Arguments
        ---------
        name : str
            Name of the source of interest (e.g. some transient)
        ra, dec : float
            Right Ascension (RA) and Declination of interest
        frac : float, optional
            Fraction of the image to consider valid, e.g., 0.9 will 
            exclude the outermost 10% of the image (default 1.0 --> whole 
            image)

        
        Parses all raw data and copies any images which contain the input RA, 
        Dec to a new sub-directory with the name of the source.
        """

        source_files = RawData.WCS_check(self, ra, dec, frac)
        if source_files:  
            run(f"mkdir -p {self.data_dir}/{name}", shell=True)
            for f in source_files:
                run(f"cp {f} {self.data_dir}/{name}", shell=True)
        else:
            print("\nNone of the raw data contains the input RA, Dec.")


###############################################################################    
### COMBINING/DIVIDING CUBES ##################################################
            
    def __combine_cube(self, fits_file):
        """For a file composed of multiple 2D image data arrays (i.e. a cube), 
        combine the image data into one single 2D array. Only needed for 
        WIRCam data, which sometimes contains a cube for each of the 4 
        detectors.
        """

        f = fits.open(f"{self.data_dir}/{fits_file}")[0]
        
        n_images = len(f.data) # no. of images in the cube 
        new_data = f.data[0] # first image's data
        
        # build list of Modified Julian Date of each slice
        t_isot = Time(f.header["SLDATE01"], format='isot', 
                      scale='utc') # first image's ISOT time
        t_MJD = t_isot.mjd # first image's MJD time 
        t_list = [t_MJD] 
        for n in range(1, n_images): 
            temp_image = f.data[n] # single slice 
            new_data += temp_image # add new slice to the cumulative image
            t_isot_slice = Time(f.header[f"SLDATE0{n+1}"], format="isot", 
                                scale="utc")  
            t_list.append(t_isot_slice.mjd) # append slice's MJD
            
        # update exposure time and dimensions
        new_header = f.header
        exptime = f.header['EXPTIME'] # initial exposure time
        new_header['EXPTIME'] = n_images*exptime # update exposure time (sum)
        new_header['NAXIS'] = 2 # update image dimensions (no longer a cube)
        
        # update SLDATE01 to be the average of all slice's values
        t_MJD_mean = np.mean(t_list)
        t_isot_mean = t_MJD_mean.isot
        new_header["SLDATE01"] = t_isot_mean
        # Get rid of SLDATE02? Maybe.
        
        combination = fits.PrimaryHDU(data=new_data, header=new_header)
        
        return combination


    def combine_WIRCam(self):
        """For a directory full of WIRCam images, if the images are cubes, 
        **combines** the multiple 2D arrays into single arrays. Then writes 
        this combination to a single file.
        
        Use this function once to take a folder full of cubes and turn them 
        into single-frame fits files. Can **not** be called on non-cubes. Only 
        used for WIRCam.
        """
        
        topfile = re.sub(".*/", "", self.data_dir) # for a file /a/b/c, extract the "c"
        if not("WIRCam" in self.instrument):
            print("Cannot call combine_WIRCam() except on WIRCam cubes.")
            return
        
        # combin_dir encodes the detector number, instrument, and date
        combin_dir = f'{os.path.abspath(self.data_dir+"/..")}/combined_{topfile}'
        run(f"mkdir -p {combin_dir}", shell=True) # make combin_dir
        
        for f in self.files: 
            if len((fits.getdata(f"{self.data_dir}/{f}")).shape) > 2: # if a cube 
                combin = RawData.__combine_cube(self, f)
                new_f = f.replace(".fits","_combined.fits")
                combin.writeto(f"{combin_dir}/{new_f}", overwrite=True, 
                              output_verify="ignore") # write them
            else: # if not, just copy it over without changing filename
                run(f"cp -p {self.data_dir}/{f} {combin_dir}", shell=True)

            
    def __divide_cube(self, fits_file):
        """For a file composed of multiple 2D image data arrays (i.e. a cube), 
        divides the image data into separate files. Only needed for WIRCam 
        data, which sometimes contains a cube for each of the 4 detectors.
        """

        f = fits.open(f"{self.data_dir}/{fits_file}")[0]
        
        n_images = len(f.data) # no. of images in the cube 
        new_header = f.header
        new_header["NAXIS"] = 2 # no longer a cube 

        divisions = []
        for n in range(0, n_images):  
            temp_image = f.data[n] # a single slice 
            temp_hdr = new_header
            # new header: slice ID in cube it came from (01, 02, 03...)
            temp_hdr["SLICEID"] = (f"0{n+1}", "Slice ID in original cube")
            divisions.append(fits.PrimaryHDU(data=temp_image, header=temp_hdr))
        
        return divisions # list of PrimaryHDU objects 
    
    
    def divide_WIRCam(self):
        """For a directory full of WIRCam images, if the images are cubes, 
        **divides** the multiple 2D arrays into individual arrays. Then writes 
        each division to a separate new file.
        
        Use this function once to take a folder full of cubes and turn them 
        into single-frame fits files. Can **not** be called on non-cubes. 
        Only used for WIRCam.
        """
        
        topfile = re.sub(".*/", "", self.data_dir) # for a file /a/b/c, extract the "c"
        if not("WIRCam" in self.instrument):
            print("Cannot call divide_WIRCam() except on WIRCam cubes. "+
                  "Exiting.")
            return
        
        # div_dir encodes the detector number, instrument, date
        div_dir = f'{os.path.abspath(self.data_dir+"/..")}/divided_{topfile}'
        run(f"mkdir -p {div_dir}", shell=True) # make div_dir
        
        for f in self.files: 
            if len((fits.getdata(f"{self.data_dir}/{f}")).shape) > 2: # if a cube 
                divs = RawData.__divide_cube(self, f)
                for div in divs:
                    temp_header = div.header
                    sliceid = temp_header["SLICEID"]
                    new_f = f.replace(".fits", f"_divided_{sliceid}.fits")
                    div.writeto(f"{div_dir}/{new_f}", overwrite=True, 
                                output_verify="ignore") # write them
            else: # if not, just copy it over without changing filename
                  # but assign a SLICEID
                run(f"cp -p {self.data_dir}/{f} {div_dir}", shell=True)
                temp = fits.open(f"{div_dir}/{f}", mode="update")
                temp[0].header["SLICEID"] = "01"
                temp.close()

###############################################################################
### CROP IMAGES BASED ON PIXEL FRACTIONS/WCS ##################################
        
    def __get_crop(self, fits_file, frac_hori=[0,1], frac_vert=[0,1]):
        """Crop an image. 
        
        Arguments
        ---------
        fits_file : str
            Name of a single fits file
        frac_hori, frac_vert : length-two list, optional
            Horizontal and vertical fractions of the images to crop (default 
            [0,1] for both --> no cropping)

        Returns
        -------
        astropy.io.fits.PrimaryHDU
            New fits HDU containing the header and cropped image
        
        Examples
        --------
        `__get_crop("foo.fits", [0.5,1], [0,0.5])` would crop the bottom 
        right corner of the image

        """
            
        # get data 
        data = fits.getdata(fits_file)
        hdr = fits.getheader(fits_file)
        ydim, xdim = data.shape
        
        # get the indices in the data which bound the cropped area
        idx_x = [int(frac_hori[0]*xdim), int(frac_hori[1]*xdim)]
        idx_y = [int(frac_vert[0]*ydim), int(frac_vert[1]*ydim)]
    
        # get the cropped data, build a new PrimaryHDU object
        cropped = data[idx_y[0]:idx_y[1], idx_x[0]:idx_x[1]]
        hdr["NAXIS1"] = len(idx_x) # adjust NAXIS sizes
        hdr["NAXIS2"] = len(idx_y)
        hdr["CRPIX1"] -= idx_x[0] # update WCS reference pixel 
        hdr["CRPIX2"] -= idx_y[0]
        new_hdu = fits.PrimaryHDU(data=cropped, header=hdr)
        return new_hdu
        
    
    def crop_images(self, frac_hori=[0,1], frac_vert=[0,1]):
        """Crop several fits images. 
        
        Arguments
        ---------
        frac_hori, frac_vert : length-two list, optional
            Horizontal and vertical fractions of the images to crop (default 
            [0,1] for both --> no cropping)

            
        For a directory full of WIRCam/MegaPrime images (should **not** be 
        cubes), crops the images based on the input x-axis and y-axis 
        boundaries and writes them all to a new directory.
        
        """
        
        topfile = re.sub(".*/", "", self.data_dir) # for a file /a/b/c, extract the "c"
        
        # crop_dir encodes the detector number, instrument, date
        crop_dir = f'{os.path.abspath(self.data_dir+"/..")}/cropped_{topfile}'
        run(f"mkdir -p {crop_dir}", shell=True) # make crop_dir
        
        for f in self.files:  
            cropped_hdu = RawData.__get_crop(self, f"{self.data_dir}/{f}", frac_hori, 
                                             frac_vert)
            new_f = f.replace(".fits","_cropped.fits")
            cropped_hdu.writeto(f"{crop_dir}/{new_f}", overwrite=True, 
                          output_verify="ignore") # write them
          
            
    def crop_images_wcs(self, ra, dec, size):
        """Crop several fits images by picking out a source based on its RA 
        and Dec.
        
        Arguments
        ---------
        ra, dec : float
            Right Ascension (RA) and Declination of interest, in degrees
        size : float
            Size of a box (in pixels) to crop around the input coordinates

            
        For a directory full of WIRCam/MegaPrime images (should **not** be 
        cubes), crops the images based on the input WCS coordinates, creating 
        a box centred on these coordinates and writing the new files to a new 
        directory. If the given box extends beyond the bounds of the image, 
        the box will be truncated at these bounds.         
        """
        topfile = re.sub(".*/", "", self.data_dir) # for a file /a/b/c, extract the "c"

        # crop_dir encodes the detector number, instrument, date
        crop_dir = f'{os.path.abspath(self.data_dir+"/..")}/cropped_{topfile}'
        run(f"mkdir -p {crop_dir}", shell=True) # make crop_dir
        
        crop_counter = 0
        for f in self.files:
            hdr = fits.getheader(f"{self.data_dir}/{f}")
            img = fits.getdata(f"{self.data_dir}/{f}")
            y_size, x_size = img.shape # total image dims in pix 
            w = wcs.WCS(hdr)
            
            # compute the bounds 
            pix_scale = hdr["PIXSCAL1"] # scale of image in arcsec per pix
            size_wcs = pix_scale*size/3600.0 # size of desired box in degrees
            pix_x1 = np.array(w.all_world2pix(ra-size_wcs/2.0, dec, 1))[0]
            pix_x2 = np.array(w.all_world2pix(ra+size_wcs/2.0, dec, 1))[0]
            pix_y1 = np.array(w.all_world2pix(ra, dec-size_wcs/2.0, 1))[1]
            pix_y2 = np.array(w.all_world2pix(ra, dec+size_wcs/2.0, 1))[1]
            x_bounds = np.array(sorted([pix_x1, pix_x2])) # sorted arrays of 
            y_bounds = np.array(sorted([pix_y1, pix_y2])) # pixel boundaries
            # truncate bounds if needed
            x_bounds[x_bounds<0] = 0 
            x_bounds[x_bounds>x_size] = x_size
            y_bounds[y_bounds<0] = 0 
            y_bounds[y_bounds>y_size] = y_size
            # convert to horizontal & vertical fractions, pass to __get_crop()
            frac_hori = x_bounds/x_size
            frac_vert = y_bounds/y_size
            
            # if the crop does not contain the bounds, skip it
            # if the crop's aspect ratio is more skew than 4:1 or 1:4, skip
            # if the crop is < 50% the width/height of the desired box, skip
            if np.all(frac_hori==0) or np.all(frac_hori==1.0) or np.all(
                    frac_vert==0.0) or np.all(frac_vert==1.0):
                continue 
            if not(0.25 < ((frac_hori[1]-frac_hori[0])/
                           (frac_vert[1]-frac_vert[0])) < 4.0):
                continue
            if not((x_bounds[1]-x_bounds[0] > size/2.0) and 
                   (y_bounds[1]-y_bounds[0] > size/2.0) ):
                continue
            
            crop_counter += 1
            cropped_hdu = RawData.__get_crop(self, f"{self.data_dir}/{f}", frac_hori, 
                                             frac_vert)
            new_f = f.replace(".fits","_cropped.fits")
            cropped_hdu.writeto(f"{crop_dir}/{new_f}", overwrite=True, 
                          output_verify="ignore") # write them
            
        print(f"{crop_counter}/{len(self.files)} images could be cropped.\n")           

        
###############################################################################
### STACKING AND STACK PREPARATION/EXTRACTION #################################

    def make_badpix_masks(self):
        """Builds a bad pixel mask for all image files, and places them in a 
        new directory which is stored in `self.bp_dir`."""

        topfile = re.sub(".*/", "", self.data_dir) # for a file /a/b/c, extract the "c"
        
        from scipy.ndimage import binary_dilation
        
        # bp_dir contains the bad pixel masks
        self.__bp_dir = f'{os.path.abspath(self.data_dir+"/..")}/badpixels_{topfile}'
        run(f"mkdir -p {self.bp_dir}", shell=True)
        
        for f in self.files:
            # build the bad pixel mask
            image_data = fits.getdata(f"{self.data_dir}/{f}")
            image_hdr = fits.getheader(f"{self.data_dir}/{f}")
            
            # mask all bad pixels, which are flagged by CFHT with 0
            bp_mask = (image_data == 0)
            
            # use binary dilation to fill holes, esp. near diffraction spikes
            bp_mask = (binary_dilation(bp_mask, iterations=2)).astype(float)
            
            bp_mask = ma.masked_where(bp_mask, image_data)
            bp_mask.fill_value = 0.0
            bp_mask_img = bp_mask.filled() # 0 at bad pix 
            bp_mask_img[bp_mask_img != 0] = 1 # 1 at good pix
                
            bp = fits.PrimaryHDU(bp_mask_img, image_hdr)
            bp.writeto(f"{self.bp_dir}/"+f.replace(f".{self.fmt}", 
                                                   "_bp_mask.fits"), 
                       overwrite=True, output_verify="ignore")
            # set the bad pixel mask
            hdu = fits.open(f"{self.data_dir}/{f}", mode="update") 
            hdu[0].header["BPM"] =  f.replace(f".{self.fmt}", "_bp_mask.fits")
            hdu.close(output_verify="ignore")
            

    def __make_stack_directory(self):
        """Makes a directory to store stacked image(s)."""

        if not(type(self.stack_dir) == type(None)):
            run(f"mkdir -p {self.stack_dir}", shell=True) # create it
        else: # if not yet defined
            print("\nPlease set a stack directory using set_stack_dir() "+
                  "before attempting to stack images. Exiting.\n")
            return 
        # get rid of calibration file if it exists:
        run(f"rm -rf {self.stack_dir}/calibration", shell=True) 
        run(f"rm -rf {self.stack_dir}/*.fits", shell=True)
        run(f"rm -rf {self.stack_dir}/*.flt", shell=True)
        
        # get rid of previous .fits, .flt and .txt files
        # need to use glob to use wildcards (*):
        textfiles = glob.glob(f"{self.stack_dir}/*.txt")
        fitsfiles = glob.glob(f"{self.stack_dir}/*.fits")
        fitsfiles = glob.glob(f"{self.stack_dir}/*.flt")
        allfiles = textfiles + fitsfiles
        for a in allfiles:
            os.remove(a)

        run(f"chmod 777 {self.stack_dir}", shell=True) # give full permissions to the dir
        
        # needed for iraf[?]
        run(f"cp -f ~/iraf/login.cl {self.stack_dir}", shell=True)
        run(f"mkdir -p {self.stack_dir}/uparm", shell=True)
        run(f"cp -f ~/iraf/uparm/* {self.stack_dir}/uparm", shell=True) 
        
        for fil in self.filters:
            np.savetxt(f"{self.stack_dir}/{fil}_list.txt", 
                       self.filters_dict[fil], 
                       fmt="%s")

        # location of bad pixel masks
        topfile = re.sub(".*/", "", self.data_dir)
        self.bp_dir = f'{os.path.abspath(self.data_dir+"/..")}/badpixels_{topfile}'

        # copy fits files
        run(f"cp -f {self.data_dir}/*.{self.fmt} {self.stack_dir}", shell=True)
        run(f"cp -f {self.bp_dir}/*.fits {self.stack_dir}", shell=True)
        run(f"chmod 777 {self.stack_dir}/*", shell=True) # give full perms
        
        return True
        

    def make_stacks(self, *filters):
        """Co-add images with PyRaf.
        
        Arguments
        ---------
        filters : str, optional
            Filters of interest (defaults to all of the filters for which 
            there is raw data)


        Uses PyRaf to coadd all of the raw data into stacks based on filter.
        Can specify a subset of these filters if you don't want to process all 
        of them at once.
        
        """
        
        # check if need to make bad pixel masks 
        #hdr = fits.getheader(self.data_dir+"/"+self.files[0])
        #try:
        #    temp = hdr["BPM"]
        #    del temp
        #except KeyError:
        #    RawData.make_badpix_masks(self) # make bad pixel masks    
            
        ret = self.__make_stack_directory() # make stack directory
        
        if not(ret): # if stack directory not successfully made
            return 
        
        script_dir = os.getcwd() # script directory
        run(f"cp stack.py {self.stack_dir}", shell=True)
        os.chdir(self.stack_dir) # move to stack directory 
            
        # command line arguments:
        cmdargs = f"{self.instrument} {self.date}" 
        
         # if no argument given
        valid_filters = " "
        if not(filters): # if no arg given
            filters = self.filters
            for fil in self.filters:
                valid_filters = f"{valid_filters}{fil} "
        else:
            for fil in filters:
                valid_filters = f"{valid_filters}{fil} "
        
        # add the valid filters, remove the space at the end:
        cmdargs += valid_filters[:-1]
        
        # run the script stack.py
        run(f"bash -c 'source activate iraf27 && python2 stack.py {cmdargs}"
            " && source deactivate'", shell=True)
        
        # update exptime of stack to equal sum of exposure times
        # "true" exptime is median of input exposure times
        for fil in filters: 
            stack = f"{fil}_stack_{self.date}.fits"
            stack_file = fits.open(stack, mode="update")
            stack_header = stack_file[0].header
            new_exptime = 0
            files = self.filters_dict[fil]
            for f in files:
                f_hdr = fits.getheader(f)
                new_exptime += f_hdr['EXPTIME']
            stack_header['EXPTIME'] = new_exptime
            stack_file.close()                                
        
        os.chdir(script_dir) # return to script directory
        self.stack_made = True
        
        
    def extract_stack(self, filt): 
        """Given a filter, create a `Stack` object which contains only stacks 
        in that filter from a base `RawData` object."""
        return Stack(self.data_dir, self.stack_dir, self.qso_grade_limit, 
                     self.fmt, self.plot_ext, filt)
 
       
###############################################################################
### STACKS ####################################################################

class Stack(RawData):
    def __init__(self, location, stack_directory, qso_grade_limit, fmt, 
                 plot_ext, filt):
        super(Stack, self).__init__(location, stack_directory, 
             qso_grade_limit, fmt, plot_ext)
                
        # only one filter is spanned by a given stack
        self.__files = self.filters_dict[filt]
        for fil in self.filters:
            if filt != fil:
                delattr(self, fil)

        # stack file specifics
        self.__stack_name = f"{filt}_stack_{self.date}.fits" # name
        self.__stack_size = len(self.files) # size
        
        # check if stack file is present before continuing and make it if not
        if not(self.stack_name in os.listdir(self.stack_dir)):
            print("\nSince the stacked image is not yet present, it will be "+
                  "produced now.")
            self.make_stacks(filt)
            
        #if not(Stack.stack_made): # if stack was not successfuly made 
        #    exit
            
        delattr(self, "filters") # don't need a list anymore 
        self.__filter = filt # just one filter 
        delattr(self,'filters_dict') # filters dict no longer needed

        # image data and header for general use
        self.__image_data = fits.getdata(f"{self.stack_dir}/{self.stack_name}")
        self.__image_header = fits.getheader(
                f"{self.stack_dir}/{self.stack_name}")
        
        # total exposure time (in seconds)
        self.__exptime = self.image_header['EXPTIME']
            
        # pixel scale of image (same scale in x and y)
        self.__pixscale = self.image_header['PIXSCAL1']
        
        # dimensions of image (pixels)
        self.__y_size, self.__x_size  = self.image_data.shape 
        
        # make a directory for astrometric/photometric calibration
        # copy the stack file accordingly 
        self.__calib_dir = f"{self.stack_dir}/calibration"
        run(f"mkdir -p {self.calib_dir}", shell=True)
        run(f"cp {self.stack_dir}/{self.stack_name} {self.calib_dir}", 
            shell=True)
        
        # initialize time for all files and the overall stack
        self.__time_init()            
            
        # for later
        self.__astrometric_calib = False # astrometry performed? 
        self.__psf_fit = False # PSF of image obtained?
        self.__photometric_calib = False # photometric zero point known?
        self.__aperture_fit = False # aperture photometry obtained?
        self.__limmag_obtained = False # limiting magnitude obtained?

        self.__bkg = None # background-only
        self.__bkg_rms = None # background RMS error
        self.__bp_mask = None # bad pixel mask
        self.__source_mask = None # bad pixel AND source mask
        self.__image_error = None # image error array (Gaussian + Poisson)
        self.__image_data_bkgsub = None # background-subtracted image
        
        
    def __time_init(self):
        """Initializes a list of observation times for each file in the stack 
        and a time for the entire stack (in Modified Julian Date (MJD))."""
        self.__times = []
        for f in self.files:
            hdr = fits.getheader(self.data_dir+"/"+f)
            
            if "WIRCam" in self.instrument:
                # check if the file comes from a divided cube 
                sliceid = hdr["SLICEID"]
                if not(sliceid): # does not come from a division
                    sliceid = "01" 
                sldate = hdr[f"SLDATE{sliceid}"] # SLDATE for correct slice
                t_isot = Time(sldate, format='isot', scale='utc')
            else:
                date_isot = hdr["DATE"] # full ISOT time
                t_isot = Time(date_isot, format='isot', scale='utc')
                
            t_MJD = t_isot.mjd # convert ISOT in UTC to MJD
            self.__times.append(t_MJD)
        
        self.__stack_time = np.mean(self.times) # stack time in MJD 


###############################################################################
### GETTERS ###################################################################
        
    def stack_size(self):
        return self.__stack_size
    
    def stack_time(self):
        return self.__stack_time
    
    def image_data(self):
        return self.__image_data
    
    def image_header(self):
        return self.__image_header
    
    def bp_dir(self):
        return self.__bp_dir
    
    def calib_dir(self):
        return self.__calib_dir
    
    def exptime(self):
        return self.__exptime
    
    def pixscale(self):
        return self.__pixscale
    
    def x_size(self):
        return self.__x_size
    
    def y_size(self):
        return self.__y_size
    
    def astrometric_calib(self):
        return self.__astrometric_calib
    
    def psf_fit(self):
        return self.__psf_fit
    
    def photometric_calib(self):
        return self.__photometric_calib
    
    def aperture_fit(self):
        return self.__aperture_fit
    
    def limmag_obtained(self):
        return self.__limmag_obtained
    
    def bkg(self):
        return self.__bkg
    
    def bp_mask(self):
        return self.__bp_mask
    
    def source_mask(self):
        return self.__source_mask
    
    def image_error(self):
        return self.__image_error
    
    def image_data_bkgsub(self):
        return self.__image_data_bkgsub
    
    def xy_data(self):
        return self.__xy_data
    
    def xy_name(self):
        return self.__xy_name
    
    def epsf_data(self):
        return self.__epsf_data
    
    def epsf_radius(self):
        return self.__epsf_radius
    
    def ref_cat(self):
        return self.__ref_cat
    
    def ref_cat_name(self):
        return self.__ref_cat_name
    
    def zp_mean(self):
        return self.__zp_mean
    
    def zp_med(self):
        return self.__zp_med
    
    def zp_std(self):
        return self.__zp_std
    
    def ra_offsets_mean(self):
        return self.__ra_offsets_mean
    
    def dec_offsets_mean(self):
        return self.__dec_offsets_mean
    
    def nmatches(self):
        return self.__nmatches
    
    def sep_mean(self):
        return self.__sep_mean
    
    def mag_diff_mean(self):
        return self.__mag_diff_mean
    
    def psf_sources(self):
        return self.__psf_sources
    
    def aperture_sources(self):
        return self.__aperture_sources
    
    def limmag_sources(self):
        return self.__limmag_sources
    

###############################################################################
### MASKS, BACKGROUND, & IMAGE ERROR ARRAY ####################################

    def mask_bp(self):
        """Make a bad pixel mask of pixels = 0 or nan. The mask is an array of 
        **bools**."""
        print("\nCreating a bad pixel mask for the stack...")
        start = timer()
        self.__bp_mask = np.logical_or(self.image_data==0, 
                                       np.isnan(self.image_data))
        end = timer()
        print(f"{(end-start):.2f} s")


    def mask_source(self, sigma=3.0):
        """Make a mask of pixels containing sources. 
        
        Arguments
        ---------
        sigma : float, optional
            Detection sigma to use in image segmentation (default 3.0)

        
        Use crude image segmentation to make a proto source mask, use the mask 
        to get the background, and then perform proper image segmentation on 
        the background-subtracted image data. The resulting segmentation image 
        is a proper source mask, to be used in other steps in aperture 
        photometry. The output source mask also flags bad pixels. 
        
        Notes
        -----
        The mask is an array of **ints**, where 1=masked, 0=non-masked. Not 
        to be confused with `bp_mask`.
        
        """   
        
        data = self.image_data # the unsubtracted image 
        
        # if bad pixel mask is not yet computed, compute it
        if not(type(self.bp_mask) == np.ndarray): 
            self.mask_bp() 

        ## set the threshold for image segmentation
        # use *crude* image segmentation to find sources above SNR=3, build a 
        # source mask, and estimate the background RMS 
        print("\nCreating a source + bad pixel mask for the stack...")
        start = timer()
        source_mask = make_source_mask(self.image_data, snr=3, npixels=5, 
                                       dilate_size=15, mask=self.bp_mask)
        rough_mask = source_mask
        
        # estimate the background standard deviation
        try:
            sigma_clip = SigmaClip(sigma=3, maxiters=5) # sigma clipping
        except TypeError: # in older versions of astropy, "maxiters" was "iters"
            sigma_clip = SigmaClip(sigma=3, iters=5)
            
        bkg_estimator = MMMBackground()
        bkg = Background2D(data, 100, filter_size=10, 
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, 
                           mask=rough_mask)
        bkg_rms = bkg.background_rms
        threshold = sigma*bkg_rms # threshold for proper image segmentation
    
        ## proper image segmentation to get a source mask    
        segm = detect_sources(data-bkg.background, threshold, npixels=5).data
        segm[segm>0] = 1 # sources have value 1, background has value 0 
        segm[self.bp_mask] = 1 # bad pixels also get a value of 1 
        segm = segm.astype(bool) # finally, convert to bool
        end = timer()
        print(f"{(end-start):.2f} s")
        
        self.__source_mask = segm

   
    def bkg_compute(self, box_size=50, filter_size=5, thresh_sigma=3.0):
        """Compute the background of the image.
        
        Arguments
        ---------
        box_size : int, optional
            Size of box for background estimation (default 50 --> 50x50 boxes)
        filter_size : int, optional
            Size of the median filter applied during background estimation 
            (default 5 --> 5x5 filter)
        thresh_sigma : float, optional
            Detection to use in image segmentation when producing a source 
            mask (default 3.0; only relevant if a source mask does not already 
            exist and needs to be computed)        
        
            
        Computes the background of the image, the error on the background (as 
        the RMS deviation), and the background-subtracted image. Assigns these 
        to the attributes `self.bkg`, `self.bkg_rms`, `self.image_data_bkgsub`.

        Notes
        -----
        A new source mask will be obtained only if `self.source_mask` does 
        not exist. 
        """

        # if source mask is not yet computed, compute it
        if not(type(self.source_mask) == np.ndarray): 
            self.mask_source(sigma=thresh_sigma) 

        print("\nComputing the background of the image...")     
        start = timer()
        # estimate background 
        bkg_est = MMMBackground()
        bkg = Background2D(self.image_data, 
                           box_size=50, filter_size=5, 
                           bkg_estimator=bkg_est, 
                           mask=self.source_mask)
        end = timer()
        print(f"{(end-start):.2f} s")
        
        # save the backgrond, background RMS error, and background-subtracted 
        # image to attributes 
        self.__bkg = bkg.background
        self.__bkg_rms = bkg.background_rms
        self.__image_data_bkgsub = self.image_data - self.bkg


    def error_array(self, box_size=50, filter_size=5, thresh_sigma=3.0):
        """Compute the error (uncertainty) on the background-only image.

        Arguments
        ---------
        box_size : int, optional
            Size of box for background estimation (default 50 --> 50x50 boxes; 
            only relevant if background not previously computed)
        filter_size : int, optional
            Size of the median filter applied during background estimation 
            (default 5 --> 5x5 filter, only relevant if background not 
            previously computed)
        thresh_sigma : float, optional
            Detection to use in image segmentation when producing a source 
            mask (default 3.0; only relevant if source mask not previously 
            computed) 
 
       
        Computes the error on the background-only image as the RMS deviation 
        of the background, and then computes the total image error including 
        the contribution of the Poisson noise from detected sources. Necessary 
        for error propagation in aperture photometry. 
 
        """

        from photutils.utils import calc_total_error
            
        # if background is not yet computed, compute it 
        if not(type(self.bkg) == np.ndarray): 
            self.bkg_compute(box_size=box_size, filter_size=filter_size,
                             thresh_sigma=thresh_sigma) 
            
        print("\nComputing the error array...")
        start = timer()
        if "WIRCam" in self.instrument:
            eff_gain = 3.8 # effective gain (e-/ADU) for WIRCam
        else: 
            eff_gain = self.image_header["GAIN"] # effective gain for MegaPrime
        
        # compute sum of Poisson error and background error  
        # currently, this seems to overestimate unless the input data is 
        # previously background-subtracted
        err = calc_total_error(self.image_data - self.bkg, 
                               self.bkg_rms, eff_gain)
        end = timer()
        print(f"{(end-start):.2f} s")
        
        self.__image_error = err
    
    
###############################################################################
### MAKING IMAGES (PLOTTING) ##################################################
        
    def make_image(self, bkgsub=False, border=False, sources=False, 
                   ra=None, dec=None, scale=None, title=None, output=None):
        """
        Input: 
            - whether plot the background-subtracted data (optional; default 
              False)
            - whether to show the region where sources are considered valid for 
              photometry (optional; default False)
            - whether to show detected sources (optional; default False), 
            - RA, Dec for a crosshair (optional; default None)
            - scale to apply to the plot (optional; default None=linear; 
              options are "linear", "log", "asinh") 
            - title for the plot (optional; default None)
            - name for the image file (optional; default set below)
            
        Produces and saves an an image of the coadd.
        
        Output: None
        """
        # image data
        if bkgsub and not(self.bkg == None):
            image_data = self.image_data_bkgsub # background-sub'd image data
        elif bkgsub and (self.bkg == None):
            print("To obtain a background-subtracted, smoothed image, use "+
                  "the bkg_compute() function first.")
        else:
            image_data = self.image_data
                 
        w = wcs.WCS(self.image_header)
        
        if not(output): # if none given, defaults to the following 
            output = f"{self.filter}_{self.instrument}_{self.date}"
            output = f"{output}.{self.plot_ext}"
            
        # set figure dimensions
        if "WIRCam" in self.instrument:
            plt.figure(figsize=(10,9))
        else:
            plt.figure(figsize=(12,14))

        # plot a circle/rectangle bounding the border 
        if border:  
            import matplotlib.patches as ptc
            if "WIRCam" in self.instrument:
                circ = ptc.Circle((self.x_size/2.0,self.y_size/2.0), 
                                  radius=self.x_size/2.0, 
                                  facecolor="None", lw=2.0,
                                  edgecolor="#95d0fc", linestyle=":")
                ax = plt.subplot(projection=w)
                ax.add_patch(circ)
                ax.coords["ra"].set_ticklabel(size=15)
                ax.coords["dec"].set_ticklabel(size=15)
            else:
                rect = ptc.Rectangle((0.05*self.x_size, 0.05*self.y_size), 
                                     width=0.9*self.x_size,
                                     height=0.9*self.y_size,
                                     facecolor="None", lw=2.0,
                                     edgecolor="#95d0fc", linestyle=":")
                ax = plt.subplot(projection=w)
                ax.add_patch(rect)
                ax.coords["ra"].set_ticklabel(size=15)
                ax.coords["dec"].set_ticklabel(size=15)
                
        else:
            ax = plt.subplot(projection=w) # show WCS
            ax.coords["ra"].set_ticklabel(size=15)
            ax.coords["dec"].set_ticklabel(size=15)
            
        if sources and self.astrometric_calib: # if we want to see sources
            sources_data = self.xy_data
            sources = Table()
            sources['x_mean'] = sources_data['X']
            sources['y_mean'] = sources_data['Y']
            # mask out edge sources:
            # a bounding circle for WIRCam, rectangle for MegaPrime
            if "WIRCam" in self.instrument:
                dist_to_center = np.sqrt(
                        (sources['x_mean']-self.x_size/2.0)**2 + 
                        (sources['y_mean']-self.y_size/2.0)**2)
                mask = dist_to_center <= self.x_size/2.0
                sources = sources[mask]
            else: 
                x_lims = [int(0.05*self.x_size), int(0.95*self.x_size)] 
                y_lims = [int(0.05*self.y_size), int(0.95*self.y_size)]
                mask = (sources['x_mean']>x_lims[0]) & (
                        sources['x_mean']<x_lims[1]) & (
                        sources['y_mean']>y_lims[0]) & (
                        sources['y_mean']<y_lims[1])
                sources = sources[mask] 
            plt.plot(sources['x_mean'],sources['y_mean'], marker='.', 
                     markerfacecolor="None", markeredgecolor="#95d0fc",
                     linestyle="") # sources as unfilled light blue circles
            
        elif sources and not(self.astrometric_calib):
            print("\nSources cannot be shown because astrometric calibration"+ 
                  " has not yet been performed.")
    
        if ra and dec: # if we want to mark a specific location
            rp, dp = w.all_world2pix(ra, dec,1)
            # create a marker which looks like a crosshair: 
            plt.plot(rp-20, dp, color="red", marker=0, markersize=20, lw=0.8)
            plt.plot(rp+20, dp, color="red", marker=1, markersize=20, lw=0.8)
            plt.plot(rp, dp+20, color="red", marker=2, markersize=20, lw=0.8)
            plt.plot(rp, dp-20, color="red", marker=3, markersize=20, lw=0.8)
        
        if not scale: # if no scale to apply 
            plt.imshow(image_data, cmap='magma', aspect=1, 
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label="ADU", fontsize=16)
            
        elif scale == "log": # if we want to apply a log scale 
            image_data = np.log10(image_data)
            lognorm = simple_norm(image_data, "log", percent=99.0)
            plt.imshow(image_data, cmap='magma', aspect=1, norm=lognorm,
                       interpolation='nearest', origin='lower')
            cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08) 
            cb.set_label(label=r"$\log(ADU)$", fontsize=16)
            
        elif scale == "asinh": # asinh scale
            image_data = np.arcsinh(image_data)
            asinhnorm = simple_norm(image_data, "asinh")
            plt.imshow(image_data, cmap="magma", aspect=1, norm=asinhnorm,
                       interpolation="nearest", origin="lower")
            cb = plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            cb.set_label(label="a"+r"$\sinh{(ADU)}$", fontsize=16)
        
        if title:
            plt.title(title, fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.savefig(output, bbox_inches="tight")
        plt.close()         

###############################################################################
### ASTROMETRY ################################################################

    def astrometry(self, verbose=0):
        """
        Input: level of verbosity
        
        Performs source extraction using astrometry.net, solves the field, and 
        outputs a list of x, y coordinates for sources. 
        
        Output: 
        """
        start = timer() # timing the function
        
        script_dir = os.getcwd() # script directory
        os.chdir(self.calib_dir) # move to calibration directory
        
        # don't check WCS headers, don't plot, and input a fits image:
        solve_options = "--no-verify --overwrite --no-plot --fits-image" 
        # build a new fits file: 
        solve_options = f"{solve_options} --new-fits "
        solve_options += self.stack_name.replace(".fits", "_updated.fits")
            
        # options to speed up astrometry: pixscale and rough RA, Dec
        pixmin = self.pixscale-0.005
        pixmax = self.pixscale+0.005
        solve_options = f"{solve_options} --scale-low {pixmin} --scale-high"
        solve_options = f"{solve_options} {pixmax} --scale-units 'app'"
        
        cent = [i//2 for i in self.image_data.shape]
        centy, centx = cent
        w = wcs.WCS(self.image_header)
        ra, dec = w.all_pix2world(centx, centy, 1) 
        rad = 0.5 # look in a radius of 0.5 deg
        solve_options = f"{solve_options} --ra {ra} --dec {dec} --radius {rad}"
        
        # stop when this file is produced:
        solve_options = f"{solve_options} --cancel "
        solve_options += self.stack_name.replace(".fits", "_updated.fits")

        # don't bother producing these files 
        solve_options = f'{solve_options} --match "none" --solved "none"'
        solve_options = f'{solve_options} --rdls "none" --corr "none"'
        solve_options = f'{solve_options} --wcs "none"'
        
        # only write a temporary *augmented* xy (deleted after solving)
        solve_options = f'{solve_options} --temp-axy' 
        
        # keep the normal xy list of sources and write to this file
        xyname = self.stack_name.replace(".fits", ".xy.fits")      
        solve_options = f'{solve_options} --keep-xylist {xyname}'

        # set level of verbosity (-v = verbose, -v -v = very verbose)
        for i in range(min(verbose, 2)):
            solve_options = f"{solve_options} -v" 
        
        # solve the field: 
        run(f"solve-field {solve_options} {self.stack_name}", shell=True)
        # update the name of the stack file:
        self.stack_name = self.stack_name.replace(".fits", "_updated.fits")
        run("find . -type f -not -name '*updat*' -print0 | xargs -0 rm --",
            shell=True) # remove all files not in format *updat*
        
        # check if any files are present and exit if not 
        updats = os.listdir()
        if len(updats) == 0:
            os.chdir(script_dir)
            print("The WCS solution could not be obtained. This likely means "+
                  "that data is of poor quality (e.g. many cosmic rays or "+
                  "other artifacts) or the images making up the stack "+
                  "require astrometric alignment BEFORE stacking. Exiting.")
            return

        # print confirmation 
        print("The WCS solution has been updated for the stack image.")
        print(f"Stack name is now {self.stack_name}")
        
        # store the xy list output in attributes 
        # source data for general use
        self.__xy_name = xyname
        self.__xy_data = fits.getdata(self.xy_name) 
         
        # store updated WCS solution in header
        self.image_header = fits.getheader(self.stack_name)
        
        os.chdir(script_dir) # return to script directory 
        self.astrometric_calib = True 
        
        end = timer()
        print(f"Time for astrometric calibration: {(end-start):.2f} s\n")

###############################################################################
### PSF PHOTOMETRY ############################################################

    def __ePSF_FWHM(self, epsf_data, verbose=True):
        """
        Input: 
            - ePSF data
            - be verbose (optional; default True)
        
        Output: the FWHM of the input ePSF
        """
        
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
    
        if verbose:
            print(f"ePSF FWHM: {(epsf_radius*2.0/10.0):.1f} pix")
        return epsf_radius*2.0/10.0


    def __fit_PSF(self, nstars=40, thresh_sigma=5.0, 
                  pixelmin=20, elongation_lim=1.4, area_max=500, 
                  cutout=35, 
                  source_lim=None,
                  write=False, output=None,
                  plot_ePSF=True, ePSF_name=None, 
                  plot_residuals=False, resid_name=None, 
                  verbose=True):
        """        
        Input: 
            - maximum number of stars to use (optional; default 40; set to None
              to impose no limit)
            - sigma threshold for source detection with image segmentation 
              (optional; default 5.0)
            - *minimum* number of isophotal pixels (optional; default 20)
            - *maximum* allowed elongation for sources found by image 
              segmentation (optional; default 1.4)
            - *maximum* allowed area for sources found by image segmentation 
              (optional; default 500 pix**2)
            - cutout size around each star in pix (optional; default 35 pix; 
              must be ODD, rounded down if even)
            - limit on number of sources to fit with ePSF (optional; default 
              None) 
            - whether to write the built ePSF (optional; default False)
            - name for output ePSF .fits file (optional; default set below)
            - whether to plot the derived ePSF (optional; default True)
            - name for output ePSF plot (optional; default set below)
            - whether to plot the residuals of the iterative PSF fitting 
              (optional; default False)
            - name for output residuals plot (optional; default set below)
            - be verbose (optional; default True)
    
        Uses image segmentation to obtain a list of sources in the smoothed, 
        background-subtracted image (provided by astrometry.net) with their 
        x, y coordinates. Uses EPSFBuilder to empirically obtain the ePSF of 
        these stars. Uses astrometry.net to find all sources in the image, and 
        fits them with the empirically obtained ePSF.
    
        The ePSF obtained here should NOT be used in convolutions. Instead, it 
        can serve as a tool for estimating the seeing of an image. Builds a 
        table containing the instrumental magnitudes and corresponding 
        uncertainties to be used in obtaining the zero point for PSF 
        calibration.
        
        Output: None
        """
        
        if not(self.astrometric_calib):
            print("\nPSF photometry cannot be obtained because astrometric "+
                  "calibration has not yet been performed. Exiting.")
            return

        from photutils import EPSFBuilder
        from photutils.psf import (extract_stars, BasicPSFPhotometry, DAOGroup)      
        from astropy.nddata import NDData
        from astropy.modeling.fitting import LevMarLSQFitter
            
        image_data = self.image_data_bkgsub # the bkg-subtracted image data 
        sources_data = self.xy_data # sources
        
        ### SOURCE DETECTION   
        ### use image segmentation to find sources with an area > pixelmin 
        ### pix**2 which are above the threshold sigma*std
        image_data = np.ma.masked_where(self.bp_mask, image_data) # mask bp       
        std = np.std(np.ma.masked_where(self.source_mask, 
                                        image_data))
        
        ## use the segmentation image to get the source properties 
        # use <bp_mask>, which does not mask sources
        segm = detect_sources(image_data, thresh_sigma*std, npixels=pixelmin,
                              mask=self.bp_mask) 
        cat = source_properties(image_data, segm, mask=self.bp_mask)
    
        ## get the catalog and coordinates for sources
        try:
            tbl = cat.to_table()
        except ValueError:
            print("SourceCatalog contains no sources. Exiting.")
            return
        
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
        w = wcs.WCS(self.image_header)
        sources["ra"], sources["dec"] = w.all_pix2world(sources["x"],
                                                        sources["y"], 1)
 
        ## mask out edge sources: 
        # a bounding circle for WIRCam, rectangle for MegaPrime
        if "WIRCam" in self.instrument:
            rad_limit = self.x_size/2.0
            dist_to_center = np.sqrt((sources['x']-self.x_size/2.0)**2 + 
                                     (sources['y']-self.y_size/2.0)**2)
            dmask = dist_to_center <= rad_limit
            sources = sources[dmask]
        else: 
            x_lims = [int(0.05*self.x_size), int(0.95*self.x_size)] 
            y_lims = [int(0.05*self.y_size), int(0.95*self.y_size)]
            dmask = (sources['x']>x_lims[0]) & (sources['x']<x_lims[1]) & (
                     sources['y']>y_lims[0]) & (sources['y']<y_lims[1])
            sources = sources[dmask]
            
        ## empirically obtain the effective Point Spread Function (ePSF)  
        nddata = NDData(image_data) # NDData object
        if cutout%2 == 0: # if cutout even, subtract 1
            cutout -= 1
        stars = extract_stars(nddata, sources, size=cutout) # extract stars
    
        ## build the ePSF
        nstars_epsf = len(stars.all_stars) # no. of stars used in ePSF building
        
        if nstars_epsf == 0:
            print("\nNo valid sources were found to build the ePSF with the "+
                  "given conditions. Exiting.")
            return
        
        if verbose:
            print(f"\n{nstars_epsf} stars used in building the ePSF")
            
        start = timer()
        epsf_builder = EPSFBuilder(oversampling=1, maxiters=7, # build it
                                   progress_bar=False)
        epsf, fitted_stars = epsf_builder(stars)
        self.__epsf_data = epsf.data # store ePSF data for later 
        
        end = timer() # timing 
        print(f"Time required for ePSF building: {(end-start):.2f} s\n")

        if write: # write, if desired
            epsf_hdu = fits.PrimaryHDU(data=self.epsf_data)
            if not(output):
                output = self.stack_name.replace("_updated.fits", 
                                                 "_ePSF.fits")               
            epsf_hdu.writeto(output, overwrite=True, output_verify="ignore")

        psf_model = epsf # set the model
        psf_model.x_0.fixed = True # fix centroids (known beforehand) 
        psf_model.y_0.fixed = True
        start = timer() # timing ePSF building time

        # get ePSF FHWM, store for later 
        self.__epsf_radius = self.__ePSF_FWHM(epsf.data, verbose)
        
        ### USE ASTROMETRY.NET'S SOURCES FOR FITTING
        astrom_sources = Table() # build a table 
        astrom_sources['x_mean'] = sources_data['X'] # for BasicPSFPhotometry
        astrom_sources['y_mean'] = sources_data['Y']
        astrom_sources['flux'] = sources_data['FLUX']
    
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
        
        # if we have a limit on the number of sources
        if source_lim:
            try: 
                import random # pick a given no. of random sources 
                source_rows = random.choices(astrom_sources, k=source_lim)
                astrom_sources = Table(names=['x_mean', 'y_mean', 'flux'], 
                                       rows=source_rows)
                pos = Table(names=['x_0', 'y_0','flux_0'], 
                            data=[astrom_sources['x_mean'], 
                                  astrom_sources['y_mean'], 
                                  astrom_sources['flux']])
                
                
            except IndexError:
                print("The input source limit exceeds the number of sources"+
                      " detected by astrometry, so no limit is imposed.\n")
        
        photometry = BasicPSFPhotometry(group_maker=daogroup,
                                        bkg_estimator=None, # already bkg-sub'd 
                                        psf_model=psf_model,
                                        fitter=fitter_tool,
                                        fitshape=(11,11))
        
        result_tab = photometry(image=image_data, init_guesses=pos) # results
        residual_image = photometry.get_residual_image() # residuals of PSF fit
        residual_image = np.ma.masked_where(self.bp_mask, residual_image)
        residual_image.fill_value = 0 # set to zero
        residual_image = residual_image.filled()
        
        end = timer() # timing 
        print("Time to fit ePSF to all sources: {(end-start):.2f} s\n")
        
        # include WCS coordinates
        pos["ra"], pos["dec"] = w.all_pix2world(pos["x_0"], pos["y_0"], 1)
        result_tab.add_column(pos['ra'])
        result_tab.add_column(pos['dec'])
            
        # mask out negative flux_fit values in the results 
        mask_flux = (result_tab['flux_fit'] >= 0.0)
        psf_sources = result_tab[mask_flux]
        
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
            plt.imshow(epsf.data, origin='lower', aspect=1, cmap='magma',
                       interpolation="nearest")
            plt.xlabel("Pixels", fontsize=16)
            plt.ylabel("Pixels", fontsize=16)
            plt.title('Effective Point-Spread Function (1 pixel = '+
                      f'{self.pixscale:.3f}"', 
                      fontsize=16)
            plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            plt.rc("xtick",labelsize=16) # not working?
            plt.rc("ytick",labelsize=16)
            if not(ePSF_name):
                ePSF_name = f"{self.filter}_{self.instrument}_{self.date}"
                ePSF_name = f"{ePSF_name}_ePSF.{self.plot_ext}"
            plt.savefig(ePSF_name, bbox_inches="tight")
            plt.close()
        
        if plot_residuals: # if we wish to see a plot of the residuals
            if "WIRCam" in self.instrument:
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
            if not(resid_name):
                resid_name = f"{self.filter}_{self.instrument}_{self.date}"
                resid_name = f"{resid_name}_PSF_resid.{self.plot_ext}"
            plt.savefig(resid_name, bbox_inches="tight")
            plt.close()
        
        # save psf_sources as an attribute
        self.__psf_sources = psf_sources        
        # update bool
        self.__psf_fit = True      
        
        
    def __zero_point(self, sep_max=2.0,
                     plot_corr=True, corr_name=None, 
                     plot_source_offsets=True, source_offs_name=None,
                     plot_field_offsets=False, field_offs_name=None,
                     gaussian_blur_sigma=30.0, cat_num=None):
        """
        Input:
            - maximum allowed separation when cross-matching sources (optional;
              default 2.0 pix ~ 0.6" for WIRCam and ~ 0.37" pix for MegaPrime)
            - whether or not to plot the correlation with linear fit (optional; 
              default True)
            - name for output correlation plot (optional; default set below)
            - whether to plot the offsets in RA and Dec of each catalog-matched 
              source (optional; default True) 
            - name for output plot of source offsets (optional; default set 
              below)
            - whether to show the overall offsets as an image with a 
              Gaussian blur to visualize large-scale structure (optional; 
              default False)
            - name for output plot of field offsets (optional; default set
              below)
            - sigma to apply to the Gaussian filter (optional; default 30.0)
            - a Vizier catalog number to choose which catalog to cross-match 
              (optional; defaults are PanStarrs 1, SDSS DR12, and 2MASS for 
              relevant filters)
        
        Uses astroquery and Vizier to query an online catalog for sources 
        which match those detected by astrometry. Computes the offset between
        the apparent and instrumental magnitudes of the queried sources for 
        photometric calibration. Computes the mean, median and standard 
        deviation.
        
        Output: None
        """
        
        from astroquery.vizier import Vizier
        
        # determine the catalog to compare to for photometry
        if cat_num: # if a Vizier catalog number is given 
            self.__ref_cat = cat_num
            self.__ref_cat_name = cat_num
        else:  
            if self.filter in ['g','r','i','z','Y']:
                zp_filter = (self.filter).lower() # lowercase needed for PS1
                self.__ref_cat = "II/349/ps1" # PanStarrs 1
                self.__ref_cat_name = "PS1" 
            elif self.filter == 'u':
                zp_filter = 'u' # closest option right now 
                self.__ref_cat = "V/147" 
                self.__ref_cat_name = "SDSS DR12"
            else: 
                zp_filter = self.filter[0] # Ks must be K for 2MASS 
                self.__ref_cat = "II/246/out" # 2MASS
                self.__ref_cat_name = "2MASS"
            
        w = wcs.WCS(self.image_header) # WCS object and coords of centre           
        wcs_centre = np.array(w.all_pix2world(
                self.x_size/2.0, self.y_size/2.0, 1)) 
    
        ra_centre = wcs_centre[0]
        dec_centre = wcs_centre[1]
        radius = self.pixscale*np.max([self.x_size,self.y_size])/60.0 #arcmins
        minmag = 13.0 # magnitude minimum
        maxmag = 20.0 # magnitude maximum
        max_emag = 0.4 # maximum allowed error 
        nd = 5 # minimum no. of detections for a source (across all filters)
         
        # actual querying (internet connection needed) 
        print(f"\nQuerying Vizier {self.ref_cat} ({self.ref_cat_name}) "+
              f"around RA {ra_centre:.4f}, Dec {dec_centre:.4f} "+
              f"with a radius of {radius:.4f} arcmin")
        
        v = Vizier(columns=["*"], 
                   column_filters={f"{zp_filter}mag":f"{minmag}..{maxmag}",
                                   f"e_{zp_filter}mag":f"<{max_emag}",
                                   "Nd":f">{nd}"}, 
                   row_limit=-1) # no row limit 
        Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                            unit=(u.deg, u.deg)), radius=f"{radius}m", 
                            catalog=self.ref_cat, cache=False)
    
        if len(Q) == 0: # if no matches
            print(f"\nNo matches were found in the {self.ref_cat_name} "+
                  "catalog. The requested region may be in an unobserved "+
                  "region of this catalog. Exiting.")
            return 
                    
        # pixel coords of found sources
        cat_coords = w.all_world2pix(Q[0]['RAJ2000'], Q[0]['DEJ2000'], 1)
        
        # mask out edge sources
        # a bounding circle for WIRCam, rectangle for MegaPrime
        if "WIRCam" in self.instrument:
            rad_limit = self.x_size/2.0
            dist_to_center = np.sqrt((cat_coords[0]-self.x_size/2.0)**2 + 
                                     (cat_coords[1]-self.y_size/2.0)**2)
            mask = dist_to_center <= rad_limit
            good_cat_sources = Q[0][mask]
        else:
            x_lims = [int(0.05*self.x_size), int(0.95*self.x_size)] 
            y_lims = [int(0.05*self.y_size), int(0.95*self.y_size)]
            mask = (cat_coords[0] > x_lims[0]) & (
                    cat_coords[0] < x_lims[1]) & (
                    cat_coords[1] > y_lims[0]) & (
                    cat_coords[1] < y_lims[1])
            good_cat_sources = Q[0][mask] 
        
        # cross-matching coords of sources found by astrometry
        source_coords = SkyCoord(ra=self.psf_sources['ra'], 
                                 dec=self.psf_sources['dec'], 
                                 frame='icrs', unit='degree')
        # and coords of valid sources in the queried catalog 
        cat_source_coords = SkyCoord(ra=good_cat_sources['RAJ2000'], 
                                     dec=good_cat_sources['DEJ2000'], 
                                     frame='icrs', unit='degree')
        
        # indices of matching sources (within 2*(pixel scale) of each other) 
        idx_image, idx_cat, d2d, d3d = cat_source_coords.search_around_sky(
                source_coords, sep_max*self.pixscale*u.arcsec)
        
        if len(idx_image) <= 3:
            raise TooFewMatchesError(f"\nFound {len(idx_image)} matches "+
                                     f"between image and {self.ref_cat_name} "+
                                     "and >3 matches are required. Exiting.")
            return
        
        self.__nmatches = len(idx_image) # store number of matches 
        self.__sep_mean = np.mean(d2d.value*3600.0) # store mean separation in "
        print(f'\nFound {self.nmatches:d} sources in {self.ref_cat_name} '+
              f'within {sep_max} pix of sources detected by astrometry, with '+
              f'average separation {self.sep_mean:.3f}" ')
        
        # get coords for sources which were matched
        source_matches = source_coords[idx_image]
        cat_matches = cat_source_coords[idx_cat]
        source_matches_ra = [i.ra.value for i in source_matches]
        cat_matches_ra = [i.ra.value for i in cat_matches]
        source_matches_dec = [i.dec.value for i in source_matches]
        cat_matches_dec = [i.dec.value for i in cat_matches]
        # compute offsets (in arcsec)
        ra_offsets = np.subtract(source_matches_ra, cat_matches_ra)*3600.0
        dec_offsets = np.subtract(source_matches_dec, cat_matches_dec)*3600.0
        self.__ra_offsets_mean = np.mean(ra_offsets)
        self.__dec_offsets_mean = np.mean(dec_offsets)

        # plot the correlation
        if plot_corr:
            # fit a straight line to the correlation
            from scipy.optimize import curve_fit
            def f(x, m, b):
                return b + m*x
            
            xdata = good_cat_sources[f"{zp_filter}mag"][idx_cat] # catalog
            xdata = [float(x) for x in xdata]
            ydata = self.psf_sources['mag_fit'][idx_image] # instrumental 
            ydata = [float(y) for y in ydata]
            popt, pcov = curve_fit(f, xdata, ydata) # obtain fit
            m, b = popt # fit parameters
            perr = np.sqrt(np.diag(pcov))
            m_err, b_err = perr # errors on parameters 
            fitdata = [m*x + b for x in xdata] # plug fit into data 
            
            # plot correlation
            fig, ax = plt.subplots(figsize=(10,10))
            ax.errorbar(good_cat_sources[f"{zp_filter}mag"][idx_cat], 
                     self.psf_sources['mag_fit'][idx_image], 
                     self.psf_sources['mag_unc'][idx_image],
                     marker='.', mec="#fc5a50", mfc="#fc5a50", 
                     ls="", color='k', 
                     markersize=12, label=f"Data [{self.filter}]", zorder=1) 
            corr_label = r"$y = mx + b $"+"\n"
            corr_label += r"$ m=$%.3f$\pm$%.3f, $b=$%.3f$\pm$%.3f"%(
                          m, m_err, b, b_err)
            ax.plot(xdata, fitdata, color="blue", label=corr_label, 
                    zorder=2) # the linear fit 
            ax.set_xlabel(f"Catalog magnitude [{self.ref_cat_name}]", 
                          fontsize=15)
            ax.set_ylabel('Instrumental PSF-fit magnitude', fontsize=15)
            ax.set_title("PSF Photometry", fontsize=15)
            ax.legend(loc="upper left", fontsize=15, framealpha=0.5)
            if not(corr_name):
                corr_name = f"{self.filter}_{self.instrument}_{self.date}"
                corr_name = f"{corr_name}_PSF_photometry.{self.plot_ext}"
            plt.savefig(corr_name, bbox_inches="tight")
            plt.close()
        
        # plot the RA, Dec offset for each matched source 
        if plot_source_offsets:             
            # plot
            plt.figure(figsize=(10,10))
            plt.plot(ra_offsets, dec_offsets, marker=".", linestyle="", 
                    color="#ffa62b")
            plt.xlabel('RA (J2000) offset ["]', fontsize=15)
            plt.ylabel('Dec (J2000) offset ["]', fontsize=15)
            plt.title(f"Source offsets from {self.ref_cat_name} catalog",
                      fontsize=15)
            plt.axhline(0, color="k", linestyle="--", alpha=0.3) # (0,0)
            plt.axvline(0, color="k", linestyle="--", alpha=0.3)
            plt.plot(self.ra_offsets_mean, self.dec_offsets_mean, marker="X", 
                     color="blue", label = "Mean", linestyle="") # mean
            plt.legend(fontsize=15)
            plt.rc("xtick",labelsize=14)
            plt.rc("ytick",labelsize=14)
            if not (source_offs_name):
                source_offs_name = f"{self.filter}_{self.instrument}"
                source_offs_name = f"{source_offs_name}_{self.date}"
                source_offs_name += "_source_offsets_astrometry"
                source_offs_name = f"{source_offs_name}.{self.plot_ext}"
            plt.savefig(source_offs_name, bbox_inches="tight")
            plt.close()
        
        # plot the overall offset across the field 
        if plot_field_offsets:
            from scipy.ndimage import gaussian_filter
            # add offsets to a 2d array
            offsets_image = np.zeros(self.image_data.shape)
            for i in range(len(d2d)): 
                x = self.psf_sources[idx_image][i]["x_0"]
                y = self.psf_sources[idx_image][i]["y_0"]
                intx, inty = int(x), int(y)
                offsets_image[inty, intx] = d2d[i].value*3600.0    
            # apply a gaussian blur to visualize large-scale structure
            blur_sigma = gaussian_blur_sigma
            offsets_image_gaussian = gaussian_filter(offsets_image, blur_sigma)
            offsets_image_gaussian *= np.max(offsets_image)
            offsets_image_gaussian *= np.max(offsets_image_gaussian)
            
            # plot
            if "WIRCam" in self.instrument:
                plt.figure(figsize=(10,9))
            else:
                plt.figure(figsize=(9,13))                
            ax = plt.subplot(projection=w)
            plt.imshow(offsets_image_gaussian, cmap="magma", 
                       interpolation="nearest", origin="lower")
            # textbox indicating the gaussian blur and mean separation
            textstr = r"Gaussian blur: $\sigma = %.1f$"%blur_sigma+"\n"
            textstr += r'$\overline{offset} = %.3f$"'%self.sep_mean
            box = dict(boxstyle="square", facecolor="white", alpha=0.8)
            if "WIRCam" in self.instrument:
                plt.text(0.6, 0.91, transform=ax.transAxes, s=textstr, 
                         bbox=box, fontsize=15)
            else:
                plt.text(0.44, 0.935, transform=ax.transAxes, s=textstr, 
                         bbox=box, fontsize=15)    
            plt.xlabel("RA (J2000)", fontsize=16)
            plt.ylabel("Dec (J2000)", fontsize=16)
            plt.title(f"Field offsets from {self.ref_cat_name} catalog",
                      fontsize=15)
            ax.coords["ra"].set_ticklabel(size=15)
            ax.coords["dec"].set_ticklabel(size=15)
            if not (field_offs_name):
                field_offs_name = f"{self.filter}_{self.instrument}"
                field_offs_name = f"{field_offs_name}_{self.date}"
                field_offs_name += "_field_offsets_astrometry"
                field_offs_name = f"{field_offs_name}.{self.plot_ext}"
            plt.savefig(field_offs_name, bbox_inches="tight")
            plt.close()
        
        # compute magnitude differences and zero point mean, median and error
        mag_offsets = ma.array(good_cat_sources[f"{zp_filter}mag"][idx_cat] - 
                      self.psf_sources['mag_fit'][idx_image])

        zp_mean, zp_med, zp_std = sigma_clipped_stats(mag_offsets)
        
        # update attributes 
        self.__zp_mean, self.__zp_med, self.__zp_std = zp_mean, zp_med, zp_std
        
        # add these to the header of the image 
        scrip_dir = os.getcwd()
        os.chdir(self.calib_dir)
        f = fits.open(self.stack_name, mode="update")
        f[0].header["ZP_MEAN"] = zp_mean
        f[0].header["ZP_MED"] = zp_med
        f[0].header["ZP_STD"] = zp_std
        f.close()
        os.chdir(scrip_dir)
        
        # add a mag_calib and mag_calib_unc column to psf_sources
        mag_calib = self.psf_sources['mag_fit'] + zp_mean
        mag_calib.name = 'mag_calib'
        # propagate errors 
        mag_calib_unc = np.sqrt(self.psf_sources['mag_unc']**2 + zp_std**2)
        mag_calib_unc.name = 'mag_calib_unc'
        self.__psf_sources['mag_calib'] = mag_calib
        self.__psf_sources['mag_calib_unc'] = mag_calib_unc
        
        # add flag indicating if source is in a catalog and which catalog 
        in_cat = []
        for i in range(len(self.psf_sources)):
            if i in idx_image:
                in_cat.append(True)
            else:
                in_cat.append(False)
        in_cat_col = Column(data=in_cat, name="in_catalog")
        self.__psf_sources[f"in {self.ref_cat_name}"] = in_cat_col
        
        # add new columns 
        nstars = len(self.psf_sources)
        col_filt = Column([self.filter for i in range(nstars)], "filter",
                           dtype = np.dtype("U2"))
        col_mjd = Column([self.stack_time for i in range(nstars)], "MJD")
        self.__psf_sources["filter"] = col_filt
        self.__psf_sources["MJD"] = col_mjd
        
        # compute magnitude differences between catalog and calibration 
        # diagnostic for quality of zero point determination 
        sources_mags = self.psf_sources[idx_image]["mag_calib"]
        cat_mags = good_cat_sources[idx_cat][zp_filter+"mag"]
        mag_diff_mean = np.mean(sources_mags - cat_mags)
        print("\nMean difference between calibrated magnitudes and "+
              f"{self.ref_cat_name} magnitudes = {mag_diff_mean}")
        self.__mag_diff_mean = mag_diff_mean
        
        # update bool
        self.__photometric_calib = True
        
        
    def PSF_photometry(self, nstars=40, thresh_sigma=5.0, pixelmin=20, 
                       elongation_lim=1.4, area_max=500, cutout=35, 
                       source_lim=None, gaussian_blur_sigma=30.0, cat_num=None,
                       sep_max=2.0, verbose=True, box_size=50, filter_size=5, 
                       write_ePSF=True, ePSF_data_name=None,
                       plot_ePSF=True, ePSF_name=None,
                       plot_resid=False, resid_name=None,
                       plot_corr=True, corr_name=None,
                       plot_source_offsets=True, source_offs_name=None,
                       plot_field_offsets=False, field_offs_name=None):
        """        
        Input: 
            general:
            - maximum number of stars to use in ePSF building (optional; 
              default 40; set to None to impose no limit)
            - sigma threshold for source detection with image segmentation 
              (optional; default 5.0)
            - *minimum* number of isophotal pixels (optional; default 20)
            - *maximum* allowed elongation for sources found by image 
              segmentation (optional; default 1.4)
            - *maximum* allowed area for sources found by image segmentation 
              (optional; default 500 pix**2)
            - cutout size around each star in pix (optional; default 35 pix; 
              must be ODD, rounded down if even)
            - limit on number of sources to fit with ePSF (optional; default 
              None which imposes no limit)
            - sigma to use for the Gaussian blur, if relevant (optional; 
              default 30.0)
            - Vizier catalog number to choose which catalog to cross-match 
              (optional; defaults are PanStarrs 1, SDSS DR12, and 2MASS for 
              relevant filters)
            - maximum allowed separation when cross-matching sources (optional;
              default 2.0 pix ~ 0.6" for WIRCam and ~ 0.37" pix for MegaPrime)
            - be verbose (optional; default True)
            
            only relevant if background not computed beforehand:
            - box size for the background computation (optional; default 50x50)
            - size for the median filter applied during background computation
              (optional; default 5x5)

            - whether to write the built ePSF (optional; default False)
            - name for output ePSF .fits file (optional; default set below)
            - whether to plot the derived ePSF (optional; default True)
            - name for output ePSF plot (optional; default set below)
            - whether to plot the residuals of the iterative PSF fitting 
              (optional; default False)
            - name for output residuals plot (optional; default set below)
            - whether to plot instrumental magnitude versus catalogue magnitude 
              correlation when obtaining the zero point (optional; default 
              True)
            - name for output correlation (optional; default set below)
            - whether to plot the offsets between the image WCS and catalogue 
              WCS (optional; default True)
            - name for output source offsets plot (optional; default set below)
            - whether to plot the offsets across the field with a Gaussian blur
              to visualize large-scale structure in the offsets if any is 
              present (optional; default False)
            - name for the output field offsets plot (optional; default set 
              below)         
        
        Using image segmentation, finds as many sources as possible in the 
        image with an elongation below some elongation limit. Uses these 
        sources to build an empirical effective PSF (ePSF). Using a list of 
        sources found by astrometry.net, fits the ePSF to all of those sources. 
        Computes the instrumental magnitude of all of these sources. Queries 
        the correct online catalogue for the given filter to crossmatch sources
        in the image with those in the catalogue (e.g. Pan-STARRS 1). Finds the
        zero point which satisfies AB_mag = ZP + instrumental_mag and gets the 
        calibrated AB mags for all PSF-fit sources. 
        
        Output: None 
        """
        # compute any missing masks/background
        if not(type(self.bkg) == np.ndarray): 
            # --> forces all others to be built, too
            self.bkg_compute(box_size=box_size, 
                             filter_size=filter_size, 
                             thresh_sigma=thresh_sigma) 
        
        self.__fit_PSF(nstars, thresh_sigma, pixelmin, elongation_lim, 
                       area_max, cutout, source_lim, 
                       write_ePSF, ePSF_data_name,
                       plot_ePSF, ePSF_name, 
                       plot_resid, resid_name, 
                       verbose)
        
        if self.psf_fit: # if the PSF Was properly fit 
            self.__zero_point(sep_max, plot_corr, corr_name, 
                              plot_source_offsets, source_offs_name, 
                              plot_field_offsets, field_offs_name, 
                              gaussian_blur_sigma, cat_num)
    
        
    def write_PSF_photometry(self, nstars=40, thresh_sigma=5.0, pixelmin=20, 
                             elongation_lim=1.4, area_max=500, cutout=35, 
                             source_lim=None, gaussian_blur_sigma=30.0, 
                             cat_num=None, sep_max=2.0, verbose=True,  
                             box_size=50, filter_size=5, 
                             write_ePSF=False, ePSF_data_name=None,
                             plot_ePSF=True, ePSF_name=None,
                             plot_resid=False, resid_name=None,
                             plot_corr=True, corr_name=None,
                             plot_source_offsets=True, source_offs_name=None,
                             plot_field_offsets=False, field_offs_name=None, 
                             output=None):
        """
        Input: the same as PSF_photometry, with an additional arg for the 
        filename of the output file 
        
        Performs PSF photometry if it has not already been performed, and then
        writes a table of the PSF-fit sources to a .fits table. 
        
        Output: None
        """
        
        if not(self.photometric_calib):
            print("\nObtaining photometric calibration...")
            self.PSF_photometry(nstars, thresh_sigma, pixelmin, 
                                elongation_lim, area_max, cutout, source_lim, 
                                gaussian_blur_sigma, cat_num, sep_max, 
                                verbose, box_size, filter_size,
                                write_ePSF, ePSF_data_name,
                                plot_ePSF, ePSF_name, 
                                plot_resid, resid_name,
                                plot_corr, corr_name, 
                                plot_source_offsets, source_offs_name, 
                                plot_field_offsets, field_offs_name)
        
        if self.photometric_calib: # if successful
            to_write = self.psf_sources 
            
            if not(output): # if no name given
                output = self.stack_name.replace("_updated.fits", 
                                                 "_PSF_photometry.fits")
            to_write.write(output, overwrite=True)
        else: 
            print("\nPhotometry failed, so no file was written. Exiting.")
            
        
    def adjust_astrometry(self, ra=None, dec=None):
        """            
        Input: 
            - ra, dec by which to adjust the reference pixel (optional; default
              None, in which case the offsets are taken from PSF photometry)
        
        If PSF photometry has been completed, then the average offset in ra, 
        dec between the astrometric solution and the coorinates of the 
        relevent catalog is known and can be used to adjust the astrometric 
        solution of the image. PSF photometry can then be re-done. OR, if the 
        offset is known via some other method, this can be used to correct it.
        
        ** Not very reliable right now.
        
        Output: None
        """

        # move to calibration directory 
        script_dir = os.getcwd()
        os.chdir(self.calib_dir)
        
        # update the main stack file 
        f = fits.open(self.stack_name, mode="update")
        if not(ra) and not(dec): # if no RA, Dec given, automatic adjustment
            if not(self.photometric_calib):
                print("\nCannot adjust astrometry for the offset from the "+
                  "relevant catalog because PSF_photometry has not yet been "+
                  "called. Please use this function first. Exiting.")
                return
            print(f'\nThe astrometry of {self.stack_name} has been updated '+
                  'according to the offsets computed in PSF_photometry. '+
                  'These offsets are: '+
                  f'\nRA_offset = {self.ra_offsets_mean:.4f}" '+
                  f'\nDec_offset = {self.dec_offsets_mean:.4f}" ')
            f[0].header["CRVAL1"] -= self.ra_offsets_mean/3600.0
            f[0].header["CRVAL2"] -= self.dec_offsets_mean/3600.0
        else: # if RA, Dec given, manual adjustment
            f[0].header["CRVAL1"] -= ra
            f[0].header["CRVAL2"] -= dec
        f.close()
        self.__image_header = fits.getheader(self.stack_name)
        
        # separations are now approx. 0, can be computed again by running
        # PSF_photometry once more 
        self.__sep_mean = 0.0  
        self.__ra_offsets_mean = 0.0
        self.__dec_offsets_mean = 0.0
              
        print("Offset attributes have been set to 0.0, but will be updated if"+
              " PSF_photometry is called again.")
        
        os.chdir(script_dir)    

###############################################################################
    #### APERTURE PHOTOMETRY ##################################################

    def __drop_aperture(self, ra, dec, ap_radius=1.2, r1=2.0, r2=5.0,
                        plot_annulus=False, ann_name=None,
                        plot_aperture=False, ap_name=None, bkgsub_verify=True):
        """
        Input: 
            - ra, dec (in degrees) of a source around which to build an 
              aperture of radius ap_radius (in arcsec; optional; default 0.9")
            - radius of aperture to place around "source" (optional; default 
              1.2")
            - inner r1 and outer r2 radii of annulus in which to estimate the 
              background in the region (in arcsec; optional; default 2.0",5.0") 
            - whether or not to plot the annulus image data (optional; default 
              False)
            - whether or not to plot the aperture and annulus as rings 
              (optional; default False)
            - name for the output annulus plot (optional; default set below)
            - name for the output aperture plot (optional; default set below)
            - whether to verify that the background-subtracted flux is positive
              (optional; default True)
        
        This method finds the total flux in a defined aperture, computes the 
        background in an annulus around this aperture, and computes the 
        background-subtracted flux of the "source" defined by the aperture.
        
        Output: a table containing the pix coords, ra, dec, aperture flux, 
        aperture radius, annulus inner and outer radii, the median background, 
        total background in aperture, standard deviation in this background, 
        and background-subtracted aperture flux 
        """
                
        # wcs object
        w = wcs.WCS(self.image_header)
        
        # lay down the aperture 
        position = SkyCoord(ra, dec, unit="deg", frame="icrs") # source posn
        ap = SkyCircularAperture(position, r=ap_radius*u.arcsec) # aperture 
        ap_pix = ap.to_pixel(w) # aperture in pix
        
        # build a bad pixel mask which excludes negative pixels
        bp_mask = np.logical_or(self.image_data<=0, self.bp_mask)
        
        # table of the source's x, y, and total flux in aperture
        # mask out only bad pixels
        phot_table = aperture_photometry(self.image_data, ap_pix, 
                                         error=self.image_error,
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
        image_data_masked = np.ma.masked_where(self.source_mask, 
                                               self.image_data)
        image_data_masked.fill_value = 0 # set to zero
        image_data_masked = image_data_masked.filled()
        annulus_data = annulus_masks[0].multiply(image_data_masked)
        
        try: # mask invalid data in specific annulus
            ann_mask = np.logical_or(annulus_data==0, np.isnan(annulus_data))
        except TypeError: # if annulus_data is None
            print("\nThere is no annulus data at this aperture. Either the "+
                  "input target is out of bounds or the entire annulus is "+
                  "filled with sources. Consider using a different radius "+
                  "for the aperture/annuli. Exiting.")
            return      
        annulus_data = np.ma.masked_where(ann_mask, annulus_data)
        
        # estimate background as median in the annulus, ignoring data <= 0 
        annulus_data_1d = annulus_data[annulus_data>0]
        bkg_mean, bkg_med, bkg_std = sigma_clipped_stats(annulus_data_1d)
        bkg_total = bkg_med*ap_pix.area()
        
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
                  "Alternatively, get a limiting magnitude at these "+
                  "coorindates instead. Exiting.")
            return         
        
        if plot_annulus:
            self.__plot_annulus(annulus_data, ra, dec, r1, r2, ann_name)   
        if plot_aperture:
            self.__plot_aperture(annulus_apertures, ra, dec, ap_pix, 
                                 r1, r2, ap_name)  
        return phot_table
    
    
    def __plot_annulus(self, annulus_data, ra, dec, r1, r2, ann_name=None):
        """
        Input: 
            - data of the annulus itself (2D array)
            - ra, dec for the centre of an annulus
            - inner and outer radii for the annuli
            - name for the output annulus plot (optional; default set below)
            
        Plots an image of the annulus for a given aperture computation. 
        
        Output: None
        """
        # plotting
        fig, ax = plt.subplots(figsize=(10,10)) 
        plt.imshow(annulus_data, origin="lower", cmap="magma")
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        cb.set_label(label="ADU",fontsize=15)
        plt.title(f'Annulus around {ra:.5f}, {dec:.5f} '+
                  f'(1 pixel = {self.pixscale:.3f}")', fontsize=15)
        plt.xlabel("Pixels", fontsize=15)
        plt.ylabel("Pixels", fontsize=15)
        
        # textbox indicating inner/outer radii of annulus 
        textstr = r'$r_{in} = %.1f$"'%r1+'\n'+r'$r_{out} = %.1f$"'%r2
        box = dict(boxstyle="square", facecolor="white", 
           alpha=0.6)
        plt.text(0.81, 0.91, transform=ax.transAxes, s=textstr, bbox=box,
                 fontsize=14)
        
        if not(ann_name):
            ann_name = f"{self.filter}_{self.instrument}_{self.date}"
            ann_name = f"{ann_name}_annulus_RA{ra:.5f}_DEC{dec:.5f}"
            ann_name = f"{ann_name}.{self.plot_ext}"
        plt.savefig(ann_name, bbox_inches="tight")
        plt.close()
        
        
    def __plot_aperture(self, annulus_pix, ra, dec, ap_pix, r1, r2, 
                        ap_name=None):
        """
        Input: 
            - pixel data of the *annulus* 
            - ra, dec for the centre of the aperture
            - radius of the aperture, inner annulus, and outer annulus
            - name for the output aperture plot
            
        Plots an image of the aperture and annuli drawn around a source of 
        interest for aperture photometry.
        
        Output: None
        """        
        # wcs object
        w = wcs.WCS(self.image_header)

        # update wcs object and image to span a box around the aperture
        xpix, ypix = ap_pix.positions[0] # pix coords of aper. centre 
        boxsize = int(annulus_pix.r_out)+5 # size of box around aperture 
        idx_x = [int(xpix-boxsize), int(xpix+boxsize)]
        idx_y = [int(ypix-boxsize), int(ypix+boxsize)]
        w.wcs.crpix = w.wcs.crpix - [idx_x[0], idx_y[0]] 
        image_data_temp = self.image_data[idx_y[0]:idx_y[1], idx_x[0]:idx_x[1]] 
        
        # update aperture/annuli positions 
        ap_pix.positions -= [idx_x[0], idx_y[0]] 
        annulus_pix.positions -= [idx_x[0], idx_y[0]] 
        
        # plotting
        plt.figure(figsize=(10,10))
        ax = plt.subplot(projection=w) # show wcs 
        plt.imshow(image_data_temp, origin="lower", cmap="magma")
        ap_pix.plot(color='white', lw=2) # aperture as white cirlce
        annulus_pix.plot(color='red', lw=2) # annuli as red circles 
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        cb.set_label(label="ADU", fontsize=15)
        plt.title(f"Aperture photometry around {ra:.5f}, {dec:.5f}", 
                  fontsize=15)
        textstr = r'$r_{aper} = %.1f$"'%(ap_pix.r*self.pixscale)+'\n'
        textstr += r'$r_{in} = %.1f$"'%r1+'\n'
        textstr += r'$r_{out} = %.1f$"'%r2
        box = dict(boxstyle="square", facecolor="white", alpha=0.6)
        plt.text(0.83, 0.88, transform=ax.transAxes, s=textstr, bbox=box, 
                 fontsize=14)
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)

        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        if not(ap_name):
            ap_name = f"{self.filter}_{self.instrument}_{self.date}"
            ap_name = f"{ap_name}_aperture_RA{ra:.5f}_DEC{dec:.5f}"
            ap_name = f"{ap_name}.{self.plot_ext}"       
        
        plt.savefig(ap_name, bbox_inches="tight")
        plt.close()
        
    
    def aperture_photometry(self, ra_list, dec_list, 
                            sigma=None, bkgsub_verify=True, 
                            ap_radius=1.2, r1=2.0, r2=5.0, 
                            box_size=50, filter_size=5, 
                            thresh_sigma=3.0,
                            plot_annulus=False, ann_name=None,
                            plot_aperture=False, ap_name=None):
        """
        Input: 
            - list OR single float/int of ra, dec of interest
            - limiting sigma below which a source is labelled non-detected 
              (optional; default no limit)
            - radius for the aperture (in arcsec; optional; default 1.2")
            - inner radius r1 and outer radius r2 for the annulus (in arcsec; 
              optional; default 2.0", 5.0")
            
            - box size for the background computation (optional; default 50x50;
              only relevant if background not found beforehand)
            - size for the median filter applied during background computation
              (optional; default 5x5; only relevant if background not found 
              beforehand) 
            - sigma to use as the threshold for image segmentation (optional; 
              default 3.0; only relevant if a source mask does not exist for 
              the object and needs to be computed)
            
            - whether to plot the annulus (optional; default False)
            - name for the output annulus plot (optional; default set below)
            - whether to plot the aperture (optional; default False)
            - name for the output aperture plot (optional; default set below)
        
        Computes the the total flux in a defined aperture around the given RA,
        Dec, computes the background in an annulus around this aperture, and 
        computes the background-subtracted flux of the "source" defined by the 
        aperture. Can be called multiple times if a list of RA/Dec is given. 
        
        Output: None
        """        

        # if PSF photometry has not been performed, can't get aperture mags 
        if not(self.photometric_calib):
            print("Cannot obtain magnitudes through aperture photometry "+ 
                  "because photometric calibration has not yet been obtained."+
                  " Exiting.\n")
            return
            
        # compute any missing masks and/or the error array
        # --> forces all others to be built, too 
        if not(type(self.image_error) == np.ndarray): 
            self.error_array(box_size=box_size, filter_size=filter_size,
                             thresh_sigma=thresh_sigma) 
            
        # initialize table of sources found by aperture photometry if needed
        if not(self.aperture_fit):
            cols = ["xcenter","ycenter", "ra","dec", "aperture_sum", 
                    "aperture_sum_err", "aper_r", "annulus_r1", "annulus_r2",
                    "annulus_median", "aper_bkg", "aper_bkg_std", 
                    "aper_sum_bkgsub", "aper_sum_bkgsub_err", "mag_fit", 
                    "mag_unc", "mag_calib", "mag_calib_unc", "sigma"]
            self.__aperture_sources = Table(names=cols)
            filt_col = Column([], "filter", dtype='S2') # specify
            mjd_col = Column([], "MJD")
            self.__aperture_sources.add_column(filt_col)
            self.__aperture_sources.add_column(mjd_col)
                
        # convert to lists if needed 
        if (type(ra_list) in [float, int]):
            ra_list = [ra_list]
        if (type(dec_list) in [float, int]):
            dec_list = [dec_list]
        
        # compute background-subtracted flux for the input aperture(s) 
        # add these to the list of sources found by aperture photometry 
        print("\nAttemtping to perform aperture photometry...")
        for i in range(0, len(ra_list)):
            phot_table = self.__drop_aperture(ra_list[i], dec_list[i],
                                              ap_radius, r1, r2, 
                                              plot_annulus, ann_name,
                                              plot_aperture, ap_name,
                                              bkgsub_verify)
                                               
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
                mag_calib = phot_table['mag_fit'] + self.zp_mean
                mag_calib.name = 'mag_calib'
                mag_calib_unc = np.sqrt(phot_table['mag_unc']**2 + 
                                        self.zp_std**2)
                mag_calib_unc.name = 'mag_calib_unc'
                phot_table['mag_calib'] = mag_calib
                phot_table['mag_calib_unc'] = mag_calib_unc
                 
                # compute sigma 
                phot_table["sigma"] = phot_table['aper_sum_bkgsub']
                phot_table["sigma"] /= phot_table['aper_sum_bkgsub_err']
                
                # other useful columns 
                col_filt = Column(self.filter, "filter",
                                  dtype = np.dtype("U2"))
                col_mjd = Column(self.stack_time, "MJD")
                phot_table["filter"] = col_filt
                phot_table["MJD"] = col_mjd
                phot_table.remove_column("id") # id is not accurate 
                
                if sigma and (phot_table["sigma"] >= sigma):
                    self.__aperture_sources.add_row(phot_table[0])
                    self.__aperture_fit = True # update                  
                elif sigma and (phot_table["sigma"] < sigma):
                    print("\nA source was detected, but below the requested "+
                          f"{sigma} sigma level. The source is therefore "+
                          "rejected.")
                    return
                else:
                    self.__aperture_sources.add_row(phot_table[0])
                    self.__aperture_fit = True # update 
                    
                a = phot_table[0]
                s = f'\n{a["filter"]} = {a["mag_calib"]:.2f} +/- '
                s += f'{a["mag_calib_unc"]:.2f}, {a["sigma"]:.1f} sigma'
                print(s)
                
                
    def limiting_magnitude(self, ra, dec, sigma=5.0, 
                           thresh_sigma=3.0, box_size=50, filter_size=5,
                           plot_annulus=True, ann_name=True, 
                           plot_aperture=None, ap_name=None,
                           write=False, output=None):
        """        
        Input: 
            general:
            - ra, dec of interest
            - sigma defining the limiting magnitude (optional; default 5.0)
            
            background estimation:
            - box size for the background computation (optional; default 50x50;
              only relevant if background not found beforehand)
            - size for the median filter applied during background computation
              (optional; default 5x5; only relevant if background not found 
              beforehand) 
            
            image segmentation:
            - sigma to use as the threshold for image segmentation (optional; 
              default 3.0; only relevant if a source mask does not exist for 
              the object and needs to be computed)

            writing, plotting:
            - whether to plot the annulus (optional; default True)
            - whether to plot the aperture (optional; default True), 
            - name for the output annulus plot (optional; defaults set below) 
            - name for the output aperture plot (optional; defaults set below)            
            - whether to write the resultant table (optional; default False)
            - name for output table file (optional; default set below; only 
              relevant if write=True)
            
        For a given RA, Dec, finds the limiting magnitude at its location. If 
        a source was previously detected <= 3" away from the given coords by 
        astrometry.net, the aperture will randomly move about until a valid
        RA, Dec is found. 
        
        Output: the limiting magnitude 
        """

        # if PSF photometry has not been performed, can't get aperture mags 
        if not(self.photometric_calib):
            print("Cannot obtain magnitudes through aperture photometry "+ 
                  "because photometric calibration has not yet been obtained."+
                  " Exiting.\n")
            return

        # compute any missing masks and/or the error array
        # --> forces all others to be built, too 
        if not(type(self.image_error) == np.ndarray): 
            Stack.error_array(self, box_size=box_size, filter_size=filter_size,
                              thresh_sigma=thresh_sigma) 

        # initialize table of sources if needed
        if not(self.limmag_obtained):
            cols = ["ra", "dec", "lim_mag"]
            self.__limmag_sources = Table(names=cols)
            filt_col = Column([], "filter", dtype='S2') # specify
            mjd_col = Column([], "MJD")
            self.__limmag_sources.add_column(filt_col)
            self.__limmag_sources.add_column(mjd_col)


        # get coords of all sources 
        sources_data = self.xy_data 
        w = wcs.WCS(self.image_header)
        coords = w.all_pix2world(sources_data["X"], sources_data["Y"], 1)
        coords = SkyCoord(coords[0]*u.deg, coords[1]*u.deg, 1)
        target = SkyCoord(ra*u.deg, dec*u.deg)
        
        # find the smallest separation between target and all sources
        smallest_sep = np.min(target.separation(coords).value)*3600.0 
        while smallest_sep < 3.0: # while closest star is less than 3" away
            print('\nastrometry.net previously found a source < 3.0" away '+
                  'from the target. The target aperture will be randomly '+
                  'moved until it does not sit on top of a source...')
  
            # randomly move           
            new_ra = target.ra - u.arcsec*np.random.randint(-5, 5)
            new_dec = target.dec - u.arcsec*np.random.randint(-5, 5)
            
            # make sure it hasn't gone too far, return to center if so
            x, y = w.all_world2pix(new_ra.value, new_dec.value, 1)
            if (x<0.1*self.x_size) or (y<0.1*self.y_size) or (
                x>0.9*self.x_size) or (y>0.9*self.y_size):
                new = w.all_pix2world(self.x_size//2, self.y_size//2, 1)
                new_ra, new_dec = new[0]*u.deg, new[1]*u.deg
            
            target = SkyCoord(new_ra, new_dec)
            smallest_sep = np.min(target.separation(coords).value)*3600.0
            
        ra, dec = target.ra.value, target.dec.value

        print("\nFinding the limiting magnitude at (RA, Dec) = "+
              f"({ra:.4f}, {dec:.4f})")
        
        # do aperture photometry on region of interest with large annulus
        phot_table = self.__drop_aperture(ra, dec, 
                                          ap_radius=1.2, r1=2.0, r2=20.0, 
                                          plot_annulus=plot_annulus,
                                          ann_name=ann_name, 
                                          plot_aperture=plot_aperture,
                                          ap_name=ap_name,
                                          bkgsub_verify=False)
 
        phot_table["aper_sum_bkgsub_err"] = np.sqrt(
                phot_table["aperture_sum_err"]**2 +
                phot_table["aper_bkg_std"]**2)
        
        # compute limit below which we can't make a detection
        limit = sigma*phot_table["aper_sum_bkgsub_err"][0]
        self.__limiting_mag = -2.5*np.log10(limit) + self.zp_mean       
        print(f"\n{self.filter} > {self.limiting_mag:.1f} ({sigma:d} sigma)")
        
        # add table with other info as attribute to the object
        lim_table = Table(data=[[ra], [dec], [self.limiting_mag], 
                                [self.filter], [self.stack_time]],
                          names=["ra","dec","mag_calib","filter","MJD"])
        self.__limmag_sources.add_row(lim_table[0])
        self.__limmag_obtained = True
    
        return self.limiting_mag


    def write_aperture_photometry(self, output=None):
        """
        Input: name for the output file which will contain a table of the 
        sources found by aperture photometry

        Writes the table of sources detected by aperture photometry to a file.
        
        Output: None
        """
        
        if not(self.aperture_fit):
            print("No aperture photometry has been performed. Exiting.\n")
            return
            
        to_write = self.aperture_sources
        
        if not(output): # if no name given
            output = self.stack_name.replace("_updated.fits", 
                                             "_aperture_photometry.fits")
        to_write.write(output, overwrite=True, format="ascii.ecsv")


    def write_limiting_magnitude(self, output=None):
        """
        Input: name for the output file which will contain a table of the 
        sources with limiting magnitudes 

        Writes the table of sources with aperture photometry limiting 
        magnitudes to a file.
        
        Output: None
        """
        
        if not(self.limmag_obtained):
            print("No limiting magnitude has been obtained. Exiting.\n")
            return
            
        to_write = self.limmag_sources
        
        if not(output): # if no name given
            output = self.stack_name.replace("_updated.fits", "_limmag.fits")
        to_write.write(output, overwrite=True, format="ascii.ecsv")   
        
###############################################################################
### COMPARE APERTUTRE AND PSF PHOTOMETRY ######################################
        
    def compare_photometry(self, ap_radius=1.2, r1=2.0, r2=5.0, nsamples=100, 
                           output=None):
        """
        Input: 
            - radius for the aperture (in arcsec; optional; default 1.2")
            - inner radius r1 and outer radius r2 for the annulus (in arcsec; 
              optional; default 2.0", 5.0")
            - inner and outer annuli radii, the number of samples to take from 
              the the PSF photometry calibrated data (optional; default 100) 
            - name for the output figure (optional; default set below)
            
        For nsamples random sources, compares the calibrated magnitude obtained
        via aperture photometry to that obtained by PSF photometry. 
        
        Output: None 
        """ 
        temp = self.aperture_sources # store original table 
        
        # obtain coords and magnitudes of nsamples PSF-fit sources
        sample = self.psf_sources[0:nsamples]
        ras = sample["ra"].data
        decs = sample["dec"].data
        psf_mags = sample["mag_calib"].data
        psf_mags_errs = sample["mag_calib_unc"].data
        
        # perform aperture photometry on the same sources 
        self.aperture_photometry(ras, decs, ap_radius, r1, r2)
        
        # get the magnitudes from aperture photometry for these sources 
        ap_sample = self.aperture_sources[-nsamples:] 
        ap_mags = ap_sample["mag_calib"].data
        ap_mags_errs = ap_sample["mag_calib_unc"].data
        
        # fit a straight line to the correlation
        from scipy.optimize import curve_fit
        def f(x, m, b):
            return b + m*x
        popt, pcov = curve_fit(f, ap_mags, psf_mags)
        m, b = popt
        perr = np.sqrt(np.diag(pcov))
        m_err, b_err = perr
        fitdata = [m*x + b for x in ap_mags]
        
        # plot the data
        fig, ax = plt.subplots(figsize=(8,8))
        ax.errorbar(ap_mags, psf_mags, xerr=ap_mags_errs, yerr=psf_mags_errs,
                    ls="", marker=".", mfc="#8e82fe", mec="#8e82fe", color="k", 
                    label=f"Data [{self.filter}]", zorder=1)
        # plot the linear fit to the data
        lcorr=r"$y = mx + b $"+"\n"+r"$ m=$%.3f$\pm$%.3f, $b=$%.3f$\pm$%.3f"%(
                m, m_err, b, b_err)
        ax.plot(ap_mags, fitdata, color="blue", label=lcorr, zorder=2)
        ax.set_xlabel("Aperture photometry calibrated magnitude", fontsize=15)
        ax.set_ylabel("PSF photometry calibrated magnitude "+
                      f"[{self.ref_cat_name}]", fontsize=15)
        ax.legend(loc="upper left", fontsize=15)
        ax.set_title("Comparison between different methods of photometry "+
                     f"[no. of samples = {nsamples:d}", fontsize=15)
        
        if not(output): # if no name given
            output = self.stack_name.replace("_updated.fits", 
                                             "_photometry_compare"+
                                             f".{self.plot_ext}")
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        
        self.__aperture_sources = temp # restore original table 
        
###############################################################################
### SOURCE SELECTION ##########################################################

    def source_selection(self, ra, dec, radius=1.0):
        """
        WIP: 
            - add parsing of aperture photometry table
        
        Input: 
            - RA, Dec for a source of interest and 
            - radius to search in (in arcsec; optional; default 1.0")
            
        Parses the table of sources produced by PSF_photometry() for sources 
        within some distance from the given coordinates. 

        Output: a table of the source(s) satisfying the conditions
        """
        # parse PSF sources 
        radius = radius/3600.0 # convert to degrees
        mask = (abs(self.psf_sources["ra"] - ra) <= radius) & (
                abs(self.psf_sources["dec"] - dec) <= radius)
        selected_psf_sources = self.psf_sources[mask]
        # remove unnecessary columns 
        selected_psf_sources.remove_columns(['x_0','y_0','flux_0','id',
                                             'group_id','x_fit','y_fit',
                                             'flux_fit','flux_unc',
                                             'mag_fit','mag_unc', 
                                             f'in {self.ref_cat_name}'])
        return selected_psf_sources # return selected sources in astropy table 
    
    
    def write_selection(self, ra, dec, radius=1.0, output=None):
        """
        Input: the same as source_selection, with an additional arg for the 
        filename of the output file 
        
        Selects sources as per source_selection() and then writes them to an 
        astropy table with the correct format.
        
        Output: None
        """
        psf_selection = self.source_selection(ra, dec, radius)
        
        if not(output): # if no name given
            output = self.stack_name.replace("_updated.fits", 
                                             "_PSF_photom_selection.fits")
        psf_selection.write(output, overwrite=True, format="ascii.ecsv")
    
