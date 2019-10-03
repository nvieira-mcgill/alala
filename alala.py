#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 9 11:08:26 2019
@author: Nicholas Vieira
@alala.py

SECTIONS:
    - RawData class and basic functions
        - Image diagnostics + pre-solving before stacking
        - Locating coordinates among raw data + writing extensions 
        - Combining/divding (WIRCam) cubes
        - Cropping images
        - Stacking and stack preparation (bad pixel masks)
        
    - Stack class, making images, and astrometry
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

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.visualization import simple_norm
from photutils import Background2D, MMMBackground

###############################################################################
    ### RawData CLASS AND BASIC FUNCTIONS ###

class RawData:
    def __init__(self, location, stack_directory=None, qso_grade_limit=None,
                 fmt="fits"):
        """
        Input: location of the raw data, directory to store potential stacks 
        (optional; not needed if no intention to stack), and the limit on the 
        QSO grade for the observations (optional; default is to apply no limit 
        where 1=good and 5=unusable, so that no data is ignored unless 
        explicitly desired by the user; only relevant for WIRCam)
        
        Initializes a raw data object for CFHT WIRCam or MegaPrime data.
        
        Output: a RawData object
        """
        self.loc = location # location of data
        self.files = (os.listdir(self.loc)) # all files
        (self.files).sort()
        self.files = [f for f in self.files if fmt in f] # fits only
        self.stack_dir = stack_directory # location of stacks to be produced
        self.fmt = fmt
        
        # remove / from end of location/stack_dir if necessary
        if self.loc[-1] == "/":
            self.loc = self.loc[:-1]
        if stack_directory: 
            if self.stack_dir[-1] == "/":
                self.stack_dir = self.stack_dir[:-1]
                    
        # ***assume all extensions and files have same instrument/nextend:
        file = self.files[0]
        hdu = fits.open(self.loc+"/"+file)[0]
        self.instrument = hdu.header["INSTRUME"] # instrument in use
        try:
             self.nextend = int(hdu.header["NEXTEND"]) # no. of extensions
        except KeyError:
            self.nextend = 0
        
        # check the QSO grade of the files and remove from the list any which 
        # have a QSO grade of 5 (unusable); this DOES NOT delete the files
        self.qso_grade_limit = qso_grade_limit
        if "WIRCam" in self.instrument and qso_grade_limit:
            temp = []
            for f in self.files:
                if self.nextend == 0: # if a single image
                    hdr = fits.open(self.loc+"/"+f)[0].header
                else: # if a cube
                    hdr = fits.getheader(self.loc+"/"+f)
                if int(hdr["QSOGRADE"]) <= qso_grade_limit: 
                    temp.append(f)
            self.files = temp
                
        # check if data spans a single date or multiple dates
        # lazy check: only checking first and last files 
        # date is of format YYYY-MM-DDTHH:mm:ss, e.g. 2011-09-12T08:36:19
        # first 10 chars only and remove all - chars
        date1 = (fits.open(self.loc+"/"+self.files[0])[0]).header["DATE"]
        date1 = (date1[0:10]).replace("-","")
        date2 = (fits.open(self.loc+"/"+self.files[-1])[0]).header["DATE"]
        date2 = (date2[0:10]).replace("-","")
        
        if date1 == date2: # same day month year 
            self.date = (hdu.header["DATE"][0:10]).replace("-","")
        elif date1[0:6] == date2[0:6]: # if same month and year
                self.date = date1[0:6] 
        elif date1[0:4] == date2[0:4]: # same year
            self.date  = date1[0:4]
        else:            
            self.date = "multiyear"
        RawData.__dates_init(self) 
            
        # MJD at start of observations for *first* file in directory
        self.mjdate = (fits.open(
                self.loc+"/"+self.files[0])[0]).header["MJDATE"]
        
        # choose potential filters based on instrument
        if "WIRCam" in self.instrument:
            # broadband filters:
            self.J = [] # 1253 +/- 79
            self.H = [] # 1631 +/- 144.5
            self.Ks = [] # 2146 +/- 162.5
            # narrow-band filters:
            self.Y = [] # 1020 +/- 50
            #self.OH_1 = []
            #self.OH_2 = []
            #self.CH4_on = []
            #self.CH4_off = []
            #self.W = []
            #self.H2 = []
            #self.K_cont = []
            #self.bracket_gamma = []
            #self.CO = []
            self.filters=['Y','J','H','Ks'] 
            #self.filters=["Y","J","H","Ks","OH-1","OH-2","CH4_on","CH4_off",
            #              "W","H2","K_cont","bracket_gamma","CO"]
        
        elif "MegaPrime" in self.instrument:
            if self.mjdate > 57023: # if after 1 January 2015
                ### new filters system
                self.u = [] # 355 +/- 43
                self.g = [] # 475 +/- 77
                self.r = [] # 640 +/- 74
                self.i = [] # 776 +/- 77.5
                self.z = [] # 925 +/- 76.5
                # since 2015A, old filters denoted with trailing S
                # they were retired in 2017, but for a brief period, PIs could
                # use both the old and the new 
                self.uS = [] # 375 +/- 37
                self.gS = [] # 487 +/- 71.5
                self.rS = [] # 630 +/- 62
                self.iS = [] # 770 +/- 79.5
                self.zS = [] # N/A, 827 to ...
                self.filters = ["u","g","r","i","z","uS","gS","rS","iS","zS"]
            else:  
                ### old filters system
                self.u = [] # 375 +/- 37
                self.g = [] # 487 +/- 71.5
                self.r = [] # 630 +/- 62
                self.i = [] # 770 +/- 79.5
                self.z = [] # N/A, 827 to ...
                self.filters = ['u','g','r','i','z']
              
        # construct lists of files for each filter as well as a dictionary
        RawData.__filter_init(self)
        
        # filetype (png, pdf, jpg) for plots (can be set)
        self.plot_ext = "png"
        
        # bools for later
        self.stack_made = False
        
        
    def __filter_init(self):
        """
        Input: None
        Records which filters are used for each raw data file and stores them 
        in attributes of the object. Constructs a dictionary where the keys 
        are the given filters and the values are arrays containing all of the 
        files with the corresponding filters.
        Output: None
        """
        l = self.loc 

        # assume all extensions have same filter for a given file
        if "WIRCam" in self.instrument:  # if WIRCam data
            self.Y, self.J, self.H, self.Ks = [], [], [], []
            for f in self.files:
                hdu_temp = fits.open(l+"/"+f)
                hdu = hdu_temp[0]
                if 'Y' in hdu.header["FILTER"]:
                    self.Y.append(f)
                elif 'J' in hdu.header["FILTER"]:
                    self.J.append(f)
                elif 'H' in hdu.header["FILTER"]:
                    self.H.append(f)     
                elif 'Ks' in hdu.header["FILTER"]:
                    self.Ks.append(f)
                hdu_temp.close()
                
            self.filters = ["Y", "J", "H", "Ks"]        
            filter_vals = [self.Y, self.J, self.H, self.Ks]
                    
        else: # if MegaPrime data
            self.u, self.g, self.r, self.i, self.z = [], [], [], [], []
            self.uS, self.gS, self.rS, self.iS, self.zS = [], [], [], [], []
            for f in self.files:
                hdu_temp = fits.open(l+"/"+f)
                hdu = fits.open(l+"/"+f)[0]
                if 'u' in hdu.header["FILTER"]:
                    self.u.append(f)
                elif 'g' in hdu.header["FILTER"]:
                    self.g.append(f)
                elif 'r' in hdu.header["FILTER"]:
                    self.r.append(f)     
                elif 'i' in hdu.header["FILTER"]:
                    self.i.append(f)
                elif 'z' in hdu.header["FILTER"]:
                    self.z.append(f)
                elif 'uS' in hdu.header["FILTER"]:
                    self.uS.append(f)
                elif 'gS' in hdu.header["FILTER"]:
                    self.gS.append(f)
                elif 'rS' in hdu.header["FILTER"]:
                    self.rS.append(f)
                elif 'iS' in hdu.header["FILTER"]:
                    self.iS.append(f)
                elif 'zS' in hdu.header["FILTER"]:
                    self.zS.append(f)
                hdu_temp.close()
            
            if self.mjdate > 57023: # if after 1 January 2015
                self.filters = ["u", "g", "r", "i", "z", 
                                "uS", "gS", "rS", "iS", "zS"]
                filter_vals = [self.u, self.g, self.r, self.i, self.z, 
                               self.uS, self.gS, self.rS, self.iS, self.zS]
            else: 
                self.filters = ["u", "g", "r", "i", "z"]
                filter_vals = [self.u, self.g, self.r, self.i, self.z]
                
        filter_keys = self.filters # keys for a dictionary to be produced   
        self.filters_dict = dict(zip(filter_keys, filter_vals))
        
        # get rid of unecessary filters in the dict/list
        all_filters = list(self.filters) # make a copy 
        for fil in all_filters:
            if len(self.filters_dict[fil]) == 0: # if no files for a filter
                del self.filters_dict[fil]
                delattr(self, fil)
                self.filters.remove(fil)

                
    def __dates_init(self):
        """
        Input: None
        Records which dates are spanned by each raw data file and stores them 
        in a list. Constructs a dictionary where the keys are the given dates 
        and the values are arrays containing all of the files with the 
        corresponding dates.
        Output: None
        """
        l = self.loc
        self.dates = []
        self.dates_dict = {}
        for f in self.files:
            hdu_temp = fits.open(l+"/"+f)
            hdu = hdu_temp[0]
            date = (hdu.header["DATE"][0:10]).replace("-","")
            hdu_temp.close()
            if not (date in(self.dates)):
                self.dates.append(date) # add to list
                self.dates_dict[date] = [] # add to dict
            self.dates_dict[date].append(f)        
            
            
    def set_plot_ext(self, new_ext):
        """
        Input: the new plot_ext to be used
        Output: None
        """
        valid_ext = ["pdf", "jpg", "bmp", "png"]
        
        if new_ext in valid_ext:
            self.plot_ext = new_ext
        else:
            print("\nPlease input a valid filetype from "+str(valid_ext)+". "+
                  "Exiting.")
            
            
    def set_stack_dir(self, stackdir):
        """
        Input: the directory to store stacked files
        Output: None
        """
        self.stack_dir = stackdir
        
        if self.stack_dir[-1] == "/": # get rid of / at end if needed 
            self.stack_dir = self.stack_dir[:-1]

        
    def exclude_date(self, date):
        """
        Input: a date in the format "YYYYMMDD", "YYYYMM", or "YYYY"
        Removes all raw data from the RawData object which was acquired on the
        input date.
        Output: None
        
        * Does not update the self.date attribute. This should be added in the 
        future.
        """
        l = self.loc
        all_files = self.files # make a copy 
        self.files = []
        for f in all_files: # get data for every file 
            hdu_temp = fits.open(l+"/"+f)
            hdu = hdu_temp[0]
            d = (hdu.header["DATE"][0:10]).replace("-","")
            hdu_temp.close()
            
            if not(date in d): # if file is NOT from the input date 
                self.files.append(f)
        
        RawData.__dates_init(self) # rebuild list/dict of dates
        RawData.__filter_init(self) # rebuild list/dict of filters
        
        if len(self.files) == 0:
            print("Warning: after this operation, the RawData object has no "+
                  "remaining data.") 

    
    def exclude_object(self, obj):
        """
        Input: an object which was the target of given observations 
        Removes all raw data from the RawData object which was aimed at the 
        given object or pointing.
        Output: None
        """
        l = self.loc
        all_files = self.files # make a copy 
        self.files = []
        for f in all_files: # get data for every file 
            hdu_temp = fits.open(l+"/"+f)
            hdu = hdu_temp[0]
            o = hdu.header["OBJECT"]
            hdu_temp.close()
            
            if not(obj in o): # if file is NOT of the input pointing 
                self.files.append(f)
        
        RawData.__dates_init(self) # rebuild list/dict of dates
        RawData.__filter_init(self) # rebuild list/dict of filters
        
        if len(self.files) == 0:
            print("Warning: after this operation, the RawData object has no "+
                  "remaining data.") 

    
    def only_date(self, date):
        """
        Input: a date in the format "YYYYMMDD", "YYYYMM", or "YYYY"
        Removes all raw data from the RawData object which was ***NOT*** 
        acquired on the input date.
        Output: None
        """ 
        self.files = []
        for d in self.dates: # look at all dates 
            if date in d: # if input date is in the list of dates 
                self.files.append(self.dates_dict[d])
        
        RawData.__dates_init(self) # rebuild list/dict of dates
        RawData.__filter_init(self) # rebuild list/dict of filters

        if len(self.files) == 0:
            print("Warning: "+date+" was not spanned by any of the raw data "+
                  "files. After this operation, the RawData object has no "+
                  "remaining data.")
        else:
            self.date = date # update the date 

    
    def only_object(self, obj):
        """
        Input: an object which was the target of given observations 
        Removes all raw data from the RawData object which was ***NOT*** 
        aimed at the given object or pointing.
        Output: None
        """
        l = self.loc
        all_files = self.files # make a copy 
        self.files = []
        for f in all_files: # get data for every file 
            hdu_temp = fits.open(l+"/"+f)
            hdu = hdu_temp[0]
            o = hdu.header["OBJECT"]
            hdu_temp.close()
            
            if obj in o: # if file IS of the input pointing 
                self.files.append(f)
        
        RawData.__dates_init(self) # rebuild list/dict of dates
        RawData.__filter_init(self) # rebuild list/dict of filters

        if len(self.files) == 0:
            print("Warning: "+obj+" was not targeted in any of the raw data "+
                  "files. After this operation, the RawData object has no "+
                  "remaining data.")
        
        
    def make_copy(self):
        """
        Input: None
        Produces a copy of the object.
        Output: None
        """
        pass
            
###############################################################################
    #### IMAGE DIAGNOSTICS #### 
    
    def print_headers(self, ext, *headers):
        """
        Input: file extension of interest and the header(s) of interest  
        For each raw data file, prints the headers of interest in readable 
        format, if the header is present. Does so for the extension given 
        through ext. For debugging, mostly. 
        Output: None
        """
        l = self.loc # for convenience 
        headers = list(headers) # convert tuple to list 
        
        # alert user that some headers are not present and remove them
        # buggy - can not handle when multiple headers are not present 
        testfile = self.files[0]
        hdu_test_temp = fits.open(l+"/"+testfile)
        hdu_test = hdu_test_temp[ext]
        for h in headers:
            try:
                test = hdu_test.header[h]
                del test
            except KeyError:
                print("Header '"+str(h).upper()+"' not found.\n")
                headers.remove(h)
                continue                
        
        if len(headers) == 0:
            return # if no headers left, quit
        
        headers_string = ""
        for h in headers:
            headers_string += h+"\t"
        toprint = "FILE\t\t\t"+headers_string
        print(toprint)
        # print the desired headers in readable format for all raw data files
        for f in self.files:
            toprint = f+"\t"
            hdu = fits.open(l+"/"+f)[ext]
            for h in headers:
                toprint += str(hdu.header[h])+"\t"
            print(toprint)
        hdu_test_temp.close()


    def value_at(self, ra, dec):
        """
        Input: a RA and Dec
        For all of the files contained in some RawData object, returns the ADU 
        value at the given RA and Dec, IF these coordinates are within the 
        image's bounds.
        Can be used to see if the ADU is set to 0.0 over a source of interest,
        if desired.
        Output: None
        """
        l = self.loc
        for f in self.files:
            data = fits.getdata(l+"/"+f)
            hdr = fits.getheader(l+"/"+f)
            w = wcs.WCS(hdr)
            xpix, ypix = w.all_world2pix(ra, dec, 1)
            xpix = int(xpix)
            ypix = int(ypix)
            if (0 < xpix < 2048.0) and (0 < ypix < 2048.0):
                print("ADU at (%d, %d) = %.2f"%(xpix, ypix, data[ypix][xpix]))
                
                
    def background(self):
        """
        Input: None
        Naively estimates the background as the mean of the image's ADU for 
        every raw image in order to see how the data varies. Does not mask 
        sources, but this is not important for the purpose of this function:
        to see if any data is dubious 
        Output: An array containing the background level
        """
        l = self.loc
        bg_levels = []
        for f in self.files:
            data = fits.getdata(l+"/"+f)
            bg_levels.append(np.median(data))
        return bg_levels
        
    
    def radial_PSFs(self, ra, dec, solved=True, adu_min=4000, adu_max=66000):
        """
        Input: a RA and Dec for a source of interest (should be a bright 
        source, but not one that saturates the detector) a bool indicating
        whether the input images have already been solved with astrometry 
        (optional; default True), and the minimum and maximum ADU level to show 
        in the profiles (optional; default 4000 and 66000, resectively)
        For all of the files in the raw data directory, produces and saves a 
        figure of the radial profile around the input RA and Dec. If 
        solved=False, first solves the image with astrometry.net and then 
        obtains the radial profiles. (Refined astrometric solution is required 
        to get an accurate radial profile.)
        Output: None
        """
        l = self.loc
        topfile = re.sub(".*/", "", l) # for a file /a/b/c, extract the "c"
        plots_dir = os.path.abspath(l+"/..")
        plots_dir +="/profs_RA%.3f_DEC%.3f_"%(ra, dec)+topfile
        run("mkdir -p "+plots_dir, shell=True)
        
        if not solved: # if astrometry hasn't been done yet
            solved_dir = os.path.abspath(l+"/..")+"/solved_"+topfile
            print("A refined astrometric solution is required for this "+
                  "function to work. Using astrometry.net to solve the "+
                  "images now. Solved .fits files will be saved in "+
                  solved_dir+"\n")
            RawData.solve_all(self)
            files = os.listdir(os.path.abspath(l+"/..")+"/solved_"+topfile)
            
        else: # if astrometry has already been done
            files = self.files
            solved_dir = l
        
        # the radial PSF 
        for f in files: 
            image_data = fits.getdata(solved_dir+"/"+f)
            image_header = fits.getheader(solved_dir+"/"+f)
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
            plt.title("Radial profile around %.5f, %.5f"%(ra, dec),fontsize=15)
            
            # annotate with date and time; filename; filter in use
            obs_date = image_header["DATE"]
            filt = image_header["FILTER"]
            box = dict(boxstyle="square", facecolor="white", alpha=0.8)
            box_y = adu_min + 0.85*(adu_max-adu_min)
            txt = obs_date+"\n"+f+"\n"+filt 
            output_fig = f.replace("."+self.fmt, 
                                   "_prof_RA%.3f_DEC%.3f."%(ra, dec)+
                                   self.plot_ext)
            plt.text(3, box_y, s=txt, bbox=box,fontsize=14)
            plt.savefig(plots_dir+"/"+output_fig)
            plt.close()


    def solve_all(self, depth=None):
        """
        Input: the number of stars to use in solving (optional; default no 
        limit)
        e.g. depth=100 will only use the 100 brightest stars
        
        Using astrometry.net, solve all files, and put them in a new 
        directory. This is necessary when the astrometric solution obtained by 
        CFHT is inaccurate and requires refining, or if you wish to plot the 
        radial PSFs of some point in the raw data images. 
        
        Output: None
        """
        l = self.loc
        script_dir = os.getcwd()
        os.chdir(l)
        
        image_header = fits.getheader(l+"/"+self.files[0])
        image_data = fits.getdata(l+"/"+self.files[0])
  
        pixscale = image_header["PIXSCAL1"] # pixel scale
        pixmin = pixscale-0.001
        pixmax = pixscale+0.001

        cent = [i//2 for i in image_data.shape]
        centy, centx = cent
        w = wcs.WCS(image_header)
        ra, dec = w.all_pix2world(centx, centy, 1) 
        radius = 1.0/6.0 # look in a radius of 10 arcmin
        
        for f in self.files: # astrometry on each file 
            options = "--no-verify --overwrite --no-plot --fits-image"
            options += " --new-fits "+f.replace("."+self.fmt,"_solved.fits")
            
            # options to speed up astrometry: pixscale and rough, RA, Dec
            options += " --scale-low "+str(pixmin)
            options += " --scale-high "+str(pixmax)
            options += " --scale-units app"
            options += " --ra "+str(ra)+" --dec "+str(dec)
            options += " --radius "+str(radius)
            options += " --cancel "+f.replace("."+self.fmt,"_solved.fits")
            
            if type(depth) in [float, int]:
                options += " --depth "+str(int(depth))
            elif depth:
                options += " --depth "+depth
            
            # run astrometry 
            run("solve-field "+options+" "+f, shell=True)
        
        # get rid of unneeded files
        run("rm *.axy", shell=True)
        run("rm *.corr", shell=True)
        run("rm *.match", shell=True)
        run("rm *.rdls", shell=True)
        run("rm *.solved", shell=True)
        run("rm *.xyls", shell=True)
        run("rm *.wcs", shell=True)           

        # make a list of solved files, move them to a new directory, 
        # and make a list of unsolved files 
        topfile = re.sub(".*/", "", l) # for a file /a/b/c, extract the "c"
        solved_dir = os.path.abspath(l+"/..")+"/solved_"+topfile
        run("mkdir -p "+solved_dir, shell=True)
        run("rm -f "+solved_dir+"/*.fits", shell=True) # empty existing dir
        run("rm -f "+solved_dir+"/*.txt", shell=True) # empty existing dir
        
        solved = []
        unsolved = []
        files = [f.replace("."+self.fmt,"_solved.fits") for f in self.files]
        for f in files: 
            if os.path.exists(l+"/"+f):
                solved.append(f.replace("_solved.fits", "."+self.fmt))
                run("mv "+l+"/"+f+" "+solved_dir, shell=True)
            else:
                unsolved.append(f.replace("_solved.fits", "."+self.fmt))
        
        # save a text file w list of unsolved files, if necessary
        if len(unsolved) != 0:
            np.savetxt(solved_dir+"/unsolved.txt", unsolved, fmt="%s")
            print("\nThe following images could not be solved:")
            for f in unsolved:
                print(f)
            print("\nThese filenames have been recorded in a file "+solved_dir+
                  "/unsolved.txt")
        
        if len(solved) != 0:
            print("\nSolved the following images from "+self.instrument+" on "+
                  self.date+":")
            for f in solved:
                print(f)
        print("\nThese have been written to new solved .fits files in "+
              solved_dir)
            
        os.chdir(script_dir)
             
###############################################################################
    #### WCS LOCATING & EXTENSION WRITING ####

    def WCS_check(self, ra, dec, frac=1.0, verbose=True, checkall=True):
        """
        Input: an RA, Dec of interest, the fraction of the image bounds to 
        consider valid (optional; default 1.0 which is to consider the entire 
        image) a bool indicating whether to be verbose (optional; default 
        True), and a bool indicating whether to check all files or just return 
        the first file which contains the given WCS (optional; default True --> 
        return all files spanning the given coords)
        
        Checks if the input coordinates are spanned by any of the data. 
        
        Output: a list of these files' locations
        """
        l = self.loc
        good_files = []
        
        # set x and y limits of each detector
        if "WIRCam" in self.instrument: 
            x_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
            y_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
        else:
            x_lim = [32.0+(2080.0-32.0)*((1.0-frac)/2), 2080.0*((1.0+frac)/2)]
            y_lim = [0.0+4612.0*((1.0-frac)/2), 4612.0*((1.0+frac)/2)]
        
        if self.nextend == 0: # if images already divided up by detector/CCD
            for f in self.files: 
                hdr = fits.getheader(l+"/"+f)
                w = wcs.WCS(hdr)
                naxis = int(hdr["NAXIS"])

                if naxis == 3: # if a cube
                    pix_coords = np.array(w.all_world2pix(ra,dec,1,1))
                else : # if just one image 
                    pix_coords = np.array(w.all_world2pix(ra,dec,1))
            
                # check if located in detector  
                if (x_lim[0]<pix_coords[0]<x_lim[1]) and (
                    y_lim[0]<pix_coords[1]<y_lim[1]):
                    good_files.append(l+"/"+f)
                    if verbose:
                        print(l+"/"+f)
                    if not(checkall): # if we only want the first file
                        return good_files
                    
        else: # if multiple detectors/CCDs
            for f in self.files: 
                n = RawData.locate_WCS(self, ra, dec)
                if n:
                    good_files.append(l+"/"+f)
                    if verbose:
                        print(l+"/"+f+" [detector "+str(n)+"]")
                    if not(checkall): # if we only want the first file
                        return good_files
                        
                        
        if len(good_files) != 0:
            return good_files
        else:
            return
    
    def locate_WCS(self, ra, dec, frac=1.0):
        """
        Input: The RA and Dec of interest and the fraction of the image bounds 
        to consider valid (optional; default 1.0 which is to consider the 
        entire image)
        
        Examines all of the detectors of some multi-detector image (a multi-
        extension fits file) and returns the number of the detector where the 
        coordinates are located. 
        
        Output: The number of extension which contains the detector of interest
        """
        
        if self.nextend == 0:
            print("Cannot call locate_WCS() on a file without extensions.")
            return
        
        l = self.loc
        # assume the instruments do not drift and WCS does not significantly 
        # change from one image to another 
        testfile = self.files[0]
        hdu_list_test = fits.open(l+"/"+testfile)
        
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
                x_lim = [32.0+(2080.0-32.0)*((1.0-frac)/2), 
                         2080.0*((1.0+frac)/2)]
                y_lim = [0.0+4612.0*((1.0-frac)/2), 4612.0*((1.0+frac)/2)]
                
                pix_coords = np.array(w.all_world2pix(ra,dec,1))
            
            # for debugging
            #print("n="+str(n+1)+"\t [%.2f, %.2f]"%(
            #        pix_coords[0],pix_coords[1]))
            
            # check if located in detector n 
            if (x_lim[0]<pix_coords[0]<x_lim[1]) and (
                    y_lim[0]<pix_coords[1]<y_lim[1]):
                #print("Source is located in extension "+str(n+1))
                return n+1  
            
            return # nothing found 

            
    def __get_extension(self, fits_file, n_ext):
        """
        Input: A single fits file and the extension number of interest. 
        Extracts the header and image data of the given extension. Helper fn. 
        Output: The header and image data 
        """
        l = self.loc
        
        # WIRCam: 1 to 4
        # MegaPrime: 1 to 36 pre-2015A, 1 to 40 post-2015A 
        
        new_hdu_list = fits.open(l+"/"+fits_file)
        new_hdu = new_hdu_list[n_ext] # compressed
        
        new_hdr = new_hdu.header
        new_data = new_hdu.data
        extension = fits.PrimaryHDU(data=new_data, header=new_hdr)
        
        return extension  

    
    def write_extensions(self, n_ext):
        """
        Input: The extension number of interest
        Gets the header and image data for the given extension and writes them
        to a new .fits file. Does so for all raw data files. 
        Stores them in a new subdirectory. Used to extract image data for one 
        of many CCDs/detectors on either MegaPrime/WIRCam. 
        Output: None
        
        * For a given dataset, once locate_WCS() has been used to find the ext
        containing the WCS of interest, run this function once to extract that
        specific extension. Can then use this newly made folder for stacking.
        * If the extensions are themselves cubes (sometimes the case for 
        WIRCam), see combine_WIRCam() or divide_WIRCam().
        """
        l = self.loc
        
        # exten_dir encodes the detector number, instrument, and date
        exten_dir = os.path.abspath(l+"/..")+"/det"+str(n_ext)+"_"
        exten_dir += self.instrument+"_"+self.date
        run(['mkdir','-p',exten_dir]) # make exten_dir
        
        for f in self.files: 
            exten = RawData.__get_extension(self, f, n_ext)
            new_f = f.replace(".fits.fz","_det"+str(n_ext)+".fits")
            exten.writeto(exten_dir+"/"+new_f, overwrite=True, 
                          output_verify="ignore") # write them
            
        print("Extracted headers/images for detector "+str(n_ext)+
              " of "+self.instrument+" on "+self.date)
        print("Written to new .fits files in "+exten_dir)
        
        
    def write_extensions_all(self):
        """
        Input: None
        Gets the header and image data for ALL extensions of a multi-extension
        fits file and writes each extension to a new .fits file. Useful for 
        MegaPrime data where the scope moves a lot and the detectors are all 
        calibrated to the same ADU level. 
        Output: None 
        """
        l = self.loc
        
        # all_exten_dir encodes the instrument and date
        all_exten_dir = os.path.abspath(l+"/..")+"/dets_ALL_"+self.instrument
        all_exten_dir += "_"+self.date
        run(['mkdir','-p',all_exten_dir]) # make all_exten_dir
        
        for f in self.files:
            for n in range(self.nextend):
                exten = RawData.__get_extension(self, f, n+1)

                new_f = f.replace(".fits.fz","_det"+str(n+1)+".fits")
                exten.writeto(all_exten_dir+"/"+new_f, overwrite=True, 
                              output_verify="ignore") # write them
            
        print("Extracted headers/images for all detectors of "+self.instrument+
              " on "+self.date)
        print("Written to new .fits files in "+all_exten_dir)
    
    
    def write_extensions_by_WCS(self, ra, dec, frac=1.0):
        """
        Input: The RA and Dec of interest and the fraction of the image bounds 
        to consider valid (optional; default 1.0 which is to consider the 
        entire image)
        
        For a directory full of multi-extension fits files, gets the extension
        which contains the input RA, Dec and writes it to a new file 
        
        Output: None
        """
        l = self.loc
        
        # wcs_exten_dir encodes the wcs of interest, the instrument, and date
        wcs_exten_dir = os.path.abspath(l+"/..")
        wcs_exten_dir += "/dets_RA%.3f_DEC%.3f_"%(ra,dec)
        wcs_exten_dir += self.instrument+"_"+self.date
        run(['mkdir','-p',wcs_exten_dir]) # make wcs_exten_dir
        
        # set x and y limits of each detector
        if "WIRCam" in self.instrument: 
            x_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
            y_lim = [0.0+2048.0*((1.0-frac)/2), 2048.0*((1.0+frac)/2)]
        else:
            x_lim = [32.0+(2080.0-32.0)*((1.0-frac)/2), 2080.0*((1.0+frac)/2)]
            y_lim = [0.0+4612.0*((1.0-frac)/2), 4612.0*((1.0+frac)/2)]
        
        for f in self.files:
            for n in range(self.nextend):
                exten = RawData.__get_extension(self, f, n+1)
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
                    new_f = f.replace(".fits.fz","_det"+str(n+1)+".fits")
                    exten.writeto(wcs_exten_dir+"/"+new_f, overwrite=True, 
                                  output_verify="ignore") # write them
                    break # exit this for loop
                    
        print("Extracted headers/images for detectors which contain "+
              "RA %.3f, Dec %.3f"%(ra,dec)+" for data from "+
              self.instrument+" on "+self.date)
        print("Written to new .fits files in "+wcs_exten_dir)
        
        
    def write_source(self, name, ra, dec, frac=1.0):
        """
        Input: the name, RA, and Dec for some interesting source (such as a 
        transient) and the fraction of the image bounds to consider valid 
        (optional; default 1.0 which is to consider the entire image)
        
        Parses all raw data and copies any images which contain the input RA, 
        Dec to a new directory with the name of the source.
        
        Output: None
        """
        l = self.loc
        source_files = RawData.WCS_check(self, ra, dec, frac)
        if source_files:  
            run("mkdir -p "+l+"/"+name, shell=True)
            for f in source_files:
                run("cp "+f+" "+l+"/"+name, shell=True)
        else:
            print("\nNone of the raw data contains the input RA, Dec.")

###############################################################################    
    #### COMBINING/DIVIDING CUBES ####
            
    def __combine_cube(self, fits_file):
        """
        Input: the name of a fits file containing a cube 
        For a file composed of multiple 2D image data arrays (i.e. a 
        cube), combines the image data into one single 2D array. Only 
        needed for WIRCam data, which sometimes contains a cube for each 
        of the 4 detectors.
        Output: The header and image data of the new, combined image with 
        correctly updated exposure times, dimensions, and observation dates 
        """
        l = self.loc
        f = fits.open(l+"/"+fits_file)[0]
        
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
            t_isot_slice = Time(f.header["SLDATE0"+str(n+1)], format="isot", 
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
        """
        Input: None
        For a directory full of WIRCam images, if the images are cubes, 
        *combines* the multiple 2D arrays into single arrays.
        Output: None
        
        * Use this function once to take a folder full of cubes and turn them 
        into single-frame fits files. Can NOT be called on cubes. Only used for 
        WIRCam.
        """
        
        l = self.loc
        topfile = re.sub(".*/", "", l) # for a file /a/b/c, extract the "c"
        if not("WIRCam" in self.instrument):
            print("Cannot call combine_WIRCam() except on WIRCam cubes.")
            return
        
        # combin_dir encodes the detector number, instrument, and date
        combin_dir = os.path.abspath(l+"/..")+"/combined_"+topfile
        run(['mkdir','-p',combin_dir]) # make combin_dir
        
        for f in self.files: 
            if len((fits.getdata(l+"/"+f)).shape) > 2: # if a cube 
                combin = RawData.__combine_cube(self, f)
                new_f = f.replace(".fits","_combined.fits")
                combin.writeto(combin_dir+"/"+new_f, overwrite=True, 
                              output_verify="ignore") # write them
            else: # if not, just copy it over without changing filename
                run("cp -p "+l+"/"+f+" "+combin_dir)

            
    def __divide_cube(self, fits_file):
        """
        Input: the name of a fits file containing a cube 
        For an file composed of multiple 2D image data arrays (i.e. a 
        cube), divides the image data into separate files. Only 
        needed for WIRCam data, which sometimes contains a cube for each 
        of the 4 detectors. 
        Output: The header and image data of new divided images
        """
        l = self.loc
        f = fits.open(l+"/"+fits_file)[0]
        
        n_images = len(f.data) # no. of images in the cube 
        new_header = f.header
        new_header["NAXIS"] = 2 # no longer a cube 

        divisions = []
        for n in range(0, n_images):  
            temp_image = f.data[n] # a single slice 
            temp_hdr = new_header
            # new header: slice ID in cube it came from (01, 02, 03...)
            temp_hdr["SLICEID"] = ("0"+str(n+1), "Slice ID in original cube")
            divisions.append(fits.PrimaryHDU(data=temp_image, header=temp_hdr))
        
        return divisions # list of PrimaryHDU objects 
    
    
    def divide_WIRCam(self):
        """
        Input: None
        For a directory full of WIRCam images, if the images are cubes, 
        *divides* the multiple 2D arrays into individual arrays.
        Output: None
        
        * Use this function once to take a folder full of cubes and turn them 
        into single-frame fits files. Can NOT be called on cubes. Only used for 
        WIRCam.
        """
        
        l = self.loc
        topfile = re.sub(".*/", "", l) # for a file /a/b/c, extract the "c"
        if not("WIRCam" in self.instrument):
            print("Cannot call divide_WIRCam() except on WIRCam cubes. "+
                  "Exiting.")
            return
        
        # div_dir encodes the detector number, instrument, date
        div_dir = os.path.abspath(l+"/..")+"/divided_"+topfile
        run(['mkdir','-p',div_dir]) # make div_dir
        
        for f in self.files: 
            if len((fits.getdata(l+"/"+f)).shape) > 2: # if a cube 
                divs = RawData.__divide_cube(self, f)
                for div in divs:
                    temp_header = div.header
                    sliceid = temp_header["SLICEID"]
                    new_f = f.replace(".fits","_divided_"+sliceid+".fits")
                    div.writeto(div_dir+"/"+new_f, overwrite=True, 
                                output_verify="ignore") # write them
            else: # if not, just copy it over without changing filename
                  # but assign a SLICEID
                run("cp -p "+l+"/"+f+" "+div_dir, shell=True)
                temp = fits.open(div_dir+"/"+f, mode="update")
                temp[0].header["SLICEID"] = "01"
                temp.close()

###############################################################################
    #### CROP IMAGES BASED ON PIXEL FRACTIONS/WCS ####
        
    def __get_crop(self, fits_file, frac_hori=[0,1], frac_vert=[0,1]):
        """
        Input: a single fits file, the horizontal fraction of the fits file's 
        image to crop (default [0,1], which does not crop), and the vertical 
        fraction (default [0,1])
        e.g. __get_crop("foo.fits", [0.5,1], [0,0.5]) would crop the right 
        half of the image and and the bottom half of the image
        Output: a new fits HDU containing the header and the cropped image
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
        """
        Input: horizontal fraction of the fits file's image to crop (default 
        [0,1], which does not crop), and the vertical fraction (default [0,1])
        For a directory full of WIRCam/MegaPrime images (should NOT be cubes), 
        crops the images based on the input x-axis and y-axis boundaries and 
        writes them all to a new directory.
        Output: None
        """
        
        l = self.loc
        topfile = re.sub(".*/", "", l) # for a file /a/b/c, extract the "c"
        
        # crop_dir encodes the detector number, instrument, date
        crop_dir = os.path.abspath(l+"/..")+"/cropped_"+topfile
        run(['mkdir','-p',crop_dir]) # make crop_dir
        
        for f in self.files:  
            cropped_hdu = RawData.__get_crop(self, l+"/"+f, frac_hori, 
                                             frac_vert)
            new_f = f.replace(".fits","_cropped.fits")
            cropped_hdu.writeto(crop_dir+"/"+new_f, overwrite=True, 
                          output_verify="ignore") # write them
          
            
    def crop_images_wcs(self, ra, dec, size):
        """
        Input: a right ascension, declination (in decimal degrees), and the 
        size of a box (in pixels) to crop 
        For a directory full of WIRCam/MegaPrime images (should NOT be cubes), 
        crops the images based on the input WCS coordinates, creating a box 
        centred on these coordinates and writing the new files to a new 
        directory. 
        * If the given box extends beyond the bounds of the image, the box will
        be truncated at these bounds. 
        Output: None
        """
        l = self.loc
        topfile = re.sub(".*/", "", l) # for a file /a/b/c, extract the "c"
        
        # crop_dir encodes the detector number, instrument, date
        crop_dir = os.path.abspath(l+"/..")+"/cropped_"+topfile
        run(['mkdir','-p',crop_dir]) # make crop_dir
        
        crop_counter = 0
        for f in self.files:
            hdr = fits.getheader(l+"/"+f)
            img = fits.getdata(l+"/"+f)
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
            cropped_hdu = RawData.__get_crop(self, l+"/"+f, frac_hori, 
                                             frac_vert)
            new_f = f.replace(".fits","_cropped.fits")
            cropped_hdu.writeto(crop_dir+"/"+new_f, overwrite=True, 
                          output_verify="ignore") # write them
            
        print(str(crop_counter)+"/"+str(len(self.files))+
              " images could be cropped.\n")           
        
###############################################################################
    #### STACKING AND STACK PREPARATION/EXTRACTION ####

    def make_badpix_masks(self):
        """
        Input: None
        For each file, builds and points to the corresponding bad pixel mask 
        for later stacking.
        Output: None
        """
        l = self.loc
        topfile = re.sub(".*/", "", l) # for a file /a/b/c, extract the "c"
        
        # bp_dir contains the bad pixel masks
        self.bp_dir = os.path.abspath(l+"/..")+"/badpixels_"+topfile
        run(['mkdir','-p',self.bp_dir]) # make bp_dir
        
        for f in self.files:
            # build the bad pixel mask
            image_data = fits.getdata(l+"/"+f)
            image_hdr = fits.getheader(l+"/"+f)
            
            bp_mask = (image_data != 0)
            bp_mask = ma.masked_where(bp_mask, image_data)
            bp_mask.fill_value = 1.0
            bp_img = bp_mask.filled() # 1 at good pix, 0 at bad pix 
            
            bp = fits.PrimaryHDU(bp_img, image_hdr)
            bp.writeto(self.bp_dir+"/"+f.replace("."+self.fmt, "_bp_mask.fits"), 
                       overwrite=True, output_verify="ignore")
            # set the bad pixel mask
            hdu = fits.open(l+"/"+f, mode="update") 
            hdu[0].header["BPM"] =  f.replace("."+self.fmt, "_bp_mask.fits")
            hdu.close(output_verify="ignore")
            

    def __make_stack_directory(self):
        """
        Input: None
        Makes a directory to store stacked image(s).
        Output: None
        """
        sd = self.stack_dir
        if sd:
            run(['mkdir','-p',sd]) # create it
        else: # if not yet defined
            print("\nPlease set a stack directory using set_stack_dir() "+
                  "before attempting to stack images. Exiting.\n")
            return 
        # get rid of calibration file if it exists:
        run('rm -rf '+sd+'/calibration', shell=True) 
        run('rm -rf '+sd+'/*.fits', shell=True)
        run('rm -rf '+sd+'/*.flt', shell=True)
        
        # get rid of previous .fits, .flt and .txt files
        # need to use glob to use wildcards (*):
        textfiles = glob.glob(sd+"/*.txt")
        fitsfiles = glob.glob(sd+"/*.fits")
        fitsfiles = glob.glob(sd+"/*.flt")
        allfiles = textfiles + fitsfiles
        for a in allfiles:
            os.remove(a)
            
        
        
        run(['chmod','777',sd]) # give full permissions to the directory
        
        # needed for iraf[?]
        run('cp -f ~/iraf/login.cl '+sd, shell=True)
        run('mkdir -p '+sd+'/uparm', shell=True)
        run('cp -f ~/iraf/uparm/* '+sd+'/uparm', shell=True) 
        
        for fil in self.filters:
            np.savetxt(sd+'/'+fil+'_list.txt', self.filters_dict[fil], 
                       fmt="%s")

        # location of bad pixel masks
        l = self.loc
        topfile = re.sub(".*/", "", l)
        self.bp_dir = os.path.abspath(l+"/..")+"/badpixels_"+topfile

        # copy fits files
        run("cp -f "+self.loc+"/*."+self.fmt+" "+self.stack_dir, shell=True)
        run("cp -f "+self.bp_dir+"/*.fits "+self.stack_dir, shell=True)
        run('chmod 777 '+sd+'/*', shell=True) # give full perms
        
        return True
        

    def make_stacks(self, *filters):
        """
        Input: An optional valid filters argument. Defaults to all of the 
        filters for which there is raw data. Can specify a subset of these 
        filters if you don't want to process all of them. 
        Uses PyRaf to coadd all of the raw data into stacks based on filter.
        Output: None
        """
        
        # check if need to make bad pixel masks 
        hdr = fits.getheader(self.loc+"/"+self.files[0])
        try:
            temp = hdr["BPM"]
            del temp
        except KeyError:
            RawData.make_badpix_masks(self) # make bad pixel masks    
            
        ret = RawData.__make_stack_directory(self) # make stack directory
        
        if not(ret): # if stack directory not successfully made
            return 
        
        script_dir = os.getcwd() # script directory
        run('cp stack.py '+self.stack_dir, shell=True)
        os.chdir(self.stack_dir) # move to stack directory 
            
        # command line arguments:
        cmdargs = self.instrument+" "+self.date 
        
         # if no argument given
        valid_filters = " "
        if not(filters): # if no arg given
            filters = self.filters
            for fil in self.filters:
                valid_filters += fil+" "
        else:
            for fil in filters:
                valid_filters += fil+" "
        
        # add the valid filters, remove the space at the end:
        cmdargs += valid_filters[:-1]
        
        # run the script stack.py
        run("bash -c 'source activate iraf27 && python2 stack.py "+cmdargs+
            " && source deactivate'", shell=True)
        
        # update exp. time of stack to total exposure time 
        for fil in filters: 
            stack = fil+"_stack_"+self.date+".fits"
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
        """
        Input: a filter of choice
        Output: a Stack object
        """
        return Stack(self.loc, self.stack_dir, self.qso_grade_limit, 
                     self.fmt, filt)
        
###############################################################################
    #### STACKS, MAKING IMAGES & ASTROMETRY ####

class Stack(RawData):
    def __init__(self, location, stack_directory, qso_grade_limit, fmt, filt):
        super(Stack, self).__init__(location, stack_directory, 
             qso_grade_limit, fmt)
                
        # only one filter is spanned by a given stack
        self.files = self.filters_dict[filt]
        for fil in self.filters:
            if filt != fil:
                delattr(self, fil)

        # stack file specifics
        self.stack_name = filt+"_stack_"+self.date+".fits" # name
        self.stack_size = len(self.files) # size
        
        # check if stack file is present before continuing and make it if not
        if not(self.stack_name in os.listdir(self.stack_dir)):
            print("\nSince the stacked image is not yet present, it will be "+
                  "produced now.")
            Stack.make_stacks(self, filt)
            
        #if not(Stack.stack_made): # if stack was not successfuly made 
        #    exit
            
        delattr(self, "filters") # don't need a list anymore 
        self.filter = filt # just one filter 
        delattr(self,'filters_dict') # filters dict no longer needed

        # image data and header for general use
        self.image_data = fits.getdata(self.stack_dir+"/"+self.stack_name)
        self.image_header = fits.getheader(self.stack_dir+"/"+self.stack_name)
        
        # total exposure time (in seconds)
        self.exptime = fits.getheader(
                self.stack_dir+"/"+self.stack_name)['EXPTIME']
            
        # pixel scale of image (same scale in x and y)
        self.pixscale = self.image_header['PIXSCAL1']
        
        # dimensions of image (pixels)
        self.y_size, self.x_size  = self.image_data.shape 
        
        # make a directory for astrometric/photometric calibration
        # copy the stack file accordingly 
        self.calib_dir = self.stack_dir+"/calibration"
        run('mkdir -p '+self.calib_dir, shell=True)
        run('cp '+self.stack_dir+'/'+self.stack_name+' '+self.calib_dir, 
            shell=True)
        
        # initialize time for all files and the overall stack
        Stack.__time_init(self)            
            
        # bools for later
        self.astrometric_calib = False
        self.psf_fit = False
        self.photometric_calib = False
        self.image_error_computed = False
        self.aperture_fit = False
        
        
    def __time_init(self):
        """
        Input: None
        Initializes a list of observation times for each file in the stack 
        and a time for the entire stack (in Modified Julian Date (MJD)).
        Output: None
        """
        l = self.loc
        self.times = []
        for f in self.files:
            hdr = fits.getheader(l+"/"+f)
            
            if "WIRCam" in self.instrument:
                # check if the file comes from a divided cube 
                sliceid = hdr["SLICEID"]
                if not(sliceid): # does not come from a division
                    sliceid = "01" 
                sldate = hdr["SLDATE"+sliceid] # SLDATE for correct slice
                t_isot = Time(sldate, format='isot', scale='utc')
            else:
                date_isot = hdr["DATE"] # full ISOT time
                t_isot = Time(date_isot, format='isot', scale='utc')
                
            t_MJD = t_isot.mjd # convert ISOT in UTC to MJD
            self.times.append(t_MJD)
        
        self.stack_time = np.mean(self.times) # stack time in MJD 
        
        
    def make_image(self, clean=False, border=False, sources=False, 
                   ra=None, dec=None, scale=None, output=None):
        """
        Input: boolean indicating whether plot the background subtracted data 
        as found by astrometry (optional; default False), whether to show the 
        region where sources are considered valid for photometry (optional; 
        default False), whether to show detected sources (optional; default 
        False), right ascension and declination to mark a specific location 
        (optional; default None), a scale to apply to the image (optional; 
        default None=linear; options are "log" and "asinh") and a name for the 
        image file (optional; default None) 
        Produces and saves an an image of the stacked file.
        Output: None
        """
        # image data
        if clean and self.astrometric_calib:
            image_data = self.image_data_clean
        elif clean and not(self.astrometric_calib):
            print("To obtain a background-subtracted, smoothed image, use "+
                  "the astrometry() function first. Exiting.")
        else:
            image_data = self.image_data
            
        image_header = self.image_header # image header       
        w = wcs.WCS(image_header)
        
        if not(output): # if none given, defaults to the following 
            output = self.filter+"_"+self.instrument+"_"+self.date+"."
            output += self.plot_ext
            
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
            cb.set_label(label=r"$\arcsinh{(ADU)}$", fontsize=16)
            
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        plt.savefig(output, bbox_inches="tight")
        plt.close()
            
            
    def astrometry(self, downsample=None):
        """
        Input: a downsampling factor (optional; default None)
        Performs source extraction using astrometry.net. Refines the 
        astrometric solution and THEN does extraction to create a list of 
        sources
        Output: 
        """
        start = timer() # timing the function
        
        script_dir = os.getcwd() # script directory
        os.chdir(self.calib_dir) # move to calibration directory
        
        # name for cleaned-up file to be produced
        self.stack_name_clean = self.stack_name.replace(".fits", "_clean.fits")
        # name for mask to be produced
        self.mask_name = self.stack_name.replace(".fits", "_mask.fits")
        
        # don't check WCS headers, don't plot, and input a fits image:
        solve_options = "--no-verify --overwrite --no-plot --fits-image" 
        # build a new fits file: 
        solve_options += " --new-fits "+self.stack_name.replace(
                ".fits", "_updated.fits")
        if downsample: # if a downsampling fraction is given
            solve_options += " --downsample "+str(downsample)
            
        # options to speed up astrometry: pixscale and rough RA, Dec
        pixmin = self.pixscale-0.001
        pixmax = self.pixscale+0.001
        solve_options += " --scale-low "+str(pixmin)
        solve_options += " --scale-high "+str(pixmax)
        solve_options += " --scale-units app"
        
        image_header = self.image_header
        cent = [i//2 for i in self.image_data.shape]
        centy, centx = cent
        w = wcs.WCS(image_header)
        ra, dec = w.all_pix2world(centx, centy, 1) 
        radius = 1.0/6.0 # look in a radius of 10 arcmin
        solve_options += " --ra "+str(ra)+" --dec "+str(dec)
        solve_options += " --radius "+str(radius)
        
        # stop when this file is produced:
        solve_options += " --cancel "+self.stack_name.replace(
                ".fits", "_updated.fits")
        
        # solve the field: 
        run("solve-field "+solve_options+" "+self.stack_name, shell=True)
        # update the name of the stack file:
        self.stack_name = self.stack_name.replace(".fits", "_updated.fits")
        run("find . -type f -not -name '*updat*' -print0 | xargs -0 rm --",
            shell=True) # remove all files not in format *updat*
        
        # check if any files are present (i.e. if astrometry solved) and exit 
        # if not 
        updats = os.listdir()
        if len(updats) == 0:
            os.chdir(script_dir)
            print("The WCS solution could not be obtained. This likely means"+
                  " that the required index files are missing or data is of"+
                  " poor quality. Exiting.")
            return

        # print confirmation 
        print("The WCS solution has been updated for the stack image.")
        print("Stack name is now "+self.stack_name )

        # source detection: make list of sources, background-subtracted image,
        # and a mask 
        options = "-O -U "+self.stack_name_clean+" -M "+self.mask_name
        if downsample: # if a downsampling fraction is given
            options += " -d "+str(downsample)
        run("image2xy "+options+" "+self.stack_name, shell=True) 
        
        # store the output in attributes 
        # source data for general use
        self.xy_name = self.stack_name.replace(".fits", ".xy.fits")
        self.xy_data = fits.getdata(self.xy_name) 
        
        # smoothed, background-subtracted, "clean" image data
        self.image_data_clean = fits.getdata(self.stack_name_clean)     
        # store updated WCS solution in header
        self.image_header = fits.getheader(self.stack_name)
        # the mask 
        self.mask_data = fits.getdata(self.mask_name)
        
        os.chdir(script_dir) # return to script directory 
        self.astrometric_calib = True 
        
        end = timer()
        time_elaps = end-start 
        print("Time required for astrometric calibration: %.2f s\n"%
              time_elaps)
    
###############################################################################
    #### PSF PHOTOMETRY ####

    def __fit_PSF(self, plot_ePSF=True, plot_residuals=False, source_lim=None):
        """
        Input: bool indicating whether to plot the empirically determined 
        effective Point-Spread Function (ePSF) (optional; default True)
        whether to plot the residuals of the iterative PSF fitting (optional;
        default False) and a limit on the no. of sources to fit with the ePSF 
        (optional; default no limit)
        
        Uses the cleaned (smoothed, background-subtracted) image to obtain 
        the ePSF and fits this function to all of the sources previously
        detected by astrometry. Builds a table containing the instrumental
        magnitudes and corresponding uncertainties to be used in obtaining the 
        zero point for PSF calibration.
        
        Output: None
        """
        
        if not(self.astrometric_calib):
            print("\nPSF photometry cannot be obtained because astrometric "+
                  "calibration has not yet been performed. Exiting.")
            return
        
        from astropy.modeling.fitting import LevMarLSQFitter
        from photutils.psf import (BasicPSFPhotometry, DAOGroup)
        from astropy.nddata import NDData
        from photutils.psf import extract_stars
        from photutils import EPSFBuilder
            
        image_data = self.image_data_clean # the CLEANED image data 
        sources_data = self.xy_data # sources
        image_header = self.image_header # image header
        
        ### SETUP
        # get source WCS coords
        x = np.array(sources_data['X'])
        y = np.array(sources_data['Y'])
        w = wcs.WCS(image_header)
        wcs_coords = np.array(w.all_pix2world(x,y,1))
        ra = Column(data=wcs_coords[0], name='ra')
        dec = Column(data=wcs_coords[1], name='dec')
        
        sources = Table() # build a table 
        sources['x_mean'] = sources_data['X'] # for BasicPSFPhotometry
        sources['y_mean'] = sources_data['Y']
        sources['x'] = sources_data['X'] # for EPSFBuilder 
        sources['y'] = sources_data['Y']
        sources.add_column(ra)
        sources.add_column(dec)
        sources['flux'] = sources_data['FLUX']  # already bkg-subtracted 
 
        # mask out edge sources:
        # a bounding circle for WIRCam, rectangle for MegaPrime
        if "WIRCam" in self.instrument:
            rad_limit = self.x_size/2.0
            dist_to_center = np.sqrt((sources['x_mean']-self.x_size/2.0)**2 + 
                             (sources['y_mean']-self.y_size/2.0)**2)
            mask = dist_to_center <= rad_limit
            sources = sources[mask]
        else: 
            x_lims = [int(0.05*self.x_size), int(0.95*self.x_size)] 
            y_lims = [int(0.05*self.y_size), int(0.95*self.y_size)]
            mask = (sources['x_mean']>x_lims[0]) & (
                    sources['x_mean']<x_lims[1]) & (
                    sources['y_mean']>y_lims[0]) & (
                    sources['y_mean']<y_lims[1])
            sources = sources[mask]
            

        ### EMPIRICALLY DETERMINED ePSF
        start = timer() # timing ePSF building time
        
        nddata = NDData(image_data)# NDData object
        stars = extract_stars(nddata, sources, size=25) # extract stars
        
        # use only the stars with fluxes between two percentiles
        stars_tab = Table() # temporary table 
        stars_col = Column(data=range(len(stars.all_stars)), name="stars")
        stars_tab["stars"] = stars_col # column of indices of each star
        fluxes = [s.flux for s in stars]
        fluxes_col = Column(data=fluxes, name="flux")
        stars_tab["flux"] = fluxes_col # column of fluxes
        
        # get percentiles
        per_low = np.percentile(fluxes, 80) # 80th percentile flux 
        per_high = np.percentile(fluxes, 90) # 90th percentile flux
        mask = (stars_tab["flux"] >= per_low) & (stars_tab["flux"] <= per_high)
        stars_tab = stars_tab[mask] # include only stars between these fluxes
        idx_stars = (stars_tab["stars"]).data # indices of these stars
        self.nstars_epsf = len(idx_stars) # no. of stars used in ePSF building
        
        # update stars object and then build the ePSF
        # have to manually update all_stars AND _data attributes
        stars.all_stars = [stars[i] for i in idx_stars]
        stars._data = stars.all_stars
        epsf_builder = EPSFBuilder(oversampling=1, maxiters=10, # build it
                                   progress_bar=False)
        epsf, fitted_stars = epsf_builder(stars)
        
        # compute 90% radius of the ePSF to determine appropriate aperture size
        # for aperture photometry 
        epsf_data = epsf.data
        y, x = np.indices(epsf_data.shape)
        x_0 = epsf.data.shape[1]/2.0
        y_0 = epsf.data.shape[0]/2.0
        r = np.sqrt((x-x_0)**2 + (y-y_0)**2) # radial distances from source
        r = r.astype(np.int) # round to ints 
        
        # bin the data, obtain and normalize the radial profile 
        tbin = np.bincount(r.ravel(), epsf_data.ravel()) 
        norm = np.bincount(r.ravel())  
        profile = tbin/norm 
        
        # find radius at 10% of max 
        limit = np.min(profile[0:20]) 
        limit += 0.1*(np.max(profile[0:20])-np.min(profile[0:20]))
        for i in range(len(profile)):
            if profile[i] >= limit:
                continue
            else: # if below the 10% of max 
                self.epsf_radius = i # radius in pixels 
                break
        print("\nePSF 90% radius: "+str(self.epsf_radius)+" pix")
        
        end = timer() # timing 
        time_elaps = end-start
        print("Time required for ePSF building: %.2f s\n"%time_elaps)
        
        psf_model = epsf # set the model
        psf_model.x_0.fixed = True # fix centroids (known beforehand) 
        psf_model.y_0.fixed = True
        
        # initial guesses for centroids, fluxes
        pos = Table(names=['x_0', 'y_0','flux_0'], data=[sources['x_mean'],
                   sources['y_mean'], sources['flux']]) 
    
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
                source_rows = random.choices(sources, k=source_lim)
                sources = Table(names=['x_mean', 'y_mean', 'x', 'y', 'ra', 
                                       'dec', 'flux'], rows=source_rows)
                pos = Table(names=['x_0', 'y_0','flux_0'], 
                            data=[sources['x_mean'], sources['y_mean'], 
                                  sources['flux']])
                
                
            except IndexError:
                print("The input source limit exceeds the number of sources"+
                      " detected by astrometry, so no limit is imposed.\n")
        
        photometry = BasicPSFPhotometry(group_maker=daogroup,
                                bkg_estimator=None, # bg subtract already done
                                psf_model=psf_model,
                                fitter=fitter_tool,
                                fitshape=(11,11))
        
        result_tab = photometry(image=image_data, init_guesses=pos) # results
        residual_image = photometry.get_residual_image() # residuals of PSF fit
        
        end = timer() # timing 
        time_elaps = end - start
        print("Time required to fit ePSF to all sources: %.2f s\n"%time_elaps)
        
        # include previous WCS results
        result_tab.add_column(sources['ra'])
        result_tab.add_column(sources['dec'])
        
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
            plt.title("Effective Point-Spread Function (1 pixel = "+
                                                        str(self.pixscale)+
                                                        '")', fontsize=16)
            plt.colorbar(orientation="vertical", fraction=0.046, pad=0.08)
            plt.rc("xtick",labelsize=16) # not working?
            plt.rc("ytick",labelsize=16)
            plt.savefig(self.filter+"_"+self.instrument+"_"+self.date+
                        "_ePSF."+self.plot_ext, bbox_inches="tight")
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
            plt.savefig(self.filter+"_"+self.instrument+"_"+self.date+
                        "_PSF_resid."+self.plot_ext, bbox_inches="tight")
            plt.close()
        
        # save psf_sources as an attribute
        self.psf_sources = psf_sources
        
        # update bool
        self.psf_fit = True      
        
        
    def __zero_point(self, plot_corr=True, plot_source_offsets=True, 
                     plot_field_offsets=False, gaussian_blur_sigma=30.0, 
                     cat_num=None):
        """
        Input: a bool indicating whether or not to plot the correlation with 
        linear fit (optional; default True), whether to plot the offsets in 
        RA and Dec of each catalog-matched source (optional; default True), 
        whether to show the overall offsets as an image with a Gaussian blur 
        to visualize large-scale structure (optional; default False), the 
        sigma to apply to the Gaussian filter (optional; default 30.0), and 
        a Vizier catalog number to choose which catalog to cross-match 
        (optional; defaults are PanStarrs 1, SDSS DR12, and 2MASS for relevant 
        filters)
        
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
            self.ref_cat = cat_num
            self.ref_cat_name = cat_num
        else:  
            if self.filter in ['g','r','i','z','Y']:
                zp_filter = (self.filter).lower() # lowercase needed for PS1
                self.ref_cat = "II/349/ps1" # PanStarrs 1
                self.ref_cat_name = "PS1" 
            elif self.filter == 'u':
                zp_filter = 'u' # closest option right now 
                self.ref_cat = "V/147" 
                self.ref_cat_name = "SDSS DR12"
            else: 
                zp_filter = self.filter[0] # Ks must be K for 2MASS 
                self.ref_cat = "II/246/out" # 2MASS
                self.ref_cat_name = "2MASS"
            
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
         
        # actual querying 
        # internet connection needed 
        print("\nQuerying Vizier %s (%s) "%(self.ref_cat, self.ref_cat_name)+
              "around RA %.4f, Dec %.4f "%(ra_centre, dec_centre)+
              "with a radius of %.4f arcmin"%radius)
        
        v = Vizier(columns=["*"], column_filters={
                zp_filter+"mag":str(minmag)+".."+str(maxmag),
                "e_"+zp_filter+"mag":"<"+str(max_emag),
                "Nd":">"+str(nd)}, row_limit=-1) # no row limit 
        Q = v.query_region(SkyCoord(ra=ra_centre, dec=dec_centre, 
                            unit = (u.deg, u.deg)), radius = str(radius)+'m', 
                            catalog=self.ref_cat, cache=False)
    
        if len(Q) == 0: # if no matches
            print("\nNo matches were found in the "+self.ref_cat_name+
                  " catalog. The requested region may be in an unobserved"+
                  " region of this catalog. Exiting.")
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
                source_coords, 2*self.pixscale*u.arcsec)
        
        self.nmatches = len(idx_image) # store number of matches 
        self.sep_mean = np.mean(d2d.value*3600.0) # store mean separation in "
        print('\nFound %d sources in %s within 2.0 pix of'%(self.nmatches, 
                                                            self.ref_cat_name)+
              ' sources detected by astrometry, with average separation '+
              '%.3f" '%self.sep_mean)
        
        # get coords for sources which were matched
        source_matches = source_coords[idx_image]
        cat_matches = cat_source_coords[idx_cat]
        source_matches_ra = [i.ra.value for i in source_matches]
        cat_matches_ra = [i.ra.value for i in cat_matches]
        source_matches_dec = [i.dec.value for i in source_matches]
        cat_matches_dec = [i.dec.value for i in cat_matches]
        # compute offsets 
        ra_offsets = np.subtract(source_matches_ra, cat_matches_ra)*3600.0 # in arcsec
        dec_offsets = np.subtract(source_matches_dec, cat_matches_dec)*3600.0
        self.ra_offsets_mean = np.mean(ra_offsets)
        self.dec_offsets_mean = np.mean(dec_offsets)

        # plot the correlation
        if plot_corr:
            # fit a straight line to the correlation
            from scipy.optimize import curve_fit
            def f(x, m, b):
                return b + m*x
            
            xdata = good_cat_sources[zp_filter+'mag'][idx_cat] # catalog
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
            ax.errorbar(good_cat_sources[zp_filter+'mag'][idx_cat], 
                     self.psf_sources['mag_fit'][idx_image], 
                     self.psf_sources['mag_unc'][idx_image],
                     marker='.', mec="#fc5a50", mfc="#fc5a50", ls="",color='k', 
                     markersize=12, label="Data ["+self.filter+"]", zorder=1) 
            ax.plot(xdata, fitdata, color="blue", 
                     label=r"$y = mx + b $"+"\n"+r"$ m=$%.3f$\pm$%.3f, $b=$%.3f$\pm$%.3f"%(
                             m, m_err, b, b_err), zorder=2) # the linear fit 
            ax.set_xlabel("Catalog magnitude ["+self.ref_cat_name+"]", 
                          fontsize=15)
            ax.set_ylabel('Instrumental PSF-fit magnitude', fontsize=15)
            ax.set_title("PSF Photometry", fontsize=15)
            ax.legend(loc="upper left", fontsize=15, framealpha=0.5)
            plt.savefig(self.filter+"_"+self.instrument+"_"+self.date+
                        "_PSF_photometry."+self.plot_ext, bbox_inches="tight")
            plt.close()
        
        # plot the RA, Dec offset for each matched source 
        if plot_source_offsets:             
            # plot
            plt.figure(figsize=(10,10))
            plt.plot(ra_offsets, dec_offsets, marker=".", linestyle="", 
                    color="#ffa62b")
            plt.xlabel('RA (J2000) offset ["]', fontsize=15)
            plt.ylabel('Dec (J2000) offset ["]', fontsize=15)
            plt.title("Source offsets from %s catalog"%
                         self.ref_cat_name, fontsize=15)
            plt.axhline(0, color="k", linestyle="--", alpha=0.3) # (0,0)
            plt.axvline(0, color="k", linestyle="--", alpha=0.3)
            plt.plot(self.ra_offsets_mean, self.dec_offsets_mean, marker="X", 
                     color="blue", label = "Mean", linestyle="") # mean
            plt.legend(fontsize=15)
            plt.rc("xtick",labelsize=14)
            plt.rc("ytick",labelsize=14)
            plt.savefig(self.filter+"_"+self.instrument+"_"+self.date+
                        "_source_offsets_astrometry."+self.plot_ext,
                        bbox_inches="tight")
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
            plt.title("Field offsets from %s catalog"%self.ref_cat_name, 
                      fontsize=15)
            ax.coords["ra"].set_ticklabel(size=15)
            ax.coords["dec"].set_ticklabel(size=15)
            plt.savefig(self.filter+"_"+self.instrument+"_"+self.date+
                        "_field_offsets_astrometry."+self.plot_ext, 
                        bbox_inches="tight")
            plt.close()
        
        # compute magnitude differences and zero point mean, median and error
        mag_offsets = ma.array(good_cat_sources[zp_filter+'mag'][idx_cat] - 
                      self.psf_sources['mag_fit'][idx_image])

        zp_mean, zp_med, zp_std = sigma_clipped_stats(mag_offsets)
        
        # update attributes 
        self.zp_mean, self.zp_med, self.zp_std = zp_mean, zp_med, zp_std
        
        # add these to the header of the uncleaned, cleaned, and mask files
        scrip_dir = os.getcwd()
        os.chdir(self.calib_dir)
        for n in [self.stack_name, self.stack_name_clean, self.mask_name]:
            f = fits.open(n, mode="update")
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
        self.psf_sources['mag_calib'] = mag_calib
        self.psf_sources['mag_calib_unc'] = mag_calib_unc
        
        # add flag indicating if source is in a catalog and which catalog 
        in_cat = []
        for i in range(len(self.psf_sources)):
            if i in idx_image:
                in_cat.append(True)
            else:
                in_cat.append(False)
        in_cat_col = Column(data=in_cat, name="in_catalog")
        self.psf_sources["in "+self.ref_cat_name] = in_cat_col
        
        # add new columns 
        nstars = len(self.psf_sources)
        col_filt = Column([self.filter for i in range(nstars)], "filter",
                           dtype = np.dtype("U2"))
        col_mjd = Column([self.stack_time for i in range(nstars)], "MJD")
        self.psf_sources["filter"] = col_filt
        self.psf_sources["MJD"] = col_mjd
        
        # compute magnitude differences between catalog and calibration 
        # diagnostic for quality of zero point determination 
        sources_mags = self.psf_sources[idx_image]["mag_calib"]
        cat_mags = good_cat_sources[idx_cat][zp_filter+"mag"]
        mag_diff_mean = np.mean(sources_mags - cat_mags)
        print("\nMean difference between calibrated magnitudes and "+
              self.ref_cat_name+" magnitudes = "+str(mag_diff_mean))
        self.mag_diff_mean = mag_diff_mean
        
        # update bool
        self.photometric_calib = True
        
        
    def PSF_photometry(self, plot_ePSF=True, plot_resid=False, plot_corr=True,
                       plot_source_offsets=True, plot_field_offsets=False, 
                       source_lim=None, gaussian_blur_sigma=30.0, 
                       cat_num=None):
        """
        Input: bools indicating whether to plot the ePSF, residuals of the 
        PSF fitting, the correlation and fit between instrumental and known
        magnitudes for sources in the field, a scatter plot of the offsets 
        in RA and Dec of each source from the relevant catalog and/or a blurred
        image showing any possible structure in the offsets in the field (all
        optional), a limit on the number of sources to fit (optional; default 
        no limit), a sigma to use when applying a Gaussian blur to the field 
        offsets image (optional; default 30.0, only used if 
        plot_field_offsets=True), and a catalog to query for comparison sources 
        (optional, defaults are set if None is given)
        
        Uses the stars in the field with fluxes within the 80th and 90th 
        percentile to develop an empirical ePSF, fits this ePSF to all of the 
        stars some distance away from the edges of the image, obtains 
        instrumental magnitudes, and queries an online catalogue for comparison
        sources to obtain the zero point needed for photometric calibration.
        
        Output: None 
        """
        Stack.__fit_PSF(self, plot_ePSF, plot_resid, source_lim)
        
        if self.psf_fit: # if the PSF Was properly fit 
            Stack.__zero_point(self, plot_corr, plot_source_offsets, 
                               plot_field_offsets,
                                            gaussian_blur_sigma, 
                                            cat_num)
        
        
    def write_PSF_photometry(self, plot_ePSF=True, plot_resid=True, 
                             plot_corr=True, plot_source_offsets=True, 
                             plot_field_offsets=False, source_lim=None, 
                             gaussian_blur_sigma=30.0, cat_num=None, 
                             output=None):
        """
        Input: the same as PSF_photometry, with an additional arg for the 
        filename of the output file 
        Performs photometry if it has not already been performed, and then
        writes psf_sources to a .fits file. 
        Output: None
        """
        
        if not(self.photometric_calib):
            print("\nPhotometric calibration has not yet been performed, so "+
                  "it will be completed now.\n")
            Stack.PSF_photometry(self, plot_ePSF, plot_resid, plot_corr, 
                                 plot_source_offsets, plot_field_offsets, 
                                 source_lim, gaussian_blur_sigma, cat_num)
        
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
        Input: for manual adjustment, the RA and Dec by which to adjust the 
        reference pixel (optional; default None, which calls automatic 
        adjustment based on the offsets computed during PSF photometry)
        
        If PSF photometry has been completed, then the average offset in RA, 
        Dec between the astrometric solution and the coorinates of the 
        relevent catalog is known and can be used to adjust the astrometric 
        solution of the image. PSF photometry should then be re-done. 
        
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
            print("\nThe astrometry of "+self.stack_name+" has been updated "+
                  "according to the offsets computed in PSF_photometry. These "
                  'offsets are: \nRA_offset = %.4f" \nDec_offset = %.4f" '%
                  (self.ra_offsets_mean, self.dec_offsets_mean))
            f[0].header["CRVAL1"] -= self.ra_offsets_mean/3600.0
            f[0].header["CRVAL2"] -= self.dec_offsets_mean/3600.0
        else: # if RA, Dec given, manual adjustment
            f[0].header["CRVAL1"] -= ra
            f[0].header["CRVAL2"] -= dec
        f.close()
        self.image_header = fits.getheader(self.stack_name)
        

        # separations are now approx. 0, can be computed again by running
        # PSF_photometry once more 
        self.sep_mean = 0.0  
        self.ra_offsets_mean = 0.0
        self.dec_offsets_mean = 0.0
              
        print("Offset attributes have been set to 0.0, but will be updated if"+
              " PSF_photometry is called again.")
        
        os.chdir(script_dir)    

###############################################################################
    #### APERTURE PHOTOMETRY ####
        
    def __drop_aperture(self, ra, dec, ap_radius=1.2, r1=2.0, r2=5.0,
                        plot_annulus=False, plot_aperture=False, 
                        bkgsub_verify=True):
        """
        Input: the right ascension and declination (in degrees) of a source 
        around which to build an aperture of radius ap_radius (in arcsec;
        optional; default 0.9"), an annulus of inner radius r1 and outer r2 
        (in arcsec; optional; default 5.0" and 20.0") in which to estimate the 
        background in the region (radii and annuli radii are optional), and 
        bools indicating whether or not to plot the annulus image data 
        (optional; default  False), whether or not to plot the aperture and 
        annulus as rings (optional; default False) and whether to verify that
        the background-subtracted flux is positive
        
        This method finds the total flux in a defined aperture, computes the 
        background in an annulus around this aperture, and computes the 
        background-subtracted flux of the "source" defined by the aperture.
        
        Output: a table containing the pix coords, ra, dec, aperture flux, 
        aperture radius, annulus inner and outer radii, the median background, 
        total background in aperture, standard deviation in this background, 
        and background-subtracted aperture flux 
        """
        from photutils import (SkyCircularAperture, aperture_photometry,
                       SkyCircularAnnulus)
        
        image_data = self.image_data # the UNCLEANED image data 
        image_header = self.image_header # image header
        image_mask = self.mask_data # mask 
                
        # wcs object
        w = wcs.WCS(image_header)
        
        # lay down the aperture 
        position = SkyCoord(ra, dec, unit="deg", frame="icrs") # source posn
        ap = SkyCircularAperture(position, r=ap_radius*u.arcsec) # aperture 
        ap_pix = ap.to_pixel(w) # aperture in pix
        
        # table of the source's x, y, and total flux in aperture
        phot_table = aperture_photometry(image_data, ap_pix, 
                                         error=self.image_error)
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
        # mask out sources 
        image_data_masked = np.ma.masked_where(image_mask, image_data)
        image_data_masked.fill_value = 0
        image_data_masked = image_data_masked.filled()
        annulus_data = annulus_masks[0].multiply(image_data_masked)
        mask = annulus_data <= 0 # mask invalid data 
        annulus_data = np.ma.masked_where(mask, annulus_data)
        
        # estimate background as median in the annulus 
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
                  "Exiting.")
            return         
        
        if plot_annulus:
            Stack.__plot_annulus(self, ra, dec, r1, r2, annulus_data)   
        if plot_aperture:
            Stack.__plot_aperture(self, ra, dec, ap_pix, r1, r2, 
                                  annulus_apertures)  
        return phot_table
    
    
    def __error_array(self):
        """
        Input: None
        Computes the error on the background-only image as the RMS deviation 
        of the background, and then computes the total image error including 
        the contribution of the Poisson noise for detected sources. Necessary 
        for error propagation in aperture photometry. 
        Output: None 
        """

        from photutils.utils import calc_total_error
        image_data = self.image_data # the UNCLEANED image data 
        image_mask = self.mask_data # mask 
        
        if "WIRCam" in self.instrument:
            eff_gain = 3.8 # effective gain (e-/ADU) for WIRCam
        else: 
            image_header = self.image_header
            eff_gain = image_header["GAIN"] # effective gain for MegaPrime
                    
        # mask out sources and convert to bool for background estimation
        image_mask = image_mask.astype(bool)
        # mask out 0 regions near borders/corners
        zero_mask = image_data <= 0 
        # combine the masks with logical OR 
        image_mask = np.ma.mask_or(image_mask, zero_mask)
        
        # estimate background 
        bkg_est = MMMBackground()
        bkg = Background2D(image_data, (10,10), filter_size=(3,3), 
                           bkg_estimator=bkg_est, mask=image_mask)
        # compute sum of Poisson error and background error  
        ### currently, this seems to overestimate unless the input data is 
        ### background-subtracted
        err = calc_total_error(image_data-bkg.background, 
                               bkg.background_rms, eff_gain)
        
        self.image_error = err
        self.image_error_computed = True
    
    
    def __plot_annulus(self, ra, dec, r1, r2, annulus_data):
        """
        Input: the RA and Dec for the centre of an annulus, the inner and 
        outer radii for the annuli, and the data of the annulus itself
        Plots an image of the annulus for a given aperture computation. 
        Output: None
        """
        # plotting
        fig, ax = plt.subplots(figsize=(10,10)) 
        plt.imshow(annulus_data, origin="lower", cmap="magma")
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        cb.set_label(label="ADU",fontsize=15)
        plt.title('Annulus around %.5f, %.5f (1 pixel = %s")'%(ra, dec,
                  str(self.pixscale)), fontsize=15)
        plt.xlabel("Pixels", fontsize=15)
        plt.ylabel("Pixels", fontsize=15)
        
        # textbox indicating inner/outer radii of annulus 
        textstr = r'$r_{in} = %.1f$"'%r1+'\n'+r'$r_{out} = %.1f$"'%r2
        box = dict(boxstyle="square", facecolor="white", 
           alpha=0.6)
        plt.text(0.81, 0.91, transform=ax.transAxes, s=textstr, bbox=box,
                 fontsize=14)
        
        figname = self.filter+"_"+self.instrument+"_"+self.date
        figname += "_annulus_RA%.5f_DEC%.5f"%(ra, dec)+"."+self.plot_ext
        plt.savefig(figname, bbox_inches="tight")
        plt.close()
        
        
    def __plot_aperture(self, ra, dec, ap_pix, r1, r2, annulus_pix):
        """
        Input: the RA and Dec for the centre of the aperture, the radius of 
        the aperture and inner and outer annuli, and the pixel data of the 
        annulus 
        Plots an image of the aperture and annuli drawn around a source of 
        interest for aperture photometry.
        Output: None
        """
        image_data = self.image_data # the UNCLEANED image data 
        image_header = self.image_header # image header
        
        # wcs object
        w = wcs.WCS(image_header)

        # update wcs object and image to span a box around the aperture
        xpix, ypix = ap_pix.positions[0] # pix coords of aper. centre 
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
        plt.imshow(image_data_temp, origin="lower", cmap="magma")
        ap_pix.plot(color='white', lw=2) # aperture as white cirlce
        annulus_pix.plot(color='red', lw=2) # annuli as red circles 
        cb = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
        cb.set_label(label="ADU", fontsize=15)
        plt.title("Aperture photometry around %.5f, %.5f"%(ra, dec), 
                  fontsize=15)
        textstr = r'$r_{aper} = %.1f$"'%(ap_pix.r*self.pixscale)+'\n'
        textstr += r'$r_{in} = %.1f$"'%r1+'\n'
        textstr += r'$r_{out} = %.1f$"'%r2
        box = dict(boxstyle="square", facecolor="white", alpha=0.6)
        plt.text(0.83, 0.88, transform=ax.transAxes, s=textstr, bbox=box, 
                 fontsize=14)
        plt.xlabel("RA (J2000)", fontsize=16)
        plt.ylabel("Dec (J2000)", fontsize=16)
        
        figname = self.filter+"_"+self.instrument+"_"+self.date
        figname += "_aperture_RA%.5f_DEC%.5f"%(ra, dec)+"."+self.plot_ext
        
        ax.coords["ra"].set_ticklabel(size=15)
        ax.coords["dec"].set_ticklabel(size=15)
        
        plt.savefig(figname, bbox_inches="tight")
        plt.close()
    
    
    def aperture_photometry(self, ra_list, dec_list, ap_radius=1.2, 
                            r1=2.0, r2=5.0, sigma=None, plot_annulus=False,
                            plot_aperture=False):
        """
        Input: for as many sources as is desired, the RA and Dec (in degrees) 
        of the source around which to build an aperture with radii ap_radii 
        (in arcsec; optional; default 1.2"), and annuli with inner radii in 
        r1_list (in arcsec; optional; default 2.0") and outer radii in r2_list 
        (in arcsec; optional; default 5.0"), the limiting sigma below which a 
        source is labelled non-detected (optional; default no limit), bools 
        indicating whether to plot the annulus (optional; default False) and 
        whether to plot the aperture and annuli as rings (optional; default 
        False)
        
        For dim sources, this method finds the total flux in a defined 
        aperture, computes the background in an annulus around this aperture, 
        and computes the background-subtracted flux of the "source" defined by 
        the aperture. Can be called multiple times if a list of RA/Decs is 
        given. 
        
        Output: None
        """        
        if not self.image_error_computed: # if error array is not yet computed
            Stack.__error_array(self) # compute it 
            
        # initialize table of sources found by aperture photometry if needed
        if not(self.aperture_fit):
            cols = ["xcenter","ycenter", "ra","dec", "aperture_sum", 
                    "aperture_sum_err", "aper_r", "annulus_r1", "annulus_r2",
                    "annulus_median", "aper_bkg", "aper_bkg_std", 
                    "aper_sum_bkgsub", "aper_sum_bkgsub_err", "mag_fit", 
                    "mag_unc", "mag_calib", "mag_calib_unc", "sigma"]
            self.aperture_sources = Table(names=cols)
            filt_col = Column([], "filter", dtype='S2') # specify
            mjd_col = Column([], "MJD")
            self.aperture_sources.add_column(filt_col)
            self.aperture_sources.add_column(mjd_col)
            
        # if PSF photometry has not been performed, can't get aperture mags 
        if not(self.photometric_calib):
            print("Cannot obtain magnitudes through aperture photometry "+ 
                  "because photometric calibration has not yet been obtained."+
                  " Exiting.\n")
            return
                
        # convert to lists if needed 
        if (type(ra_list) in [float, int]):
            ra_list = [ra_list]
        if (type(dec_list) in [float, int]):
            dec_list = [dec_list]
        
        # compute background-subtracted flux for the input aperture(s) 
        # add these to the list of sources found by aperture photometry 
        for i in range(0, len(ra_list)):
            phot_table = Stack.__drop_aperture(self, ra_list[i], dec_list[i],
                                               ap_radius, r1, r2, plot_annulus,
                                               plot_aperture)
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
                    self.aperture_sources.add_row(phot_table[0])
                    self.aperture_fit = True # update 

                    
                elif sigma and (phot_table["sigma"] < sigma):
                    print("\nA 'source' was detected, but below the requested"+
                          " "+str(sigma)+". The source is therefore rejected.")
                    return
                else:
                    self.aperture_sources.add_row(phot_table[0])
                    self.aperture_fit = True # update 
                    
                a = phot_table[0]
                s = "\n"+a["filter"]+" = %.2f ± %.2f"%(a["mag_calib"],
                    a["mag_calib_unc"])+", %.1f"%a["sigma"]+"sigma"
                print(s)
                
                
    def limiting_magnitude(self, ra, dec, sigma=3.0):
        """
        Input: the RA and Dec around which we want a limiting magnitude and the 
        sigma defining the limiting magnitude (optional; default 3.0)
        Output: the limiting magnitude 
        """

        image_header = self.image_header
        sources_data = self.xy_data 
                
        if not self.image_error_computed: # if error array is not yet computed
            Stack.__error_array(self) # compute it 
        
        w = wcs.WCS(image_header)
        coords = w.all_pix2world(sources_data["X"], sources_data["Y"], 1)
        coords = SkyCoord(coords[0]*u.deg, coords[1]*u.deg, 1)
        target = SkyCoord(ra*u.deg, dec*u.deg)
        
        # find the smallest separation between target and all sources
        smallest_sep = np.min(target.separation(coords).value)*3600.0 
        while smallest_sep < 3.0: # while closest star is less than 1" away
            print('\nastrometry.net previously found a source < 3.0" away '+
                  'from the target. The target will be randomly moved until '+
                  ' it does not sit on top of a source...')
  
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
        
        #w = wcs.WCS(image_header) # wcs object 
        #position = SkyCoord(ra, dec, unit="deg", frame="icrs") # desired posn
        #ap = SkyCircularAperture(position, r=1.2*u.arcsec) 
        #ap_pix = ap.to_pixel(w)
        
        # do aperture photometry on region of interest
        # use a large annulus
        phot_table = Stack.__drop_aperture(self, ra, dec, r1=2.0, r2=20.0, 
                                           bkgsub_verify=False)
        
        #bkg_std_total = phot_table["aper_bkg_std"]*ap_pix.area()   
 
        phot_table["aper_sum_bkgsub_err"] = np.sqrt(
                phot_table["aperture_sum_err"]**2 +
                phot_table["aper_bkg_std"]**2)
        
        # compute limit below which we can't make a detection
        limit = sigma*phot_table["aper_sum_bkgsub_err"][0]
        self.limiting_mag = -2.5*np.log10(limit) + self.zp_mean
        
        #limit = sigma*bkg_std_total
        #self.limiting_mag = -2.5*np.log10(limit) + self.zp_mean
        
        print("\n"+self.filter+" > %.1f (%d sigma)"%(self.limiting_mag,sigma))
        return self.limiting_mag


    def write_aperture_photometry(self, output=None):
        """
        Input: a filename for the output file which will contain a table of 
        the sources found by aperture photometry.
        Performs aperture photometry if it has not already been performed, and 
        then writes aperture_sources to a .fits file. 
        Output: None
        """
        
        if not(self.aperture_fit):
            print("No aperture photometry has been performed yet. Exiting.\n")
            return
            
        to_write = self.aperture_sources
        
        if not(output): # if no name given
            output = self.stack_name.replace("_updated.fits", 
                                                  "_aperture_photometry.fits")
        to_write.write(output, overwrite=True, format="ascii.ecsv")
        
###############################################################################
    #### COMPARE APERTUTRE AND PSF PHOTOMETRY ####
        
    def compare_photometry(self, ap_radius=1.2, r1=2.0, r2=5.0, nsamples=100, 
                           output=None):
        """
        Input: the aperture radius, inner and outer annuli radii, the number of 
        samples to take from the the PSF photometry calibrated data (optional; 
        default 100) and an output name for the figure (optional)
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
        Stack.aperture_photometry(self, ras, decs, ap_radius, r1, r2)
        
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
                    label="Data ["+self.filter+"]", zorder=1)
        # plot the linear fit to the data
        ax.plot(ap_mags, fitdata, color="blue", 
                 label=r"$y = mx + b $"+"\n"+r"$ m=$%.3f$\pm$%.3f, $b=$%.3f$\pm$%.3f"%(
                         m, m_err, b, b_err), zorder=2)
        ax.set_xlabel("Aperture photometry calibrated magnitude", fontsize=15)
        ax.set_ylabel("PSF photometry calibrated magnitude ["+
                    self.ref_cat_name+"]", fontsize=15)
        ax.legend(loc="upper left", fontsize=15)
        ax.set_title("Comparison between different methods of photometry "+
                     "[no. of samples = %d]"%nsamples, fontsize=15)
        
        if not(output): # if no name given
            output = self.stack_name.replace("_updated.fits", 
                                                  "_photometry_compare."+
                                                  self.plot_ext)
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        
        self.aperture_sources = temp # restore original table 
        
###############################################################################
    #### SOURCE SELECTION ####

    def source_selection(self, ra, dec, radius=1.0):
        """
        Input: a RA and Dec for a source of interest and a radius to search in 
        (optional; default 1.0")
        Parses the astropy table of sources produced by PSF_photometry() for 
        sources within some distance radius from the given ra and dec 
        arguments. 
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
                                         "in "+self.ref_cat_name])
        return selected_psf_sources # return selected sources in astropy table 
    
    
    def write_selection(self, ra, dec, radius=1.0, output=None):
        """
        Input: the same as source_selection, with an additional arg for the 
        filename of the output file 
        Selects sources as per source_selection() and then writes them to an 
        astropy table with the correct format.
        Output: None
        """
        psf_selection = Stack.source_selection(self, ra, dec, radius)
        
        if not(output): # if no name given
            output = self.stack_name.replace("_updated.fits", 
                                                  "_PSF_photom_selection.fits")
        psf_selection.write(output, overwrite=True, format="ascii.ecsv")
    
