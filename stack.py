#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:43 2019
@author: Nicholas Vieira
@stack.py
"""

# run iraf in terminal-only mode (no graphics):
from stsci.tools import capable
capable.OF_GRAPHICS = False

import sys
import os
#from pyraf.iraf import noao, imred, ccdred
#from pyraf.iraf import immatch
from pyraf.iraf import imcombine
#from astropy.io import fits
#import numpy as np

# instrument/date/valid filters taken as command line arguments
argc = len(sys.argv)
instrument = str(sys.argv[1]) # first arg: instrument
dat = str(sys.argv[2]) # second: date

filter_list = [] 
for a in (sys.argv)[3:argc]: # remainder: valid filters
    filter_list.append(str(a))
#print("\n stack.py: filter_list = "+str(filter_list))

#if "WIRCam" in instrument:  
#    readout_noise = "30.0" # readout noise in e-
#    eff_gain = "3.8" # same gain for all detectors, in e-/ADU
#else:
#    readout_noise = "3.0" # approximate 
#    eff_gain = "!GAIN" # point to GAIN header of images 

for f in filter_list:
    filter_file = f+'_list.txt' # file of a particular filter's list
    cwd = os.getcwd() # the stack directory of the object 
    #print("\n stack.py: cwd = "+cwd)
    
    # if file present and non-empty:
    if (filter_file in os.listdir(cwd)) and (os.stat(filter_file).st_size!=0):
        # stack the images:
        imcombine("@"+f+"_list.txt", 
                  combine="median",
                  offset="world",
                  output=f+"_stack_"+dat+".fits",
                  reject="sigclip", # sigma clipping
                  lsigma="6.0", # sigma below which to reject
                  hsigma="6.0", # sigma above which to reject
                  #reject="crreject", # cosmic ray reject
                  #rdnoise=readout_noise, 
                  #gain=eff_gain, 
                  #rejmask=f+"_rejmask_"+dat+".fits", # rejected pixels
                  #sigma = f+"_sigma_"+dat+".fits", # sigma at pixels 
                  masktype="badvalue", # mask pixels with a value of 0.0
                  maskval="0.0") 
    
    
    
        
        







