==================
Using the pipeline
==================

alala.py
========

This script is object-oriented and contains two classes: the ``RawData`` class and its subclass the ``Stack``. Here, we will work through an example using WIRCam data. 

RawData
-------

Images from WIRCam arrive largely de-trended via CFHT's pipeline 'I'iwi. WIRCam is an array of 4 detectors, each approximately 10' (arcmin) x 10'.  Every file from WIRCam will be a multi-extension fits file, with one extension for each detector. These extensions are usually cubes themselves. The correspondence between the detectors and extensions is:

![](https://github.com/nvieira-mcgill/alala/blob/master/images/wircam_detectors.png?raw=true)

Importantly, the convention for WIRCam is to treat each of these 4 detectors separately. This means that the same star, observed on different detectors, can have a widely varying flux. For this reason, it's important to decide which detector you want to work with. 

Let's say you've put your data in some directory ``/data/myWIRCam/``. Let's initialize a ``RawData`` object:

     >>> import alala 
     >>> datadir = "/data/myWIRCam/" 
     >>> rawdata = alala.RawData(datadir)

This will examine all of the data in your ``datadir`` and store it in attributes of the object based on filter and dates. The filters typically in use are Y, J, H, Ks. If you want to see all the files in the Y filter, in order of acquisition time:

     >>> rawdata.Y
     'Y_file1.fits.fz', 'Y_file2.fits.fz' # and many more, probably

If you want to see the filters spanned by the data:

     >>> rawdata.filters
     ['Y', 'J', 'H']

If you want to see the date(s) spanned by data: 

     >>> rawdata.date
     '20181106'

If your data spans multiple dates, this will output ``'multidate'``, in which case the attribute ``rawdata.dates`` will contain a list of these dates in chronological order and the attribute ``rawdata.dates_dict`` will contain these dates, and their corresponding files, in a dictionary. If you want to examine one or more headers in, say, the 2nd extension of these multiextension fits files:

     >>> ext_of_interest = 2
     >>> rawdata.print_headers(ext_of_interest, "FILTER", "EXPTIME")
     FILE            FILTER          EXPTIME
     Y_file1.fits.fz Y               30.0
     Y_file2.fits.fz Y               30.0
     J_file1.fits.fz J               15.0
     J_file1.fits.fz J               15.0
     # and many more 

To decide which detector you want to use, if you know the RA and Dec of the source you care about: 

     >>> ra = 303.8325417
     >>> dec = 15.5173611
     >>> rawdata.locate_WCS(ra, dec)

Will examine the **first** file in ``datadir`` and tell you which extension contains these coordinates. Now, let's say your data is in the 3rd extension. Doing the following:

     >>> rawdata.write_extension(3)

Will write the 3rd extension of all files in ``datadir``, which we said was ``/data/myWIRCam/``, to a new directory 
``/data/myWIRCam/det3_WIRCam_20181106``. We can then make another object:

     >>> newdatadir = "/data/myWIRCam/det3_WIRCam_20181106"
     >>> newrawdata = alala.RawData(newdatadir)

Importantly, MegaCam data is typically **not** a datacube. To allow the pipeline to smoothly handle both WIRCam and MegaCam data, we take each datacube in our new data object and divide them into separate files: 

    >>> newrawdata.divide_WIRCam()

If each of the files in ``newdatadir`` was a cube of 2 images, this effectively just doubles the number of files. The new files will be located in ``/data/myWIRCam/divided_det3_WIRCam_20181106``. We again make a new object: 

     >>> finaldatadir = "/data/myWIRCam/divided_det3_WIRCam_20181106"
     >>> finalrawdata = alala.RawData(finaldatadir)

We can use several diagnostics to test the quality of these images and decide if any of the raw data should be discarded. These include: 

     >>> finalrawdata.value_at(ra, dec) # get the flux at this RA, Dec for all raw data
     >>> finalrawdata.background() # naively estimate background as median of the whole image for all raw data

We can also examine the radial PSF for a given RA, Dec. **This method is more involved and requires that you first refine the astrometry of all the raw data. It is not very useful at the moment, so feel free to skip this next snippet.** To do so: 

     >>> finalrawdata.solve_all() # solve all of the data -- this takes fairly long 
     >>> solved_finalrawdata = alala.RawData("solved"+finaldatadir, stackdir) # new object
     >>> solved_finalrawdata.radial_PSFs(ra, dec)

This will save plots of the radial PSFs to a new directory for all of the raw data.

**Important:** if you don't want to diagnose the images yourself, you can provide an additional argument when initializing the ``RawData`` object to ignore data of poor quality:

     >>> finalrawdata = alala.RawData(finaldatadir, qso_grade_limit=2)

The queue service observer (QSO) grade is a grade provided by the QSO which rates the image quality at the time of acquisition, where 1=Good and 5=Unusable. A QSO grade of 1 or 2 is good, but feel free to lower the quality to 3 or even 4 if you don't have much data to work with. **The default value is 4**, so that no data is excluded, but it is strongly recommended to apply a more strict limit if possible.

The last step we have to take before stacking is to make a bad pixel mask of each of the images. CFHT helpfully flags bad pixels with a value of 0 for us. This is done with:

     >>> finalrawdata.make_badpix_masks()

This updates the raw data to point to these masks and creates a new directory, ``/data/myWIRCam/badpixels_divided_det3_WIRCam_20181106``, to store the masks. With these steps complete, we can now make a stack. Note that the above steps **do not** need to be redone unless any of the directories are deleted. A condensed example of all the above follows. 

     >>> import alala
     >>>
     >>> # the first object 
     >>> rawdata = alala.RawData("/data/myWIRCam")
     >>> exten = raw.locate_WCS(303.5, 15.6)
     >>> rawdata.write_extension(exten) # let's say exten is 3
     >>>
     >>> # second object
     >>> newrawdata = alala.RawData("/data/myWIRCam/det3_WIRCam_20181106") # second object
     >>> newrawdata.divide_WIRCam()
     >>>
     >>> # final object 
     >>> finalrawdata = alala.RawData("/data/myWIRCam/divided_det3_WIRCam_20181106", qso_grade_limit=2)
     >>> finalrawdata.make_badpix_masks()

Stack
-----

We need to tell the object where to put stacks. We can do this via:

     >>> workingdir = "/exports/myWIRCam/workdir"
     >>> finarawdata.set_stackdir(workingdir)

Alternatively, we can do this right away when initializing the object: 

     >>> working_dir = "/exports/myWIRCam/workdir"
     >>> finalrawdata = alala.RawData(finaldatadir, stack_directory=working_dir)

Stacking is now a one-liner. If we have data in all four Y, J, H and Ks filters:

     >>> finalrawdata.make_stacks()

Will copy all raw data to the stack directory, save lists of the files in each filter in text files, initiate IRAF via the script ``stack.py``, and produce stacks for each filter. These files will all have the form ``H_stack_20181106.fits``, where the "H" and "20181106" are the filter and date, respectively. If we only care about one or more of the filters, e.g. J and H, 

     >>> finalrawdata.make_stacks("J", "H")

Will produce only those we care about. **Note:** IRAF has a limit on the number of files it can stack, and may crash if you try and stack too many images at once. If this is the case, consider stacking in batches and then stacking those stacks. To now extract the ``Stack`` object:

     >>> j = finalrawdata.extract_stack("J")

Note that, if you try to extract a stack before it has been made, the stack will automatically be produced. A Stack object can also be initialized directly:

     >>> j = alala.Stack(finaldatadir, workingdir, filt="J")

And, again, the stack will first be produced if it does not already exist. A condensed example of the process from raw data to stack follows: 

     >>> import alala
     >>>
     >>> # the first object 
     >>> rawdata = alala.RawData("/data/myWIRCam")
     >>> exten = raw.locate_WCS(303.5, 15.6)
     >>> rawdata.write_extension(exten) # let's say exten is 3
     >>>
     >>> # second object
     >>> newrawdata = alala.RawData("/data/myWIRCam/det3_WIRCam_20181106") # second object
     >>> newrawdata.divide_WIRCam()
     >>>
     >>> # final object 
     >>> finalrawdata = alala.RawData("/data/myWIRCam/divided_det3_WIRCam_20181106", qso_grade_limit=2)
     >>> finalrawdata.make_badpix_masks()
     >>>
     >>> # let's say we only care about the J band 
     >>> j = alala.Stack("/data/myWIRCam/divided_det3_WIRCam_20181106", 
                         "/exports/myWIRCam/working_dir", qso_grade_limit=2)

Performing astrometry, photometry
---------------------------------
