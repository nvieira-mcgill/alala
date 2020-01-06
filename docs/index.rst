======
Alala
======

A pipeline for the Wide-field Infra-Red Camera (WIRCam) and MegaCam instruments at the Canada-France-Hawaii Telescope (CFHT). Does stacking, astrometry, PSF photometry and aperture photometry. 

Named after the ʻAlalā, the Hawaiian crow, known for its intelligence and flying ability.

Sections:
     * `Installing the required dependencies <https://alala.readthedocs.io/en/latest/#installing-the-required-dependencies>`_
     * `Using the pipeline <https://alala.readthedocs.io/en/latest/#using-the-pipeline>`_
     * `Making light curves <https://alala.readthedocs.io/en/latest/#making-light-curves>`_

====================================
Installing the required dependencies
====================================

Modules you'll need 
-------------------

``alala.py`` and ``stack.py`` were optimized for use on the ``irulan`` server of McGill University. Using this software is much easier if you have access to this server or some other, but installation instructions are given for those who do not. If you do have access to this server, you should still read the instructions below so that your dependencies are placed in the right location. ``lightcurve.py``, conversely, can be used on any machine that has ``astropy`` (which comes with most scientific python installations, such as anaconda). 

This software makes use of certain modules you may not already have. These are:

     * `astropy <http://docs.astropy.org/en/stable/install.html>`_ - Widely-used software for astrophysical programming. 
     
     * `astroquery <https://astroquery.readthedocs.io/en/latest/#installation>`_ - A tool for querying online catalogues. 

     * `photutils <https://photutils.readthedocs.io/en/stable/install.html>`_ - Software affiliated with astropy, used specifically for photometry. 

     * `pyraf (iraf) <http://www.stsci.edu/institute/software_hardware/pyraf>`_ - A Python wrapper for an industry standard tool. Used here to stack images.

     * `astrometry.net <http://astrometry.net/doc/readme.html#installing>`_ - A tool for source detection and astrometric calibration of images.

astropy, astroquery, photutils 
------------------------------

Installing these is straightforward with ``conda``, ``pip``, etc. See the links above. You likely already have at least ``astropy`` on your own machine/on whichever server you're accessing. 

astrometry.net 
--------------

Make sure to get the **bleeding edge** version from `github <https://github.com/dstndstn/astrometry.net>`_.
Older versions of astrometry impose a limit on the number of sources that can be detected in an image, and this poses problems for the typically very dense images taken by WIRCam/MegaCam. Follow the instructions for installation given `here
<http://astrometry.net/doc/readme.html#installing>`_.

If you are on `irulan`, you only need to add the following to your .bashrc: 

     | export PATH="/sbin/:$PATH"
     | export PATH="/data/irulan/astrometry:$PATH"
     | export PATH="/data/irulan/astrometry/bin:$PATH"

Ensuring that iraf works correctly 
----------------------------------
Installing this correctly can be very difficult. If you have access to the ``irulan`` server of McGill University, this will already be done. If your institution has a server dedicated to physics/astrophysics, it will probably already have ``iraf`` as well.

Once you have installed ``iraf`` correctly, in order to stack images you will need an ``iraf`` directory in your **home directory**, with a ``login.cl`` file and ``pyraf`` and ``uparm`` directories inside this directory. You can use the ``login.cl`` included `here <https://github.com/nvieira-mcgill/alala/tree/master/iraf_setup>`_, remembering to change the following lines at the beginning of the file:

     | set	home		= "/home/johndoe/iraf/"
     | set	imdir		= "/tmp/johndoe/" 
     | set	uparm		= "home$uparm/"
     | set	userid		= "johndoe"

To match your home directory and user id. 

--------------------------

==================
Using the pipeline
==================

The pipeline's main script, ``alala.py``, is object-oriented and contains two classes: the ``RawData`` class and its subclass the ``Stack``. **Here, we will work through an example using WIRCam data**. Important notes on differing conventions for MegaCam data will be marked **MegaCam Note**. 

RawData
-------

Images from WIRCam arrive largely de-trended via CFHT's pipeline 'I'iwi. WIRCam is an array of 4 detectors, each approximately 10' (arcmin) x 10'.  Every file from WIRCam will be a multi-extension fits file, with one extension for each detector. These extensions are also usually cubes themselves. The correspondence between the detectors and extensions is:

.. image:: https://github.com/nvieira-mcgill/alala/blob/master/images/wircam_detectors.png?raw=true

Importantly, the convention for WIRCam is to treat each of these 4 detectors separately. This means that the same star, observed on different detectors, can have a widely varying flux. For this reason, it is important to decide which detector you want to work with. 

Let's say you've put your data in some directory ``/data/myWIRCam/``. Let's initialize a ``RawData`` object:

.. code-block:: python

     >>> import alala
     >>> datadir = "/data/myWIRCam/" 
     >>> rawdata = alala.RawData(datadir)
     

This will examine all of the data in your ``datadir`` and store it in attributes of the object based on filter and dates. The filters typically in use are Y, J, H, Ks. If you want to see all the files in the Y filter, in order of acquisition time:

.. code-block:: python

     >>> rawdata.Y
     'Y_file1.fits.fz', 'Y_file2.fits.fz' # and many more, probably

If you want to see the filters spanned by the data:

.. code-block:: python

     >>> rawdata.filters
     ['Y', 'J', 'H']

If you want to see the date(s) spanned by data: 

.. code-block:: python

     >>> rawdata.date
     '20181106'

If your data spans multiple dates, this will output ``'multidate'``, in which case the attribute ``rawdata.dates`` will contain a list of these dates in chronological order and the attribute ``rawdata.dates_dict`` will contain these dates and their corresponding files in a dictionary. If you want to examine one or more headers in, say, the 2nd extension of these multiextension fits files:

.. code-block:: python

     >>> ext_of_interest = 2
     >>> rawdata.print_headers(ext_of_interest, "FILTER", "EXPTIME")
     FILE            FILTER          EXPTIME
     Y_file1.fits.fz Y               30.0
     Y_file2.fits.fz Y               30.0
     J_file1.fits.fz J               15.0
     J_file1.fits.fz J               15.0
     # and many more 

Finally, to decide which detector you want to use, if you know the RA and Dec of your source: 

.. code-block:: python

     >>> ra = 303.8325417
     >>> dec = 15.5173611
     >>> rawdata.locate_WCS(ra, dec)

Will examine the **first** file in ``datadir`` and tell you which extension contains these coordinates. Now, let's say your data is in the 3rd extension. Doing the following:

.. code-block:: python

     >>> rawdata.write_extension(3)

Will write the 3rd extension of all files in ``datadir``, which we said was ``/data/myWIRCam/``, to a new directory 
``/data/myWIRCam/det3_WIRCam_20181106``. We can then make another object:

.. code-block:: python

     >>> newdatadir = "/data/myWIRCam/det3_WIRCam_20181106"
     >>> newrawdata = alala.RawData(newdatadir)
 
------------------------------------------------------------------- 
 
**MegaCam Note:** MegaCam has a much wider FOV of about 1 square degree compared to the 20' (arcmin) x 20' square spanned by WIRCam. For this reason, the FOV moves around more during MegaCam observations, especially when studying extended objects such as galaxies, nebulae, or globular clusters. Moreover, MegaCam data has the opposite convention of WIRCam for flux calibration: all detectors are calibrated to the same level. This means it is ok to stack different MegaCam CCDs into the same image. We therefore use: 

.. code-block:: python

     >>> import alala
     >>> datadir = "/data/myMegaCam/" 
     >>> rawdata = alala.RawData(datadir) # a new object 
     >>> ra, dec = 153.5590, 63.012
     >>> rawdata.write_extensions_by_WCS(ra, dec)
    
This extracts the CCD from each image which contains these RA, Dec and writes them to a new directory such as ``/data/myWIRCam/dets_RA153.559_DEC63.012_MegaCam_20110816``. 

-------------------------------------------------------------

To allow the pipeline to smoothly handle both WIRCam and MegaCam data, we take each datacube in our new data object and divide them into separate files: 

.. code-block:: python

    >>> newrawdata.divide_WIRCam()

If each of the files in ``newdatadir`` was a cube of 2 images, this effectively just doubles the number of files. The new files will be located in ``/data/myWIRCam/divided_det3_WIRCam_20181106``. We again make a new object: 

.. code-block:: python

     >>> finaldatadir = "/data/myWIRCam/divided_det3_WIRCam_20181106"
     >>> finalrawdata = alala.RawData(finaldatadir)

We can use several diagnostics to test the quality of these images and decide if any of the raw data should be discarded. These include: 

.. code-block:: python

     >>> finalrawdata.value_at(ra, dec) # get the flux at this RA, Dec for all raw data
     >>> finalrawdata.background() # naively estimate background as median of the whole image for all raw data

We can also examine the radial PSF for a given RA, Dec. **This method is more involved and requires that you first refine the astrometry of all the raw data. It is not very useful at the moment, so feel free to skip this next snippet.** To do so: 

.. code-block:: python

     >>> finalrawdata.solve_all() # solve all of the data -- this takes fairly long 
     >>> solved_finalrawdata = alala.RawData("solved"+finaldatadir, stackdir) # new object
     >>> solved_finalrawdata.radial_PSFs(ra, dec)

This will save plots of the radial PSFs to a new directory for all of the raw data.

**Important:** if you don't want to diagnose the images yourself, you can provide an additional argument when initializing the ``RawData`` object to ignore data of poor quality:

.. code-block:: python

     >>> finalrawdata = alala.RawData(finaldatadir, qso_grade_limit=2)

The queue service observer (QSO) grade is a grade provided by the QSO which rates the image quality at the time of acquisition, where 1=Good and 5=Unusable. **Note that the QSO grade is not available for older data, e.g. 2008 and before.** QSO grade of 1 or 2 is good, but quality can be lowered to 3 or even 4 if you don't have much data to work with. The default is to apply no limit, so that no data is excluded, but it is strongly recommended to apply a more strict limit if possible.

-----------------------------------------

**MegaCam Note:** For MegaCam data, a QSO grade is **not** provided. The only way to assess the quality of the data is via weather logs or program metadata.

----------------------------------------

The last step we have to take before stacking is to make a bad pixel mask of each of the images. CFHT helpfully flags bad pixels with a value of 0 for us. This is done with:

.. code-block:: python

     >>> finalrawdata.make_badpix_masks()

This updates the raw data to point to these masks and creates a new directory, ``/data/myWIRCam/badpixels_divided_det3_WIRCam_20181106``, to store the masks. With these steps complete, we can now make a stack. Note that the above steps **do not** need to be redone unless any of the directories are deleted. A condensed example of all the above follows. 

.. code-block:: python

     >>> import alala
     >>>
     >>> # the entire 4-detector mosaic
     >>> rawdata = alala.RawData("/data/myWIRCam")
     >>> exten = raw.locate_WCS(303.5, 15.6)
     >>> rawdata.write_extension(exten) # let's say exten is 3
     >>>
     >>> # only one of the detectors
     >>> newrawdata = alala.RawData("/data/myWIRCam/det3_WIRCam_20181106") 
     >>> newrawdata.divide_WIRCam()
     >>>
     >>> # divided cubes 
     >>> finalrawdata = alala.RawData("/data/myWIRCam/divided_det3_WIRCam_20181106", qso_grade_limit=2)
     >>> finalrawdata.make_badpix_masks()
     
-----------------------------------------

**MegaCam Note:** For MegaCam data, the data does **not** need to be divided. The data never consists of cubes.

----------------------------------------

Stack
-----

We need to tell the object where to put stacks. We can do this via:

.. code-block:: python

     >>> workingdir = "/exports/myWIRCam/workdir"
     >>> finarawdata.set_stackdir(workingdir)

Alternatively, we can do this right away when initializing the object: 

.. code-block:: python

     >>> working_dir = "/exports/myWIRCam/workdir"
     >>> finalrawdata = alala.RawData(finaldatadir, stack_directory=working_dir)

Stacking is now a one-liner. If we have data in all four Y, J, H and Ks filters:

.. code-block:: python

     >>> finalrawdata.make_stacks()

Will copy all raw data to the stack directory, save lists of the files in each filter in text files, initiate IRAF via the script ``stack.py``, and produce stacks for each filter. The final stacks are each the **median** of the input files, with all bad pixels ignored and sigma clipping employed for any data more than 6 sigma away. 

These files will all have the form ``H_stack_20181106.fits``, where the "H" and "20181106" are the filter and date, respectively. If we only care about one or more of the filters, e.g. J and H, 

.. code-block:: python

     >>> finalrawdata.make_stacks("J", "H")

Will produce only those we care about. **Note:** IRAF has a limit on the number of files it can stack, and may crash if you try and stack too many images at once. If this is the case, consider stacking in batches and then stacking those stacks. To now extract the ``Stack`` object:

.. code-block:: python

     >>> j_stack = finalrawdata.extract_stack("J")

Note that, if you try to extract a stack before it has been made, the stack will automatically be produced. A Stack object can also be initialized directly:

.. code-block:: python

     >>> j_stack = alala.Stack(finaldatadir, workingdir, filt="J")

And, again, the stack will first be produced if it does not already exist. A condensed example of the process from raw data to stack follows: 

.. code-block:: python

     >>> import alala
     >>>
     >>> # the entire 4-detector mosaic 
     >>> rawdata = alala.RawData("/data/myWIRCam")
     >>> exten = raw.locate_WCS(303.5, 15.6)
     >>> rawdata.write_extension(exten) # let's say exten is 3
     >>>
     >>> # only one of the detectors
     >>> newrawdata = alala.RawData("/data/myWIRCam/det3_WIRCam_20181106") 
     >>> newrawdata.divide_WIRCam()
     >>>
     >>> # divided cubes 
     >>> finalrawdata = alala.RawData("/data/myWIRCam/divided_det3_WIRCam_20181106", qso_grade_limit=2)
     >>> finalrawdata.make_badpix_masks()
     >>>
     >>> # let's say we only care about the J band 
     >>> j_stack = alala.Stack("/data/myWIRCam/divided_det3_WIRCam_20181106", "/exports/myWIRCam/working_dir", qso_grade_limit=2)

Masks, backgrounds 
------------------

We have a few convenience functions which allow us to make masks and compute the background of our image. These are:

.. code-block:: python
     
     >>> j_stack.bp_mask() # build a simple bad pixel mask where pixels=0 or NaN
     >>> j_stack.source_mask() # use image segmentation to make a mask of all sources in the image
     >>> j_stack.bkg_compute() # compute the background of the image and produce a background-subtracted image
     >>> j_stack.error_array() # compute the total error on the image, including both Gaussian and Poisson error

Computing the error array requires background subtraction, performing background subtraction requires a source mask, and masking out sources requires a basic bad pixel mask. So, if any of these steps have not yet been performed when calling any of these functions, the steps will be done automatically. The error array, background array, background-subtracted image array, etc. are all stored as attributes of the stack object for later use. 


Performing astrometry, photometry
---------------------------------

In this section, we will assume you have the ``j_stack`` object as defined above. Recall that, in our stack working directory, we have a file ``J_stack_20181106.fits``. First, let's refine the **astrometry** for the stack and extract as many sources as possible. To do so, we need the correct **index files** for our field. These are the files which astrometry.net uses to solve the field. For WIRCam and MegaCam images, we can use the  `5000 <http://data.astrometry.net/5000/>`_ series of images, built from GAIA. (The 2 numbers immediately following 50 indicate the scale of the image, with larger numbers indicating wider fields). The 4201 series of 2MASS images from `here <http://data.astrometry.net/4200/>`_ are also well-suited. We must then determine which healpix number corresponds to the approximate RA, Dec of our image. To do see, we consult the following image:

.. image:: https://github.com/nvieira-mcgill/alala/blob/master/images/astrometry.net_hp2.png?raw=true

For example, for a source at an RA ~ 150 degrees, Dec ~ 10 degrees, we would want the index file ``index-4201-25.fits``. Once this is in the ``data`` directory in your astrometry install (for ``irulan``, this is ``/data/irulan/astrometry/data/``), we can get back to pipelining: 

.. code-block:: python

     >>> j_stack.astrometry()
     
This line will do the following: 

     1. Extract as many stars as possible, solve the field, and output an updated WCS header to ``J_stack_20181106_updated.fits``
     2. Output a list of the pixel coordinates and background-subtracted flux for all the previously extracted sources in the fits bintable ``J_stack_20181106_updated.xy.fits``     

These 2 files will be output to a new directory ``calibration`` within the stack directory. It is useful now to take a look at the actual stack itself. We can do so with the ``make_image()`` function, which has many options: 

.. code-block:: python

     >>> j_stack.make_image() # make a plain image with the raw, unsubtracted data
     >>> j_stack.make_image(bkgsub=True) # use the background-subtracted data
     >>> j_stack.make_image(sources=True) # put circles around all extracted sources
     >>> j_stack.make_image(ra=275.15, dec=7.15) # plot a cross-hair at this RA, Dec
     >>> j_stack.make_image(scale="log") # use a log_10 scale 
     >>> j_stack.make_image(output="test.png") # set the name for the output file
    
These arguments, of course, can all be used in conjunction with each other. The default is to plot the unsubtracted data in a linear scale, with none of the additional features. 

Returning to our analysis, we now have all the files needed to perform **PSF photometry**. This is another one-liner: 

.. code-block:: python

     >>> j_stack.PSF_photometry()
     
This line will do the following: 

     1. Using image segmentation, find all sources in the image, and discard overly elongated or extremely large sources (default: exclude sources with elongation > 1.4 and/or area > 500 square pix.)
     2. Using the remaining sources, empirically obtain the effective Point Spread Function (ePSF) of the image 
     3. Fit this ePSF to **all** sources previously detected by astrometry.net to obtain a PSF-fit flux 
     4. Compute the instrumental magnitude of **all** detected sources 
     5. Query an external catalog for sources whose RA and Dec puts them within 2 pixels of our detected sources, and for all matches, obtain the catalog magnitude 
     6. Use sigma-clipping to obtain the mean, median and standard deviation of the offset between the instrumental and catalog magnitudes, i.e., the zero point 
     7. Add this zero point to the instrumental magnitude to obtain the calibrated magnitudes for **all** sources 


Note that the instrumental magnitude is computed as: 

.. math:: 

     m_{ins} = -2.5\cdot\log(FLUX)
     
          
When calling ``PSF_photometry()``, important optional arguments are:

     * ``nstars`` `(int, default 40)` Number of stars to use in building the ePSF
     * ``thresh_sigma`` `(float, default 5.0)` Threshold sigma for source detection with image segmentation
     * ``pixelmin`` `(float, default 20.0)` Minimum pixel area for a source 
     * ``elongation_max`` `(float, default 1.4)` Maximum allowed source elongation
     * ``area_max`` `(float, default 500.0)` Maximum pixel area for a source
     * ``sep_max`` `(float, default 2.0)` Maximum number of pixels separating two sources when cross-matching to external catalogue 
     * ``plot_ePSF`` `(bool, default True)` Plot the ePSF
     * ``plot_residuals`` `(bool, default True)` Plot the residuals of the ePSF fit 
     * ``plot_corr`` `(bool, default True)` Plot the instrumental versus catalog magnitudes, with a linear fit 
     * ``plot_source_offsets`` `(bool, default True)` Plot the RA, Dec offsets for all sources matched with an external catalogue 
     * ``plot_field_offsets`` `(bool, default True)` Plot the image with the intensity showing the relative overall (RA and Dec) offset from the external catalogue, with a Gaussian blur applied to the image
     
The ePSF plot and the residuals plot are measures of the quality of the PSF fit. The correlation is a measure of the accuracy of the PSF calibration: the slope of the linear fit should be very close to 1, although outliers are always present. Finally, the offset plots are measures of the difference between the astrometry of the queried catalogue and our solved image.  

With this step complete, we have calibrated magnitudes for several thousand stars in our image. A table of all of these sources is stored in the attribute ``j_stack.psf_sources``. **Note that in the above steps, sources near the edges of the image are ignored.** To see the border which delimits the sources which are used in photometry: 

.. code-block:: python 

     >>> j_stack.make_image(border=True)

The border used is a circle with a radius equal to the `x` dimension of the image, centered on the image center. This concludes our PSF photometry. To look for a particular source in our list of calibrated magnitudes, we can use: 

.. code-block:: python

     >>> ra = 275.15
     >>> dec = 7.15
     >>> j_stack.source_select(ra, dec)
     
This will return a table containing any source(s) within 1 pixel of the input RA, Dec. This radius can be increased via the optional ``radius`` argument. If we find the source(s) of interest, we can write this table: 

.. code-block:: python

     >>> j_stack.write_selection(ra, dec)

However, our source could easily be too dim or very close to the edges. In this case, we can also do **aperture photometry**. Suppose we know the RA and Dec of the source we care about:

.. code-block:: python 

     >>> j_stack.aperture_photometry(ra, dec)
     
This will do the following: 

     1. Drop an aperture of radius 1.2'' (arcsec) at this RA, Dec and compute the **unsubtracted** flux in this aperture and the area spanned by this aperture
     2. Drop an annulus of inner radius 2.0'' and outer radius 5.0'' at this RA, Dec, compute the median background (with sources masked), and subtract this median background per pixel from the aperture flux 
     3. If this background-subtracted flux is positive, convert into an instrumental magnitude and use the zero point obtained from the previous PSF photometry to convert to a usable catalog magnitude 
     4. Propagate errors and compute a detection sigma 

The aperture radius, inner annulus radius, and outer annulus radius can be set via the optional arguments ``ap_radius``, ``r1``, and ``r2``, respectively. Furthermore, if we want to see the aperture and annulus drawn around the source and/or the data in the annulus only:

.. code-block:: python

     >>> j_stack.aperture_photometry(ra, dec, plot_aperture=True, plot_annulus=True)
     
Will yield these plots. Aperture photometry can be performed as many times as desired. All results are appended to a table stored in the attribute ``j_stack.aperture_sources``. Finally, if neither PSF nor aperture photometry work, we can compute a **limiting magnitude**. For example: 

.. code-block:: python

     >>> j_stack.limiting_magnitude(ra, dec)
     
Will return the magnitude which would be needed for a 3-sigma detection. This sigma can be set using the optional ``sigma`` argument when calling the function. 

------------------------------------------------------------

Let's summarize all the steps we took above with an example. 

.. code-block:: python

     >>> ra, dec = 303.85, 11.06 
     >>> j_stack.astrometry()
     >>> j_stack.PSF_photometry()
     >>> j_stack.source_select(ra, dec)
     
Let's say we get no results from that last line. We decide to try aperture photometry, and plot the region around the source so that we can see what it looks like: 

.. code-block:: python 

     >>> j_stack.aperture_photometry(ra, dec, plot_aperture=True, plot_annulus=True)
     
We get a detection -- but it's only 2-sigma. We decide to get a limiting magnitude: 

.. code-block:: python

     >>> j_stack.limiting_magnitude(ra, dec)
     22.51
     
That's the best we can do. We decide to write out PSF photometry and aperture photometry results to tables anyways:

.. code-block:: python

     >>> j_stack.write_PSF_photometry()
     >>> j_stack.write_aperture_photometry()
     
And that's it. The tables output by these write functions can then be used with ``lightcurve.py``, which is handled in a different section. The above walkthrough was for WIRCam, but the steps are largely unchanged for MegaCam. Happy pipelining!


Additional notes
----------------

**NOTE:** By default, all images are saved as ``png`` files. To change this: 

.. code-block:: python 

     >>> j_stack.set_plot_ext("pdf")

Valid options are ``png``, ``pdf``, ``bmp``, and ``jpg``. 

**NOTE:** The catalogues used to match sources during PSF photometry are the Sloan Digital Sky Survey Data Release 12 (SDSS DR12) for the `u` band, PanStarrs 1 (PS1) for `grizy`, and 2MASS for `JHKs`. 2MASS is an all-sky survey and PS1 is carried out from Hawaii, so it is not an issue to match sources for the `grizy` and `JHKs` bands. However, SDSS is based in New Mexico, so it is possible that a source observed by CFHT is simply nowhere near the regions of the sky observed by SDSS. 

**NOTE:** ``PSF_photometry()`` can take a while for images which contain many sources. For example, the function requires ~ 1000 s to complete for an image with ~ 10 000 sources, **on irulan**. Speed will of course vary from machine to machine, but do not be surprised if this part of the analysis takes ~ an order of magnitude more time than the astrometry. 

--------------------------------------------------------

===================
Making light curves
===================

The script ``lightcurve.py`` is also object-oriented. The script allows you to:

* Read data which has been output by ``write_PSF_photometry()``, ``write_aperture_photometry()``, or ``write_selection()``
* Build a table with the RA, Dec, magnitudes, errors on magnitudes, filters, and MJD of all read sources
* Plot a light curve

To build a ``LightCurve`` object directly by manually inputting the RA, Dec, magnitudes, etc. or from a file/directory:

.. code-block:: python

     >>> import lightcurve
     >>> single_pt_lc = lightcurve.frompoint(300.865, 20.523, 17.7, 0.3, "g", 55950) # from a point
     >>> lc = lightcurve.fromfile("my_data/H_stack_20181106_aperture_photometry.fits") # from a file 
     >>> bigger_lc = lightcurve.fromdirectory("my_data") # from a directory 
     
Alternatively, build the light curve and then read in data:

.. code-block:: python

     >>> lc = lightcurve.LightCurve()
     >>> lc.read("H_stack_20181106_aperture_photometry.fits")

To add new data to an existing ``LightCurve`` object:

     >>> lc.add_fromtables(table1, table2) # add a table/tables
     >>> lc.add_fromfiles("J_stack_20181106_aperture_photometry.fits", "Ks_stack_20181106_aperture_photometry.fits") # a file
     >>> lc.add_point(157.777, 30.789, 18.3, 0.5, "r", 58010) # add a single point 
     
And to add a limiting magnitude, provide the RA, Dec, limiting magnitude, filter, and MJD manually:

.. code-block:: python

     >>> lc.add_limiting_magnitude(250.052, 70.5, 23.5, "i", 57850)
     
Once all of the data has been added to the object as desired, you can write it to a file for later usage. A name for the ouput file must be supplied: 

.. code-block:: python

     >>> lc.write("my-awesome-lightcurve.fits")
     
Finally, to plot your light curve:

.. code-block:: python

     >>> lc.plot()
     
This will, by default, save the file to "lightcurve.png". An alternate filename can be given by the optional argument ``output``. The plot will not have a title, by default. To supply one, use the argument ``title``. Finally, if you want to plot only certain filters, these can be specified. An example: 

.. code-block:: python 

     >>> lc.plot("g","r","i", output="gri_only_lightcurve.png", title="GRB200220 gri Light Curve")
     
And that's it. Happy light-curving!
