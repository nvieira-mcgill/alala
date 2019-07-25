# alala
A pipeline for the Wide-field Infra-Red Camera (WIRCam) and MegaCam instruments at the Canada-France-Hawaii Telescope (CFHT). Does stacking, astrometry, PSF photometry and aperture photometry. 

Named after the ʻAlalā, the Hawaiian crow, known for its intelligence and flying ability.

# Documentation
See: https://alala.readthedocs.io/en/latest/#

# Modules you'll need 
This software makes use of certain modules you may not already have. These are:

`astropy` -- Widely-used software for astrophysical programming. It's recommended to download the whole thing. 

`pyraf` -- An industry standard tool for all sorts of astrophysical programming. Used here for stacking images and applying domeflat corrections. 

`astrometry.net` -- A tool for source detection and astrometric calibration of images (finding world coordinate system solutions).

`astroquery` -- A tool for querying online catalogues. 

`photutils` -- Software affiliated with astropy, used specifically for photometry. It's also recommended to download the whole thing. 

# Where to get these modules

`astropy`: http://docs.astropy.org/en/stable/install.html

`pyraf`: http://www.stsci.edu/institute/software_hardware/pyraf (Installing this correctly can be very difficult. Follow instructions carefully to download via AstroConda).
If you have access to the irulan server of McGill University, iraf will already be installed. If your institution has a server dedicated to physics/astrophysics, it will probably already have iraf as well. 

`astrometry.net`: http://astrometry.net/doc/readme.html#installing

`astroquery`: https://astroquery.readthedocs.io/en/latest/#installation

`photutils`: https://photutils.readthedocs.io/en/stable/install.html

# How to use this software

Documentation and a step-by-step guide on the use of this software can be seen here: https://github.com/nvieira-mcgill/alala/wiki

