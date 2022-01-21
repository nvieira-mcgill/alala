# alala

Overview
========
A pipeline for the MegaCam (a.k.a. MegaPrime) Wide-field Infra-Red Camera (WIRCam) instruments of the Canada-France-Hawaii Telescope (CFHT). Includes the following:

- Stacking (co-addition) of individual exposures
- Obtaining astrometric solutions
- Point-Spread Function (PSF) photometry
- Aperture photometry
- Putting PSF/aperture photometry together to produce light curves

This pipeline was used for all of these steps in the following paper describing our CFHT MegaCam follow-up of the gravitational wave event GW190814:

[Vieira, N., Ruan, J.J, Haggard, D., Drout, M.R. et al. 2020, ApJ, 895, 96, 2. *A Deep CFHT Optical Search for a Counterpart to the Possible Neutron Star - Black Hole Merger GW190814.*](https://ui.adsabs.harvard.edu/abs/2020arXiv200309437V/abstract)  


The code is named after the ʻalalā, the Hawaiian crow.

Documentation
=============

Detailed documentation (WIP) for all modules can be found [here](https://alala.readthedocs.io/en/latest/). In the future, example scripts/notebooks will be added.

Installation
============

Currently, needs to be installed directly from github. May be install-able with ``conda`` and/or ``pip`` in the future.

**Dependencies:**

- ``numpy``
- ``scipy``
- ``matplotlib``
- [``astropy``](https://docs.astropy.org/en/stable/)
- [``astroquery``](https://astroquery.readthedocs.io/en/latest/)
- [``photutils``](https://photutils.readthedocs.io/en/stable/)
- [``pyraf``](https://pypi.org/project/pyraf/) (python wrapper for ``iraf``, only for stacking)
- [``stsci.tools``](https://github.com/spacetelescope/stsci.tools) (only for stacking)

**Non-Python:**

- [``astrometry.net``](http://astrometry.net/use.html)
- [``iraf``](http://ast.noao.edu/data/software) (only for stacking)

The code will be migrated from ``iraf`` to some other software (e.g. ``swarp``) in the future.

Contact
=======
[nicholas.vieira@mail.mcgill.ca](nicholas.vieira@mail.mcgill.ca)

Acknowledgements
================
This project was begun in May 2019 by Nicholas Vieira, working under the supervision of Dr. Daryl Haggard and Dr. John Ruan at the McGill Space Institute. Thanks go to Dr. Daryl Haggard, Dr. John Ruan, and the rest of the Haggard research group. We are very grateful for assistance from Dr. Laurie Rousseau-Nepton, instrument specialist for WIRCam, and Dr. Dustin Lang for his assistance with `astrometry.net`.

