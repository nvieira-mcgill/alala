=====
alala
=====

.. image:: https://img.shields.io/pypi/v/alala.svg
        :target: https://pypi.python.org/pypi/alala

.. image:: https://img.shields.io/travis/nvieira-mcgill/alala.svg
        :target: https://travis-ci.com/nvieira-mcgill/alala

.. image:: https://readthedocs.org/projects/alala/badge/?version=latest
        :target: https://alala.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

A pipeline for the MegaCam (a.k.a. MegaPrime) and Wide-field Infra-Red Camera (WIRCam) instruments of the Canada-France-Hawaii Telescope (CFHT). Includes the following:

- Stacking (co-addition) of individual exposures
- Obtaining astrometric solutions
- Point-Spread Function (PSF) photometry
- Aperture photometry
- Putting PSF/aperture photometry together to produce light curves

The code is named after the ʻalalā, the Hawaiian crow.



Documentation
=============

Detailed documentation (WIP) for all modules can be found [here](https://alala.readthedocs.io/en/latest/). In the future, example scripts/notebooks will be added.



Installation
============

Currently, needs to be installed directly from github. May be install-able with ``conda`` and/or ``pip`` in the future.

The code will be migrated from ``iraf`` to some other software (e.g. ``swarp``) in the future.

**Dependencies:**

- numpy
- scipy
- matplotlib
- astropy_
- photutils_
- astroquery_
- pyraf_ (python wrapper for ``iraf``, for stacking)
- `stsci.tools`_ (for stacking)

**Non-Python:**

- `astrometry.net`_ (can however be ignored in favour of source detection with the image segmentation methods of ``photutils`` instead)
- iraf_ (only for stacking)


Contact
=======

nicholas.vieira@mail.mcgill.ca



Acknowledgements
================
This project was begun in May 2019 by Nicholas Vieira, working under the supervision of Dr. Daryl Haggard and Dr. John Ruan at the McGill Space Institute. Thanks go to Dr. Daryl Haggard, Dr. John Ruan, and the rest of the Haggard research group. We are very grateful for assistance from Dr. Laurie Rousseau-Nepton, instrument specialist for WIRCam, and Dr. Dustin Lang for his assistance with `astrometry.net`.



Credits
=======

Free software: MIT license

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _astropy: https://docs.astropy.org/en/stable/
.. _photutils: https://photutils.readthedocs.io/en/stable/
.. _astroquery: https://astroquery.readthedocs.io/en/latest/
.. _pyraf: https://pypi.org/project/pyraf/
.. _`stsci.tools`: https://github.com/spacetelescope/stsci.tools


.. _`astrometry.net`: http://astrometry.net/use.html
.. _iraf: http://ast.noao.edu/data/software

.. _`Vieira, N., Ruan, J.J, Haggard, D., Drout, M.R. et al. 2020, ApJ, 895, 96, 2. *A Deep CFHT Optical Search for a Counterpart to the Possible Neutron Star - Black Hole Merger GW190814.`: https://ui.adsabs.harvard.edu/abs/2020arXiv200309437V/abstract

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
