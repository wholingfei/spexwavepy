.. spexwavepy documentation master file, created by
   sphinx-quickstart on Thu Apr 20 14:24:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
spexwavepy
==========
:Version: 1.0.0
:Authors: Lingfei Hu (DLS), Hongchang Wang (DLS)
:Dependencies: * **Numpy**
               * **Scipy**
               * **cv2**: This is OpenCV support for python.
               * **natsort**: This package is used for natural sorting.
:E-mail: lingfei.hu@diamond.ac.uk, hongchang.wang@diamond.ac.uk


What is spexwavepy
==================
*Introduction here...*

The speckle-based wavefront sensing techniques
==============================================
*Description here...*

- :doc:`The speckle-based wavefront sensing techniques <principle>`
  

  - :ref:`X-ray Speckle Scanning (XSS) technique with reference beam <prinXSSRefer>`

  - :ref:`Self-reference X-ray Speckle Scanning (XSS) technique <prinXSSSelf>`

  - :ref:`Conventional X-ray Speckle Tracking (XST) technique with reference beam <prinXSTRefer>`

  - :ref:`Self-reference conventional X-ray Speckle Tracking (XST) technique <prinXSTSelf>`

  - :ref:`X-ray Speckle Vector Tracking (XSVT) technique <prinXSVTRefer>`


Getting started
===============
*Description here...*

- :doc:`Getting started <getstart>`

  - :ref:`Installing spexwavepy <install>`

  - :ref:`Tutorial <tutorial>`

    - :ref:`1. Read the image stack <tuimstack>`

    - :ref:`2. Determine the detector pixel size <tudetpix>`

    - :ref:`3. Stability check <tustable>`

    - :ref:`4. Single CRL measurement <tuCRL>`

Examples
========
*Something here...*

- :doc:`Examples <example>`

  - :ref:`Plane mirror measurement with reference beam <expplane>`

  - :ref:`Measurement of the wavefront local curvature after a plane mirror <exp2ndderiv>`

  - :ref:`Mirror slope error curve (1D) reconstructed from the dowmstream setup <iterative>` 

  - :ref:`Comparison between self-reference XSS technique and self-reference XST technique <expxssvsxst>`

  - :ref:`KB mirror alignment using self-reference XST technique <expKBalign>`

User guide
==========
In this page, we will give a detailed explanation of this python package. 
All the functions in each module will be covered.  

- :doc:`User guide <userguide>`
  
  - :ref:`Fundamental algorithm <usealgorithm>`
  
    - :ref:`Cross-correlation <usecrosscorr>`

    - :ref:`Sub-pixel registration <usesubpix>`

      - :ref:`Default method <subdefault>`

      - :ref:`Gaussian peak fitting method <subgauss>`

      - :ref:`Parabola peak fitting method <subpara>`
        
    - :ref:`Image match <useimmatch>`
      
    - :ref:`Auxiliary functions <useauxfunc>`

  - :ref:`Image stack and its functions <useimstackclass>`

    - :ref:`Image stack <useimstack>`
      
    - :ref:`Preprocessing of the images <usepreprocess>`

      - :ref:`Smoothing <usesmooth>`

      - :ref:`Normalization <usenorm>`
        
      - :ref:`Detector pixel size determination <usedetpix>`

  - :ref:`The speckle-based techniques included in Tracking class <usetrack>`

    - :ref:`Stability checking using speckle patterns <trastable>`

    - :ref:`Reference and sample image stacks collimating before tracking <tracolli>`

    - :ref:`XSS technique with reference beam <traXSS>`

    - :ref:`Self-reference XSS technique <traXSSself>`

    - :ref:`XST technique with reference beam <traXSTrefer>`

    - :ref:`Self-reference XST technique <traXSTself>`

    - :ref:`XSVT technique <traXSVTrefer>`

  - :ref:`Post processing of the tracked speckle pattern shifts <postfun>`
    
    - :ref:`Slope reconstruction <slope>`

    - :ref:`Local curvature reconstruction <curvature>`

    - :ref:`2D integration from the slope <integral>`

API reference
=============
- :doc:`API reference <api>`


.. toctree::
   :maxdepth: 2
   :hidden:
   
   principle
   getstart
   example
   userguide
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
