# spexwavepy
> An open-source Python package for X-ray wavefront sensing using speckle-based techniques

***Spexwavepy*** is an open-source Python package dedicated to the speckle-based wavefront sensing techniques for X-ray optics. 
Its name is the abbreviation of the **spe**ckle-based **X**-ray **wave**front sensing **Py**thon package.

## Features

* ***spexwavepy*** is dedicated to the speckle-based X-ray wavefront sensing techniques.
* It provides various data processing modes.
* It is written in Python and open-source.
* It uses the built-in multiprocessing module to parallelise the code.
* It has a complete [Documentation](#documentation) for the package.
* It shares the expriment data used for the examples.

## Getting ***spexwavepy***

### 1. From GitHub

This is the **recommended** way of getting this package.
Since in this way, the users will have the example code and 
also the compiled documentation in the form of html files.

Git clone from the GitHub repository.

`git clone https://github.com/wholingfei/spexwavepy.git`

`cd spexwavepy`

`pip install -e .`

Or, if you have difficulties in using `pip install` due to various reasons, 
make sure you have **Numpy**, **Scipy**, **cv2(opencv-python)** and **natsort** 
available, you can use this package without installation as well.

### 2. From PyPI

`pip install spexwavepy`

This can install the package without the provided examples. 

## Documentation

There are two ways of reading the documentation of this package.

If you download the source code of this package from GitHub, you can find the 
local documentation in docs/build/html/ folder. Read them using your web browser.
Further, you can make the html pages from make file.

`make clean`

and 

`make html`

will generate the documentation in html files.

The documentation can also be found online.

The main page of the documention is host at 
[readthedocs.org](https://spexwavepy.readthedocs.io/en/latest/).


## Data availability

All the data used in ***spexwavepy*** is shared on [Zenodo, https://zenodo.org/records/10892838](https://zenodo.org/records/10892838).
Users can download the data and reproduce all the results from the 
example code by themselves.
