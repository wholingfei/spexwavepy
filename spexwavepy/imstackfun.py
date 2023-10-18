r"""
**This module contains the functions used for dealing with the image stacks.**
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
import natsort
import cv2
import scipy.ndimage
import multiprocessing
import copy

sys.path.append(os.path.join('..'))
from spexwavepy.corefun import indicator, read_one, crop_one, NormImage, Imagematch

__DEBUG = True#False#True

class Imagestack:
    """
    A class to represent one image stack.
    

    Parameters
    ----------
    fpath : str
        File folder path.
    roi : 4-element tuple, (int, int, int, int)
        Region of interest.
    fstep : int
        File step for file reading. (default 1)
    fnum : int
        Total file number, if <=0, consider all the files in the folder.
    fstart : int
        Starting file number for dataset loading. (defaul 0)
    verbose : bool
        Print out information or not. (default True)
    rawdata : numpy.ndarray
        3D image stack data. Read-only.
    data : numpy.ndarray
        3D image stack data. Data used for future processing. Initially, it copied the raw image stack.
    normalize : bool
        Do the normalization or not. (default False)
    flip : string 
        To flip the images in the stack or not. 
        'x' is to flip horizontally, 'y' is to flip vertically.
        It is necessary when the test optic is a mirror. (default None)
    """
    def __init__(self, fileFolder, ROI):
        """
        Parameters
        ----------
        fileFolder : str
            Data filefolder name
        ROI : [int, int, int, int]
            ROI is [y_begin, y_end, x_begin, x_end]
        """
        self.fpath = fileFolder
        self.roi = tuple(ROI)
        self.fstep = 1
        self.fnum = -10
        self.fstart = 0
        self.verbose = True
        self.normalize = False
        self.flip = None 
        self.rawdata = None
        self.data = None

    @property
    def fstep(self):
        return self._fstep

    @fstep.setter
    def fstep(self, value):
        self._fstep = value

    @property
    def fnum(self):
        return self._fnum

    @fnum.setter
    def fnum(self, value):
        self._fnum = value

    @property
    def fstart(self):
        return self._fstart

    @fstart.setter
    def fstart(self, value):
        self._fstart = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        self._normalize = value

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, value):
        if value in [None, 'x', 'y']:
            self._flip = value
        else:
            print("self.flip must be 'x', 'y' or None.")
            sys.exit(0)

    def read_data(self):
        """
        Read the raw image data. 
        """
        if self.rawdata is None:
            fileFolder = self.fpath
            file_start = self.fstart
            file_step = self.fstep
            file_num = self.fnum
            fileNames = natsort.natsorted(os.listdir(os.chdir(fileFolder)))
            if file_num <= 0:
                fileNames = fileNames[file_start::file_step]
                self.fnum = len(fileNames)
            else:
                fileNames = fileNames[file_start:file_start+file_num:file_step]
            ###Read the first image. get x_dim, y_dim
            im_one = read_one(fileNames[0], ShowImage=False)
            im_crop = crop_one(im_one, self.roi, ShowImage=False)
            y_dim, x_dim = im_crop.shape

            ####################################################################
            ###Build Image stack
            data = np.empty((len(fileNames), y_dim, x_dim))
            if self.verbose:
                print("Start loading data...")
            for i in range(len(fileNames)):
                data_tmp = read_one(fileNames[i])
                data[i] = crop_one(data_tmp, self.roi)
                if self.verbose:
                    indicator(i, len(fileNames))
            if self.verbose:
                print("Image stack acquired.")

            self.rawdata = data
            self.rawdata.setflags(write=False)
            
            if self.normalize:
                self.norm()
            else:
                self.data = copy.deepcopy(self.rawdata)

            if self.flip is not None:
                self.flipstack()

    def norm(self):
        """
        Normalize the raw images.
        """
        if self.rawdata is not None:
            data = copy.deepcopy(self.rawdata)
            imNo, y_dim, x_dim = data.shape
            if self.verbose:
                print("Start normalization...")
            for i in range(imNo):
                if self.verbose:
                    indicator(i, imNo)
                data_tmp = data[i]
                data[i] = NormImage(data_tmp)
            self.data = data
            if self.verbose:
                print("Data normalized.")
        else:
            print("Please read the raw data first!")
            sys.exit(0)

    def flipstack(self):
        """
        Flip the images in the stack.
        """
        if self.data is not None:
            imNo, _, _ = self.data.shape
            if self.verbose:
                print("Start to flip the images...")
            if self.flip == 'x':
                for k in range(imNo):
                    self.data[k] = np.fliplr(self.data[k])
                    if self.verbose:
                        indicator(k, len(self.data))
            if self.flip == 'y':
                for k in range(imNo):
                    self.data[k] = np.flipud(self.data[k])
                    if self.verbose:
                        indicator(k, len(self.data))
            if self.verbose:
                print("Images flipped.")
        else:
            print("Please read the raw data first!")
            sys.exit(0)

    def rot90deg(self):
        """
        Rotate the raw images in the stack 90 degrees.
        It is very useful when dealing the x scan data.
        """
        if self.rawdata is None:
            self.read_data()
        imNo, y_size, x_size = self.data.shape
        data_new = np.zeros((imNo, x_size, y_size))
        if self.verbose: print("Start rotation...")
        for i in range(imNo):
            data_new[i] = np.rot90(self.data[i])
            if self.verbose:
                indicator(i, imNo)

        self.data = data_new
        if self.verbose:
            print("Image stack rotated.")

    def rotate(self, angle):
        """
        Rotate the raw images in the stack according to the angle.

        Parameters
        ----------
        angle : float
            Rotation angle, in [deg].
        """
        if self.data is None:
            self.read_data()
        imNo, y_dim, x_dim = self.data.shape
        Rot_M = cv2.getRotationMatrix2D((x_dim//2, y_dim//2), angle, 1)
        data_new = np.zeros((imNo, y_dim, x_dim))
        if self.verbose:
            print("Start to rotate the image stack...")
        for i in range(imNo):
            data_new[i] = cv2.warpAffine(self.data[i], Rot_M, (x_dim, y_dim))
            if self.verbose:
                indicator(i, imNo)

        self.data = copy.deepcopy(data_new)

        if self.verbose: print("Image stack rotated.")

    def smooth(self, meth='Gaussian', pixel=0, verbose=False):
        """
        Smoothing the raw images in the image stack.

        Parameters
        ----------
        meth : string
            'Gaussian|Box', method used for smoothing. (default 'Gaussian')
            'Gaussian' is for Gaussian smoothing; 'Box' is for smoothing using
            a n*n square. 
        pixel : int
            If meth is 'Gaussian', it is the sigma; if meth is 'Box', it is 
            the width and height of the square used for smoothing. (default 0) 
        verbose : bool
            To show the information or not. (default False)
        """
        if self.data is None:
            self.read_data()
        if self.verbose:
            print("Start to smooth the image stack...")
        imNo, y_dim, x_dim = self.data.shape
        data_new = np.zeros((imNo, y_dim, x_dim))
        if meth not in ['Gaussian', 'Box']:
            print("Unknown smoothing method. The supported methods are 'Gaussian' and 'Box'.")
            sys.exit(0)
        if meth == 'Gaussian':
            for jc in range(imNo):
                im_tmp = scipy.ndimage.gaussian_filter(self.data[jc], pixel, mode='wrap') 
                data_new[jc] = im_tmp
                if verbose:
                    indicator(jc, imNo)
        if meth == 'Box':
            kernel = np.ones((pixel, pixel), dtype=np.float64) / (pixel**2)
            for jc in range(imNo):
                im_tmp = cv2.filter2D(self.data[jc], -1, kernel)
                data_new[jc] = im_tmp
                if verbose:
                    indicator(jc, imNo)

        self.data = copy.deepcopy(data_new)
        if self.verbose: print("Image stack smoothed.")

    def smooth_multi(self, meth='Gaussian', pixel=0, cpu_no=1, verbose=False):
        """
        The multiprocessing version of :py:func:`~spexwavepy.imstackfun.Imagestack.smooth` function.

        Parameters
        ----------
        meth : string
            'Gaussian|Box', method used for smoothing. (default 'Gaussian')
            'Gaussian' is for Gaussian smoothing; 'Box' is for smoothing using
            a n*n square. 
        pixel : int
            If meth is 'Gaussian', it is the sigma; if meth is 'Box', it is 
            the width and height of the square used for smoothing. (default 0) 
        cpu_no : int
            CPU number to be used.
        verbose : bool
            To show the information or not. (default False)
        """
        if self.data is None:
            self.read_data()
        if self.verbose:
            print("Start to smooth the image stack...")
        imNo, y_dim, x_dim = self.data.shape
        data_new = np.zeros((imNo, y_dim, x_dim))
        jxs = list(np.arange(0, imNo, 1))
        if meth not in ['Gaussian', 'Box']:
            print("Unknown smoothing method. The supported methods are 'Gaussian' and 'Box'.")
            sys.exit(0)
        if meth == 'Gaussian':
            global process_tmp_gaussian
            def process_tmp_gaussian(j):
                if verbose:
                    indicator(j, imNo)
                im_tmp = self.data[j]
                im_tmp_smooth = scipy.ndimage.gaussian_filter(im_tmp, pixel, mode='wrap')

                return im_tmp_smooth

            with multiprocessing.Pool(cpu_no) as pool:
                results = pool.map(process_tmp_gaussian, jxs)

        if meth == 'Box':
            kernel = np.ones((pixel, pixel), dtype=np.float64) / (pixel**2)
            global process_tmp_box
            def process_tmp_box(j):
                if verbose:
                    indicator(j, imNo)
                im_tmp = self.data[j]
                im_tmp_smooth = cv2.filter2D(im_tmp, -1, kernel)

                return im_tmp_smooth

            with multiprocessing.Pool(cpu_no) as pool:
                results = pool.map(process_tmp_box, jxs)

        for i in range(imNo):
            data_new[i] = results[i]

        self.data = copy.deepcopy(data_new)
        if self.verbose: print("Image stack smoothed.")

    def getpixsize(self, subROI, dim, step, verbose=True, display=True):
        """
        To obtain the pixel size from a 1D scan.

        Parameters
        ----------
        subROI : [int, int, int, int]
            ROI of the subregion used for template matching.
            It relates to the cropped image. In other words, 
            the image stack has been cropped using ROI, the 
            subROI is the coordinates of the cropped image 
            from the crooped image stack.
        dim : str
            'x'|'y'|'both'. The data scanned direction. 
        step : float
            The data scanned step size. Unit :math:`\mu m`.
        verbose : bool
            To show the information or not. (default True)
        display : bool
            To display the fitting result or not. (default True)

        Return
        ------
        pixsize : float
            The pixel size of the detector. Unit is :math:`\mu m`.
        """
        if dim not in ['x', 'y', 'both']:
            print("dim should be 'x'|'y'|'both'.")
        self.step = step            # [um]
        imNo, y_size, x_size = self.data.shape
        ixs = np.zeros(imNo - 1)
        iys = np.zeros(imNo - 1)
        res = np.zeros(imNo - 1)
        template = crop_one(self.data[0], subROI)
        if verbose: print("Start tracking...")
        for i in range(imNo-1):
            if verbose:
                print(i)
            ix_tmp, iy_tmp, res_tmp = Imagematch(self.data[i], template) 
            ixs[i] = ix_tmp
            iys[i] = iy_tmp
            res[i] = np.max(res_tmp)
        if verbose: print("Tracking ends.")
        if dim == 'x':
            ixs -= ixs[0]
            x_fit = np.arange(0, len(ixs), 1) * self.step 
            params = np.polyfit(x_fit, ixs, 1)
            pixsize = 1. / np.abs(params[0])        #[um]
            if display:
                fit_func = np.poly1d(params)
                plt.figure()
                plt.plot(x_fit, ixs, 'o')
                plt.plot(x_fit, fit_func(x_fit))
                plt.xlabel('Scan steps ['+r'$\mu$'+'m]')
                plt.ylabel('Pixels')
                plt.title('Pixel size is {:.4f} '.format(pixsize)+r'$\mu$'+'m.')

                plt.figure()
                plt.plot(x_fit/self.step, ixs-fit_func(x_fit))
                plt.xlabel('Scan No.')
                plt.ylabel('Residual error [pixels]')
                plt.title('The RMS is {:.4f} pixels.'.format(np.std(ixs-fit_func(x_fit))))

                plt.show()
        if dim == 'y':
            iys -= iys[0]
            x_fit = np.arange(0, len(iys), 1) * self.step 
            params = np.polyfit(x_fit, iys, 1)
            pixsize = 1. / np.abs(params[0])        #[um]
            if display:
                fit_func = np.poly1d(params)
                plt.figure()
                plt.plot(x_fit, iys, 'o')
                plt.plot(x_fit, fit_func(x_fit))
                plt.xlabel('Scan steps ['+r'$\mu$'+'m]')
                plt.ylabel('Pixels')
                plt.title('Pixel size is {:.4f} '.format(pixsize)+r'$\mu$'+'m.')

                plt.figure()
                plt.plot(x_fit/self.step, iys-fit_func(x_fit))
                plt.xlabel('Scan No.')
                plt.ylabel('Residual error [pixels]')
                plt.title('The RMS is {:.4f} pixels.'.format(np.std(iys-fit_func(x_fit))))

                plt.show()
        if dim == 'both':
            print("Only 1D case is supported now. Please scan in x or y direction.")
            sys.exit(0)

        return pixsize


if __name__ == "__main__":
    if __DEBUG:
        fileFolder = "/dls/science/groups/b16/SpeckleData/planemXSSself/Xscan/354343-pcoedge-files/"
        ROI = [400, 1600, 700, 1250]           #[y_start, y_end, x_start, x_end]
        Imstack_2 = Imagestack(fileFolder, ROI)
        Imstack_2.read_data()
        #Imstack_2.smooth('Gaussian', 5, verbose=True)
        Imstack_2.smooth_multi('Gaussian', 2, cpu_no=16, verbose=True)
        Imstack_3 = Imagestack(fileFolder, ROI)
        Imstack_3.read_data()
        #Imstack_3.smooth('Box', 5, verbose=True)
        Imstack_3.smooth_multi('Box', 5, cpu_no=16, verbose=True)
        sys.exit(0)
        #Imstack_1.norm() 
        subROI = [1500, 2000, 500, 2000]      #[y_start, y_end, x_start, x_end]
        pixsize = Imstack_1.getpixsize(subROI, dim='x', step=10.0)
