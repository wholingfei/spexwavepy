import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
import natsort
import cv2
import multiprocessing
import copy
import warnings

sys.path.append(os.path.join('..'))
from spexwavepy.imstackfun import Imagestack
from spexwavepy.corefun import _indicator, read_one, crop_one, Imagematch, NormImage
from spexwavepy.corefun import _initDisplay, _contiDisplay
from spexwavepy.postfun import slope_scan, slope_pixel, curv_scan, curv_XST 

__DEBUG = True

class Tracking:
    """
    A container for different speckle tracking algorithms. 
    
    Parameters 
    ----------
    imstack1 : Imagestack class
        The first image stack.
        For XSS technique, it is scanned in x | y direction.
    imstack2 : Imagestack class
        The second image stack.
        For XSS technique, it is scanned in x | y direction.
        Not always necessary. (default None)
    imstack3 : Imagestack class
        The third image stack. 
        For XSS technique, it is scanned in y direction.
        /Not always necessary. (default None)
    imstack4 : Imagestack class
        The fourth image stack. 
        For XSS technique, it is scanned in y direction.
        Not always necessary. (default None)
    dimension : str
        '1D' or '2D'. To do 1D or 2D data processing. (default '2D')
    scandim : str
        'x'|'y'|'xy'|'random'. Scan direction for the image stack. 
        (default 'x')
    mempos : str
        'downstream' or 'upstream'. Use this to define the position of 
        the diffusor in respect to the focus of the optics. 'downstream'
        means the diffuser is placed downstream of the focus. See 
        *User guide: Local curvature reconstruction* for more 
        details of it. (default 'downstream')
    scanstep : float
        scan step size, unit in :math:`\mu m`. (default None)
    pixsize : float
        detector pixel size, unit in :math:`\mu m`. (default None)
    dist : float
        distance from diffuser to detector plane, 
        if it's the downstream mode, ``mempos`` is 'downstream' (default),
        unit in mm. If the diffuser is placed in the upstream,
        ``mempos`` is 'upstream',
        usually it is set to be the distance between the centre 
        of the optic to the detector. (default None)
    subpixelmeth : str
        'default', 'gausspeak' or 'parapeak'. (default 'default'). 
        Method used for subpixel registration.
    delayX : numpy.ndarray
        The tracked 1D/2D shifts in X direction.
    delayY : numpy.ndarray
        The tracked 1D/2D shifts in Y direction.
    resX : numpy.ndarray
        The tracking 1D/2D coeffcient in X direction.
    resY : numpy.ndarray
        The tracking 1D/2D coeffcient in Y direction.
    sloX : numpy.ndarray
        The reconstructed slope error in X direction.
    sloY : numpy.ndarray
        The reconstructed slope error in Y direction.
    curvX : numpy.ndarray
        The reconstructed local curvature in X direction.
    curvY : numpy.ndarray
        The reconstructed local curvature in Y direction.
    """
    def __init__(self, imstack1, imstack2=None, imstack3=None, imstack4=None):
        """
        Parameters
        ----------
        imstack1 : Imagestack class
            The first image stack.
        imstack2 : Imagestack class
            The second image stack. (default None)
        imstack3 : Imagestack class
            The third image stack. (default None)
        imstack4 : Imagestack class
            The fourth image stack. (default None)
        """
        self.imstack1 = imstack1
        self.imstack2 = imstack2
        self.imstack3 = imstack3
        self.imstack4 = imstack4
        self.dimension = '2D'
        self.scandim = 'x' 
        self.mempos = 'downstream'
        self.scanstep = None
        self.pixsize = None
        self.subpixelmeth = 'default'
        self.delayX = None
        self.delayY = None
        self._delayX = None
        self._delayY = None
        self.resX = None
        self.resY = None
        self.sloX = None
        self._sloX = None
        self.sloY = None
        self._sloY = None
        self.curvX = None
        self._curvX = None
        self.curvY = None
        self._curvY = None
        self._flag = None

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        if value not in ['1D', '2D']:
            print("Tracking.dimension needs to be '1D' or '2D'.")
            sys.exit(0)
        else:
            self._dimension = value

    @property
    def scandim(self):
        return self._scandim

    @scandim.setter
    def scandim(self, value):
        if value not in ['x', 'y', 'xy', 'random']:
            print("Unrecognized scan mode. It should be 'x', 'y'or 'xy', or 'random'.")
            sys.exit(0)
        else:
            self._scandim = value

    @property
    def mempos(self):
        return self._mempos

    @mempos.setter
    def mempos(self, value):
        if value not in ['downstream', 'upstream']:
            print("Unrecognized mempos. It should be 'downstream' or 'upstream'.")
            sys.exit(0)
        else:
            self._mempos = value

    @property
    def scanstep(self):
        return self._scanstep

    @scanstep.setter
    def scanstep(self, value):
        self._scanstep = value

    @property
    def pixsize(self):
        return self._pixsize

    @pixsize.setter
    def pixsize(self, value):
        self._pixsize = value

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, value):
        self._dist = value

    @property
    def subpixelmeth(self):
        return self._subpixelmeth

    @subpixelmeth.setter
    def subpixelmeth(self, value):
        if value not in ['default', 'gausspeak', 'parapeak']:
            print("Not supported subpixel registration method.")
            sys.exit(0)
        else:
            self._subpixelmeth= value

    def stability(self, edge_x, edge_y, verbose=True):
        """
        Check the stability using speckle pattern.

        Parameters
        ----------
        edge_x : int, or [int, int]
            Cutting area of the reference image in x direction.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        edge_y : int, or [int, int]
            Cutting area of the reference image in y direction.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        verbose : bool
            To show the information or not. (default True)

        Returns
        -------
        ixs, iys, res : numpy.array
            Shifts in two dimensions and the correlation coefficient.
        """
        if self.imstack1.rawdata is not None:
            imNo, y_size, x_size = self.imstack1.data.shape
            if isinstance(edge_x, int):
                edge_x = (edge_x, edge_x)
            if isinstance(edge_y, int):
                edge_y = (edge_y, edge_y)
            ixs = np.zeros(imNo-1)
            iys = np.zeros(imNo-1)
            res = np.zeros(imNo-1)
            im1 = self.imstack1.data[0]
            for j in range(imNo-1):
                if verbose: _indicator(j, imNo, 'Stability')
                im2 = self.imstack1.data[j+1]
                im2 = im2[edge_y[0]:-edge_y[1], edge_x[0]:-edge_x[1]]
                ix_tmp, iy_tmp, res_tmp = Imagematch(im1, im2, subpixelmeth=self.subpixelmeth)
                ixs[j] = ix_tmp - edge_x[0]
                iys[j] = iy_tmp - edge_y[0]
                res[j] = np.max(res_tmp)

            return ixs, iys, res
        else:
            data_folder = self.imstack1.fpath
            file_start = self.imstack1.fstart
            file_step = self.imstack1.fstep
            fileNum = self.imstack1.fnum
            fileNames = natsort.natsorted(os.listdir(os.chdir(data_folder)))
            if fileNum <= 0:
                fileNames = fileNames[file_start::file_step]
                self.fnum = len(fileNames)
            else:
                fileNames = fileNames[file_start:file_start+fileNum*file_step:file_step]
            im1_tmp = read_one(fileNames[0], ShowImage=False)
            im1 = crop_one(im1_tmp, self.imstack1.roi)
            imNo = len(fileNames)
            if isinstance(edge_x, int):
                edge_x = (edge_x, edge_x)
            if isinstance(edge_y, int):
                edge_y = (edge_y, edge_y)
            ixs = np.zeros(imNo-1)
            iys = np.zeros(imNo-1)
            res = np.zeros(imNo-1)
            for j in range(imNo-1):
                if verbose: _indicator(j, imNo-1, comments='Stability')
                im2_tmp = read_one(fileNames[j+1])
                im2_tmp = crop_one(im2_tmp, self.imstack1.roi)
                y_dim_tmp, x_dim_tmp = im2_tmp.shape
                im2 = im2_tmp[edge_y[0]:y_dim_tmp-edge_y[1], edge_x[0]:x_dim_tmp-edge_x[1]]
                ix_tmp, iy_tmp, res_tmp = Imagematch(im1, im2, subpixelmeth=self.subpixelmeth) 
                ix_tmp -= edge_x[0]
                iy_tmp -= edge_y[0]
                res_tmp = np.max(res_tmp)
                ixs[j] = ix_tmp
                iys[j] = iy_tmp
                res[j] = res_tmp

            return ixs, iys, res


    def stability_multi(self, edge_x, edge_y, cpu_no, verbose=True):
        """
        Multiprocessing version of stability function.

        .. warning:: **BE CAREFUL** to check the available and safe cpu numbers before run this function!!

        Parameters
        ----------
        edge_x : int, or [int, int]
            Cutting area of the reference image in x direction.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        edge_y : int, or [int, int]
            Cutting area of the reference image in y direction.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        cpu_no : int
            Number of CPU to be used. 
        verbose : bool
            To show the information or not. (default True)

        Returns
        -------
        ixs, iys, res : numpy.array
            Shifts in two dimensions and the coefficient.
        """
        data_folder = self.imstack1.fpath
        file_start = self.imstack1.fstart
        file_step = self.imstack1.fstep
        fileNum = self.imstack1.fnum
        fileNames = natsort.natsorted(os.listdir(os.chdir(data_folder)))
        if fileNum <= 0:
            fileNames = fileNames[file_start::file_step]
            self.fnum = len(fileNames)
            fileNum = self.fnum
        else:
            fileNames = fileNames[file_start:file_start+fileNum*file_step:file_step]
        jxs = list(np.arange(0, fileNum-1, 1))
        if isinstance(edge_x, int):
            edge_x = (edge_x, edge_x)
        if isinstance(edge_y, int):
            edge_y = (edge_y, edge_y)
        ixs = np.zeros(fileNum-1)
        iys = np.zeros(fileNum-1)
        res = np.zeros(fileNum-1)
        global process_tmp
        def process_tmp(j):
            if verbose:
                #The indications may not appear in correct order, however, it may still be useful to keep outputing some information.
                _indicator(j, len(jxs), comments='Stability') 
            im1_tmp = read_one(fileNames[0])
            im1 = crop_one(im1_tmp, self.imstack1.roi)
            im2_tmp = read_one(fileNames[j+1])
            im2_tmp = crop_one(im2_tmp, self.imstack1.roi)
            y_dim_tmp, x_dim_tmp = im2_tmp.shape
            im2 = im2_tmp[edge_y[0]:y_dim_tmp-edge_y[1], edge_x[0]:x_dim_tmp-edge_x[1]]
            ix_tmp, iy_tmp, res_tmp = Imagematch(im1, im2, subpixelmeth=self.subpixelmeth) 
            ix_tmp -= edge_x[0]
            iy_tmp -= edge_y[0]
            res_tmp = np.max(res_tmp)

            return ix_tmp, iy_tmp, res_tmp

        with multiprocessing.Pool(cpu_no) as pool:
            results = pool.map(process_tmp, jxs)

        for i in range(len(results)):
            ixs[i] = results[i][0]
            iys[i] = results[i][1]
            res[i] = results[i][2]

        return ixs, iys, res

    def collimate(self, edge_x, edge_y):
        """
        This function uses the first image from both imstack1 and imstack2
        to align the speckle patterns in the image stacks. 
        It is called before any further tracking method.

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        """
        if isinstance(edge_x, int):
            edge_x = (edge_x, edge_x)
        if isinstance(edge_y, int):
            edge_y = (edge_y, edge_y)
        subpixelmeth = self.subpixelmeth 

        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
            self.imstack2.read_data()
        im_temp = self.imstack1.data[0]
        im_ref = self.imstack2.data[0]
        y_size_tmp, x_size_tmp = im_temp.shape
        im_temp = im_temp[edge_y[0]:y_size_tmp-edge_y[1], edge_x[0]:x_size_tmp-edge_x[1]]
        ix_tmp, iy_tmp, res_tmp = Imagematch(im_ref, im_temp, subpixelmeth=subpixelmeth)
        ix_tmp -= edge_x[0]
        iy_tmp -= edge_y[0]
        ix_tmp = round(ix_tmp)
        iy_tmp = round(iy_tmp)

        if np.max(res_tmp) < 0.5:
            warnings.warn("Too low correlate coefficient, may have wrong tracking value!")
        
        print("The coefficient is {:.3f}".format(np.max(res_tmp)))

        if iy_tmp > 0:
            data1_tmp = self.imstack1.data[0]
            y_tmp1, _ = data1_tmp.shape
            data2_tmp = self.imstack2.data[0, iy_tmp:, :]
            y_tmp2, _ = data2_tmp.shape
            y_tmp = min(y_tmp1, y_tmp2)
            self.imstack1.data = self.imstack1.data[:, 0:y_tmp, :]
            self.imstack2.data = self.imstack2.data[:, iy_tmp:iy_tmp+y_tmp, :]
        else:
            data1_tmp = self.imstack1.data[0, np.abs(iy_tmp):, :]
            y_tmp1, _ = data1_tmp.shape
            data2_tmp = self.imstack2.data[0]
            y_tmp2, _ = data2_tmp.shape
            y_tmp = min(y_tmp1, y_tmp2)
            self.imstack1.data = self.imstack1.data[:, np.abs(iy_tmp):np.abs(iy_tmp)+y_tmp, :]
            self.imstack2.data = self.imstack2.data[:, 0:y_tmp, :]
        if ix_tmp > 0:
            data1_tmp = self.imstack1.data[0]
            _, x_tmp1 = data1_tmp.shape
            data2_tmp = self.imstack2.data[0, :, ix_tmp:]
            _, x_tmp2 = data2_tmp.shape
            x_tmp = min(x_tmp1, x_tmp2)
            self.imstack1.data = self.imstack1.data[:, :, 0:x_tmp]
            self.imstack2.data = self.imstack2.data[:, :, ix_tmp:ix_tmp+x_tmp]
        else:
            data1_tmp = self.imstack1.data[0, :, np.abs(ix_tmp):]
            _, x_tmp1 = data1_tmp.shape
            data2_tmp = self.imstack2.data[0]
            _, x_tmp2 = data2_tmp.shape
            x_tmp = min(x_tmp1, x_tmp2)
            self.imstack1.data = self.imstack1.data[:, :, np.abs(ix_tmp):np.abs(ix_tmp)+x_tmp]
            self.imstack2.data = self.imstack2.data[:, :, 0:x_tmp]

        return

    def _XSS_2stacks_1D(self, edge_xy, edge_z, normalize=False, display=False, verbose=True, _Resreturn=False):
        """
        **1D** speckle tracking for XSS technique with reference beam.

        Returns
        -------
        ixs, iys, resmax : numpy.array
            Shifts in x and y and the coefficient if _Resreturn=True.
        """
        scandim = self.scandim
        if scandim not in ['x', 'y']:
            print("Unrecognized 1D scan mode. The supported methods are 1D X and 1D Y scan.")
            sys.exit(0)
        if isinstance(edge_xy, int):
            edge_xy = (edge_xy, edge_xy)
        if isinstance(edge_z, int):
            edge_z = (edge_z, edge_z)
        subpixelmeth = self.subpixelmeth 
        imNo, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
        self.imstack1.data = self.imstack1.data[edge_z[0]:imNo-edge_z[1], :, edge_xy[0]:x_dim_tmp-edge_xy[1]]
        y_dim = y_dim_tmp
        ixs = np.zeros(y_dim)
        iys = np.zeros(y_dim)
        resmax = np.zeros(y_dim)
        plane1_init = self.imstack1.data[:, 0, :].astype(np.float32)
        plane2_init = self.imstack2.data[:, 0, :].astype(np.float32)
        height, width = plane1_init.shape
        if normalize:
            plane1_init = NormImage(plane1_init) 
            plane2_init = NormImage(plane2_init) 
        ix_init, iy_init, res_init = Imagematch(plane2_init, plane1_init, subpixelmeth=subpixelmeth)

        if display:
            fig, h1, h2, h3, h4 = _initDisplay(plane1_init, plane2_init, res_init)

        for i in range(y_dim):
            if verbose: 
                if scandim == 'x':
                    _indicator(i, y_dim, comments = self._flag + ' technique in X direction')
                if scandim == 'y':
                    _indicator(i, y_dim, comments = self._flag + ' technique in Y direction')
            plane1 = self.imstack1.data[:, i, :].astype(np.float32)
            plane2 = self.imstack2.data[:, i, :].astype(np.float32)
            if normalize:
                plane1 = NormImage(plane1)
                plane2 = NormImage(plane2)
            ix_tmp, iy_tmp, res_tmp = Imagematch(plane2, plane1, subpixelmeth=subpixelmeth) 
            ixs[i] = ix_tmp - edge_xy[0]
            iys[i] = iy_tmp - edge_z[0]
            resmax[i] = np.max(res_tmp)
            if display:
                _contiDisplay(fig, h1, h2, h3, h4, plane1, plane2, res_tmp)

        if display and self.dimension == '2D': plt.close('all')

        if scandim == 'y':
            self.delayY = iys
            self._delayX = ixs
            self.resY = resmax
        if scandim == 'x':
            self.delayX = np.flip(iys)
            self._delayY = np.flip(ixs)
            self.resX = resmax

        if _Resreturn:
            return ixs, iys, resmax

    def XSS_withrefer(self, edge_x, edge_y, edge_z, hw_xy=None, pad_xy=None, normalize=False, display=False, verbose=True):
        """
        Speckle tracking for XSS technique with reference beam.
        Two image stacks are needed to define the Tracking class.
        The fisrt image stack is the one with test optic.
        The second image stack is the reference image stack.

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        edge_z : int, or [int, int]
            Area needs to be cut in scan number dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
        hw_xy : int
            The width/height of the image subregion. If Tracking.scandim is 'x',
            it is the height of the subregion; if Tracking.scandim is 'y',
            it is the width of the subregion.
            Needed when do 2D data processing. (default None) 
        pad_xy : int, or [int, int]
            It defines the extra part the reference image needed to do the tracking.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
            Needed when do 2D data processing. (default None)
        normalize : bool
            To normalize the stitched image or not. (default False)
        display : bool
            To display or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        scandim = self.scandim
        if self._flag is None: self._flag = 'XSS(with reference)'
        width = hw_xy
        if scandim not in ['x', 'y', 'xy']:
            print("Unrecognized scan mode. It should be 'x', 'y' or 'xy'.")
            sys.exit(0)
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if scandim == 'xy' and self.imstack3 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if scandim == 'xy' and self.imstack4 == None:
            print("Please provide another image stack.")
            sys.exit(0)

        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()
        # For 'xy' case, ROI should be a square, for 2D integaration 
        if scandim == 'xy': 
            if self.imstack1.data.shape[1] != self.imstack1.data.shape[2] \
                    or self.imstack2.data.shape[1] != self.imstack2.data.shape[2]: 
                        print("For scandim is 'xy', the ROI should be a square, please re-select.")
                        sys.exit(0)
        #if scandim == 'diag':
        #    self.imstack3 = copy.deepcopy(self.imstack1)
        #    self.imstack4 = copy.deepcopy(self.imstack2)
        if scandim == 'x': edge_x = None
        if scandim == 'y': edge_y = None
        if isinstance(edge_z, int):
            edge_z = (edge_z, edge_z)
        if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
            verbose_tmp1 = self.imstack1.verbose
            verbose_tmp2 = self.imstack2.verbose
            self.imstack1.verbose = False
            self.imstack2.verbose = False
            self.imstack1.rot90deg()
            self.imstack2.rot90deg()
            self.imstack1.verbose = verbose_tmp1
            self.imstack2.verbose = verbose_tmp2

        if self.dimension == '1D':
            if scandim == 'y':
                if isinstance(edge_x, int):
                    edge_xy = (edge_x, edge_x)
                else:
                    edge_xy = edge_x
            if scandim == 'x':
                if isinstance(edge_y, int):
                    edge_xy = (edge_y, edge_y)
                else:
                    edge_xy = edge_y 
            self._XSS_2stacks_1D(edge_xy, edge_z, normalize, display, verbose)
            if self.delayX is not None: self.sloX = slope_scan(self.delayX, self.pixsize, self.dist)
            if self.delayY is not None: self.sloY = slope_scan(self.delayY, self.pixsize, self.dist)

        if self.dimension == '2D':
            subpixelmeth = self.subpixelmeth 
            if scandim == 'y':
                if isinstance(edge_x, int):
                    edge_xy = (edge_x, edge_x)
                else:
                    edge_xy = edge_x
            if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
                if isinstance(edge_y, int):
                    edge_xy = (edge_y, edge_y)
                else:
                    edge_xy = edge_y
            if isinstance(pad_xy, int):
                pad_xy = (pad_xy, pad_xy)
            if scandim == 'xy':
                if edge_x != edge_y or edge_xy[0] != edge_xy[1]: 
                    print("For scandim == 'xy', the edges should be symmetrical, edge_x should be the same as edge_y, and also the elements of each.")
                    sys.exit(0)
            if scandim == 'xy':
                if pad_xy[0] != pad_xy[1]: 
                    print("For scandim == 'xy', the pad should be symmetrical.")
                    sys.exit(0)
            if pad_xy[0] > edge_xy[0] or pad_xy[1] > edge_xy[1]:
                print("pad_xy should not be greater than edge_xy.")
                sys.exit(0)
            imNo, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
            self.imstack1.data = self.imstack1.data[edge_z[0]:imNo-edge_z[1], :, edge_xy[0]:x_dim_tmp-edge_xy[1]]
            imNo_tmp, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
            jxs = np.arange(0, x_dim_tmp-width+1, 1)
            delayY_2D = np.empty((y_dim_tmp, len(jxs)))
            delayX_2D = np.empty((y_dim_tmp, len(jxs)))
            res_2D = np.empty((y_dim_tmp, len(jxs)))
            imstack1_data = copy.deepcopy(self.imstack1.data)
            imstack2_data = copy.deepcopy(self.imstack2.data)
            #if self.scandim == 'xy' or self.scandim == 'diag': self.scandim = 'x'
            if self.scandim == 'xy': self.scandim = 'x'
            for index, jx in enumerate(jxs):
                if verbose:
                    _indicator(jx, len(jxs), comments = self._flag + ' technique in X direction')
                self.imstack1.data = imstack1_data[:, :, jx:width+jx]
                self.imstack2.data = imstack2_data[:, :, jx+edge_xy[0]-pad_xy[0]:width+jx+edge_xy[0]+pad_xy[1]]
                edge_xy_new = 0
                edge_z_new = 0
                ix_new, iy_new, res_new = self._XSS_2stacks_1D(edge_xy_new, edge_z_new, normalize, display, verbose=False, _Resreturn=True)
                delayX_2D[:, index] = ix_new - pad_xy[0]
                delayY_2D[:, index] = iy_new - edge_z[0]
                res_2D[:, index] = res_new

            if scandim == 'y':
                self.delayY = delayY_2D
                self._delayX = delayX_2D
                self.resY = res_2D
                self.sloY = slope_scan(self.delayY, self.scanstep, self.dist)
                self._sloX = slope_pixel(self._delayX, self.pixsize, self.dist)

            if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
                self.delayX = np.rot90(delayY_2D, k=-1)
                self._delayY = np.rot90(delayX_2D, k=-1)
                self.resX = np.rot90(res_2D, k=-1)
                self.sloX = slope_scan(self.delayX, self.scanstep, self.dist)
                self._sloY = slope_pixel(self._delayY, self.pixsize, self.dist)

            if scandim == 'xy': #or scandim == 'diag':
                if self.imstack3.rawdata is None:
                    self.imstack3.read_data()
                if self.imstack4.rawdata is None:
                    self.imstack4.read_data()
            # For 'xy' case, ROI should be a square, for 2D integaration 
            if scandim == 'xy': 
                if self.imstack3.data.shape[1] != self.imstack3.data.shape[2] \
                        or self.imstack4.data.shape[1] != self.imstack4.data.shape[2]: 
                            print("For scandim is 'xy', the ROI should be a square, please re-select.")
                            sys.exit(0)
                if isinstance(edge_x, int):
                    edge_xy = (edge_x, edge_x)
                else:
                    edge_xy = edge_x
                if pad_xy[0] > edge_xy[0] or pad_xy[1] > edge_xy[1]:
                    print("pad_xy should not be greater than edge_xy.")
                    sys.exit(0)
                imNo2, y_dim_tmp2, x_dim_tmp2 = self.imstack3.data.shape
                self.imstack3.data = self.imstack3.data[edge_z[0]:imNo2-edge_z[1], :, edge_xy[0]:x_dim_tmp2-edge_xy[1]]
                imNo_tmp2, y_dim_tmp2, x_dim_tmp2 = self.imstack3.data.shape
                jxs2 = np.arange(0, x_dim_tmp2-width+1, 1)
                delayY_2D_2 = np.empty((y_dim_tmp2, len(jxs2)))
                delayX_2D_2 = np.empty((y_dim_tmp2, len(jxs2)))
                res_2D_2 = np.empty((y_dim_tmp2, len(jxs2)))
                imstack1_data = copy.deepcopy(self.imstack1.data)
                imstack2_data = copy.deepcopy(self.imstack2.data)
                imstack3_data = copy.deepcopy(self.imstack3.data)
                imstack4_data = copy.deepcopy(self.imstack4.data)
                self.scandim = 'y'
                for index, jx in enumerate(jxs2):
                    if verbose:
                        _indicator(jx, len(jxs), comments = self._flag + ' technique in Y direction')
                    self.imstack1.data = imstack3_data[:, :, jx:width+jx]
                    self.imstack2.data = imstack4_data[:, :, jx+edge_xy[0]-pad_xy[0]:width+jx+edge_xy[0]+pad_xy[1]]
                    edge_xy_new = 0
                    edge_z_new = 0
                    ix_new2, iy_new2, res_new2 = self._XSS_2stacks_1D(edge_xy_new, edge_z_new, normalize, display, verbose=False, _Resreturn=True)
                    delayX_2D_2[:, index] = ix_new2 - pad_xy[0]
                    delayY_2D_2[:, index] = iy_new2 - edge_z[0]
                    res_2D_2[:, index] = res_new2

                self.scandim = scandim
                self.delayY = delayY_2D_2
                self._delayX = delayX_2D_2
                self.resY = res_2D_2
                self.imstack1.data = imstack1_data
                self.imstack2.data = imstack2_data

                ### To cut the 2D map (align the results in two directions) for post-processing
                if isinstance(edge_x, int):
                    edge_x = (edge_x, edge_x)
                if isinstance(edge_y, int):
                    edge_y = (edge_y, edge_y)
                self.delayX = self.delayX[:, edge_x[0]:-edge_x[1]-width+1]
                self._delayY = self._delayY[:, edge_x[0]:-edge_x[1]-width+1]
                self.delayY = self.delayY[edge_y[0]:-edge_y[1]-width+1, :]
                self._delayX = self._delayX[edge_y[0]:-edge_y[1]-width+1, :]
                self.sloX = slope_scan(self.delayX, self.scanstep, self.dist)
                self.sloY = slope_scan(self.delayY, self.scanstep, self.dist)
                #if scandim != 'diag':
                self._sloX = slope_pixel(self._delayX, self.pixsize, self.dist)
                self._sloY = slope_pixel(self._delayY, self.pixsize, self.dist)

        return
    
    def XSS_withrefer_multi(self, edge_x, edge_y, edge_z, hw_xy, pad_xy, cpu_no, normalize=False, verbose=True):
        """
        Speckle tracking for XSS technique with reference beam.
        The fisrt image stack is the one with test optic.
        The second image stack is the reference image stack.

        .. warning:: **BE CAREFUL** to check the available and safe cpu numbers before run this function!!

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        edge_z : int, or [int, int]
            Area needs to be cut in scan number dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
        hw_xy : int
            The width/height of the image subregion. If Tracking.scandim is 'x',
            it is the height of the subregion; if Tracking.scandim is 'y',
            it is the width of the subregion.
            Needed when do 2D data processing. (default None) 
        pad_xy : int, or [int, int]
            It defines the extra part the reference image needed to do the tracking.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
            Needed when do 2D data processing. (default None)
        cpu_no : int
            The number of CPUs that is available.
        normalize : bool
            To normalize the stitched image or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        if self._flag is None: self._flag = 'XSS(with reference)'
        width = hw_xy
        if width % 2 != 0: width += 1       # width should be even
        scandim = self.scandim
        if scandim not in ['x', 'y', 'xy']:
            print("Unrecognized scan mode. It should be 'x', 'y' or 'xy'.")
            sys.exit(0)
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if scandim == 'xy' and self.imstack3 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()
        # For 'xy' case, ROI should be a square, for 2D integaration 
        if scandim == 'xy': 
            if self.imstack1.data.shape[1] != self.imstack1.data.shape[2] \
                    or self.imstack2.data.shape[1] != self.imstack2.data.shape[2]: 
                        print("For scandim is 'xy', the ROI should be a square, please re-select.")
                        sys.exit(0)
        #if scandim == 'diag':
        #    self.imstack3 = copy.deepcopy(self.imstack1)
        #    self.imstack4 = copy.deepcopy(self.imstack2)
        if scandim == 'x': edge_x = None
        if scandim == 'y': edge_y = None
        if isinstance(edge_z, int):
            edge_z = (edge_z, edge_z)
        if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
            verbose_tmp1 = self.imstack1.verbose
            verbose_tmp2 = self.imstack2.verbose
            self.imstack1.verbose = False
            self.imstack2.verbose = False
            self.imstack1.rot90deg()
            self.imstack2.rot90deg()
            self.imstack1.verbose = verbose_tmp1
            self.imstack2.verbose = verbose_tmp2

        if self.dimension == '1D':
            print("No need to use multiprocessing for 1D analysis.")
            sys.exit(0)

        if self.dimension == '2D':
            subpixelmeth = self.subpixelmeth 
            if scandim == 'y':
                if isinstance(edge_x, int):
                    edge_xy = (edge_x, edge_x)
                else:
                    edge_xy = edge_x
            if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
                if isinstance(edge_y, int):
                    edge_xy = (edge_y, edge_y)
                else:
                    edge_xy = edge_y
            if isinstance(pad_xy, int):
                pad_xy = (pad_xy, pad_xy)
            if scandim == 'xy':
                if edge_x != edge_y or edge_xy[0] != edge_xy[1]:
                    print("For scandim == 'xy', the edges should be symmetrical, edge_x should be the same as edge_y, and also the elements of each.")
                    sys.exit(0)
            if scandim == 'xy':
                if pad_xy[0] != pad_xy[1]:
                    print("For scandim == 'xy', the pad should be symmetrical, pad_x should be the same as pad_y, and also the elements of each.")
                    sys.exit(0)
            if pad_xy[0] > edge_xy[0] or pad_xy[1] > edge_xy[1]:
                print("pad_xy should not be greater than edge_xy.")
                sys.exit(0)
            imNo, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
            self.imstack1.data = self.imstack1.data[edge_z[0]:imNo-edge_z[1], :, edge_xy[0]:x_dim_tmp-edge_xy[1]]
            imNo_tmp, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
            jxs = np.arange(0, x_dim_tmp-width+1, 1)
            delayY_2D = np.empty((y_dim_tmp, len(jxs)))
            delayX_2D = np.empty((y_dim_tmp, len(jxs)))
            res_2D = np.empty((y_dim_tmp, len(jxs)))
            imstack1_data = copy.deepcopy(self.imstack1.data)
            imstack2_data = copy.deepcopy(self.imstack2.data)
            #if self.scandim == 'xy' or self.scandim == 'diag': self.scandim = 'x'
            if self.scandim == 'xy': self.scandim = 'x'
            jxs1 = list(jxs)
            global process_tmp1
            def process_tmp1(jx):
                if verbose:
                    if self.scandim == 'x':
                        _indicator(jx, len(jxs1), comments = self._flag + ' technique in X direction')
                    if self.scandim == 'y':
                        _indicator(jx, len(jxs1), comments = self._flag + ' technique in Y direction')
                self.imstack1.data = imstack1_data[:, :, jx:width+jx]
                self.imstack2.data = imstack2_data[:, :, jx+edge_xy[0]-pad_xy[0]:width+jx+edge_xy[0]+pad_xy[1]]
                edge_xy_new = 0
                edge_z_new = 0
                ix_new1, iy_new1, res_new1 = self._XSS_2stacks_1D(edge_xy_new, edge_z_new, normalize, display=False, verbose=False, _Resreturn=True)
                ix_new1 -= pad_xy[0]
                iy_new1 -= edge_z[0]

                return ix_new1, iy_new1, res_new1

            with multiprocessing.Pool(cpu_no) as pool:
                results = pool.map(process_tmp1, jxs1)
            for i in range(len(results)):
                delayX_2D[:, i] = results[i][0]
                delayY_2D[:, i] = results[i][1]
                res_2D[:, i] = results[i][2]

            if scandim == 'y':
                self.delayY = delayY_2D
                self._delayX = delayX_2D
                self.resY = res_2D
                self.sloY = slope_scan(self.delayY, self.scanstep, self.dist)
                self._sloX = slope_pixel(self._delayX, self.pixsize, self.dist)

            if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
                self.delayX = np.rot90(delayY_2D, k=-1)
                self._delayY = np.rot90(delayX_2D, k=-1)
                self.resX = np.rot90(res_2D, k=-1)
                if scandim == 'x':
                    self.sloX = slope_scan(self.delayX, self.scanstep, self.dist)
                    self._sloY = slope_pixel(self._delayY, self.pixsize, self.dist)

            if scandim == 'xy': #or scandim == 'diag':
                if self.imstack3.rawdata is None:
                    self.imstack3.read_data()
                if self.imstack4.rawdata is None:
                    self.imstack4.read_data()
                # For 'xy' case, ROI should be a square, for 2D integaration 
                if self.imstack3.data.shape[1] != self.imstack3.data.shape[2] \
                        or self.imstack4.data.shape[1] != self.imstack4.data.shape[2]: 
                            print("For scandim is 'xy', the ROI should be a square, please re-select.")
                            sys.exit(0)
                if isinstance(edge_x, int):
                    edge_xy = (edge_x, edge_x)
                else:
                    edge_xy = edge_x
                if pad_xy[0] > edge_xy[0] or pad_xy[1] > edge_xy[1]:
                    print("pad_xy should not be greater than edge_xy.")
                    sys.exit(0)
                imNo2, y_dim_tmp2, x_dim_tmp2 = self.imstack3.data.shape
                self.imstack3.data = self.imstack3.data[edge_z[0]:imNo2-edge_z[1], :, edge_xy[0]:x_dim_tmp2-edge_xy[1]]
                imNo_tmp2, y_dim_tmp2, x_dim_tmp2 = self.imstack3.data.shape
                jxs2 = np.arange(0, x_dim_tmp2-width+1, 1)
                delayY_2D_2 = np.empty((y_dim_tmp2, len(jxs2)))
                delayX_2D_2 = np.empty((y_dim_tmp2, len(jxs2)))
                res_2D_2 = np.empty((y_dim_tmp2, len(jxs2)))
                imstack1_data = copy.deepcopy(self.imstack1.data)
                imstack2_data = copy.deepcopy(self.imstack2.data)
                imstack3_data = copy.deepcopy(self.imstack3.data)
                imstack4_data = copy.deepcopy(self.imstack4.data)
                self.scandim = 'y'
                jxs2 = list(jxs2)
                global process_tmp2
                def process_tmp2(jx):
                    if verbose:
                        _indicator(jx, len(jxs2), comments = self._flag + ' technique in Y direction')
                    self.imstack1.data = imstack3_data[:, :, jx:width+jx]
                    self.imstack2.data = imstack4_data[:, :, jx+edge_xy[0]-pad_xy[0]:width+jx+edge_xy[0]+pad_xy[1]]
                    edge_xy_new = 0
                    edge_z_new = 0
                    ix_new2, iy_new2, res_new2 = self._XSS_2stacks_1D(edge_xy_new, edge_z_new, normalize, display=False, verbose=False, _Resreturn=True)
                    ix_new2 -= pad_xy[0]
                    iy_new2 -= edge_z[0]

                    return ix_new2, iy_new2, res_new2

                with multiprocessing.Pool(cpu_no) as pool:
                    results = pool.map(process_tmp2, jxs2)
                for i in range(len(results)):
                    delayX_2D_2[:, i] = results[i][0]
                    delayY_2D_2[:, i] = results[i][1]
                    res_2D_2[:, i] = results[i][2]

                self.imstack1.data = imstack1_data
                self.imstack2.data = imstack2_data

                self.scandim = scandim
                self.delayY = delayY_2D_2
                self._delayX = delayX_2D_2
                self.resY = res_2D_2

                ### To cut the 2D map (align the results in two directions) for post-processing
                if isinstance(edge_x, int):
                    edge_x = (edge_x, edge_x)
                if isinstance(edge_y, int):
                    edge_y = (edge_y, edge_y)
                self.delayX = self.delayX[:, edge_x[0]+width//2:-edge_x[1]-width//2+1]
                self._delayY = self._delayY[:, edge_x[0]+width//2:-edge_x[1]-width//2+1]
                self.delayY = self.delayY[edge_y[0]+width//2:-edge_y[1]-width//2+1, :]
                self._delayX = self._delayX[edge_y[0]+width//2:-edge_y[1]-width//2+1, :]
                self.sloX = slope_scan(self.delayX, self.scanstep, self.dist)
                self.sloY = slope_scan(self.delayY, self.scanstep, self.dist)
                #if scandim != 'diag':
                self._sloX = slope_pixel(self._delayX, self.pixsize, self.dist)
                self._sloY = slope_pixel(self._delayY, self.pixsize, self.dist)

        return

    def XSS_self(self, edge_x, edge_y, edge_z, nstep, hw_xy=None, pad_xy=None, normalize=False, display=False, verbose=True):
        """
        Speckle tracking for self-reference XSS technique.
        Only one image stack, the one with test optic, is needed.

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        edge_z : int, or [int, int]
            Area needs to be cut in scan number dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
        nstep : int
            The space between two chosen columns or rows.
        hw_xy : int
            The width/height of the image subregion. If Tracking.scandim is 'x',
            it is the height of the subregion; if Tracking.scandim is 'y',
            it is the width of the subregion.
            Needed when do 2D data processing. (default None) 
        pad_xy : int, or [int, int]
            It defines the extra part the reference image needed to do the tracking.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
            Needed when do 2D data processing. (default None)
        normalize : bool
            To normalize the stitched image or not. (default False)
        display : bool
            To display or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        self._flag = "Self-reference XSS"
        width = hw_xy
        scandim = self.scandim
        if scandim not in ['x', 'y', 'xy']:
            print("Unrecognized scan mode. It should be 'x', 'y' or 'xy'.")
            sys.exit(0)
        if scandim == 'xy':
            print("This method needs to be implemented. Try to use 'x' and 'y' seperately.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2 is None:
            self.imstack2 = copy.deepcopy(self.imstack1)
        else:
            print("Only one image stack is needed. Please check your input.")
            sys.exit(0)
        imNo, y_dim, x_dim = self.imstack1.data.shape
        if scandim == 'y': 
            self.imstack1.data = self.imstack1.data[:, nstep:, :]
            self.imstack2.data = self.imstack2.data[:, 0:y_dim-nstep, :]
        if scandim == 'x': 
            self.imstack1.data = self.imstack1.data[:, :, nstep:]
            self.imstack2.data = self.imstack2.data[:, :, 0:x_dim-nstep]

        self.XSS_withrefer(edge_x, edge_y, edge_z, width, pad_xy, normalize=normalize, display=display, verbose=verbose)
        self.sloX = None
        self.sloY = None
        if scandim == 'x':
            self.curvX = curv_scan(self.delayX, self.scanstep, self.dist, self.pixsize, nstep, self.mempos)
        if scandim == 'y':
            self.curvY = curv_scan(self.delayY, self.scanstep, self.dist, self.pixsize, nstep, self.mempos)

        return

    def XSS_self_multi(self, edge_x, edge_y, edge_z, nstep, hw_xy, pad_xy, cpu_no, normalize=False, verbose=True):
        """
        Speckle tracking for self-reference XSS technique. 
        Only one image stack, the one with test optic, is needed.

        .. warning:: **BE CAREFUL** to check the available and safe cpu numbers before run this function!!

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        edge_z : int, or [int, int]
            Area needs to be cut in scan number dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
        nstep : int
            The space between two chosen columns or rows.
        hw_xy : int
            The width/height of the image subregion. If Tracking.scandim is 'x',
            it is the height of the subregion; if Tracking.scandim is 'y',
            it is the width of the subregion.
            Needed when do 2D data processing.
        pad_xy : int, or [int, int]
            It defines the extra part the reference image needed to do the tracking.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
            Needed when do 2D data processing. 
        cpu_no : int
            The number of CPUs that is available.
        normalize : bool
            To normalize the stitched image or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        self._flag = "Self-reference XSS"
        width = hw_xy
        scandim = self.scandim
        if scandim not in ['x', 'y', 'xy']:
            print("Unrecognized scan mode. It should be 'x', 'y' or 'xy'.")
            sys.exit(0)
        if scandim == 'xy':
            print("This method needs to be implemented. Try to use 'x' and 'y' seperately.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2 is None:
            self.imstack2 = copy.deepcopy(self.imstack1)
        else:
            print("Only one image stack is needed. Please check your input.")
            sys.exit(0)
        imNo, y_dim, x_dim = self.imstack1.data.shape
        if scandim == 'y': 
            self.imstack1.data = self.imstack1.data[:, nstep:, :]
            self.imstack2.data = self.imstack2.data[:, 0:y_dim-nstep, :]
        if scandim == 'x': 
            self.imstack1.data = self.imstack1.data[:, :, nstep:]
            self.imstack2.data = self.imstack2.data[:, :, 0:x_dim-nstep]

        self.XSS_withrefer_multi(edge_x, edge_y, edge_z, width, pad_xy, cpu_no, normalize, verbose=verbose)
        self.sloX = None
        self.sloY = None
        if scandim == 'x':
            self.curvX = curv_scan(self.delayX, self.scanstep, self.dist, self.pixsize, nstep, self.mempos)
        if scandim == 'y':
            self.curvY = curv_scan(self.delayY, self.scanstep, self.dist, self.pixsize, nstep, self.mempos)

        return

    def _XST_2images_1D(self, edge_x, edge_y, pad_xy, hw_xy, normalize=False, display=False, verbose=True, _Resreturn=False):
        """
        **1D** self-reference conventional speckle tracking technique. 

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to the list [int, int].
        pad_xy : int, or [int, int]
            It defines the extra part the reference image needed to do the tracking.
            Needed when do 2D data processing. (default None)
        hw_xy : int
            It defines the width or height that needed in the template image 
            for x or y data processing. 
        normalize : bool
            To normalize the stitched image or not. (default False)
        display : bool
            To display or not. (default False)
        verbose : bool
            To show the information or not. (default True)

        Returns
        -------
        ixs, iys, resmax : numpy.array
            Shifts in x and y and the coefficient.
        """
        scandim = self.scandim
        if scandim not in ['x', 'y']:
            print("Unrecognized 1D scan mode. The supported methods are 1D X and 1D Y scan.")
            sys.exit(0)
        if isinstance(edge_x, int):
            edge_x = (edge_x, edge_x)
        if isinstance(edge_y, int):
            edge_y = (edge_y, edge_y)
        if isinstance(pad_xy, int):
            pad_xy = (pad_xy, pad_xy)
        #if scandim == 'x':
        #    if pad_xy[0] > edge_x[0] or pad_xy[1] > edge_x[1]:
        #        print("pad_xy should not be greater than edge_x.")
        #        sys.exit(0)
        #if scandim == 'y':
        #    if pad_xy[0] > edge_y[0] or pad_xy[1] > edge_y[1]:
        #        print("pad_xy should not be greater than edge_y.")
        #        sys.exit(0)
        subpixelmeth = self.subpixelmeth 
        imNo, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
        y_dim = y_dim_tmp - edge_y[0] - edge_y[1]- hw_xy + 1
        ixs = np.zeros(y_dim)
        iys = np.zeros(y_dim)
        resmax = np.zeros(y_dim)
        plane1_init = self.imstack1.data[0, 0+edge_y[0]:hw_xy+edge_y[0], edge_x[0]:x_dim_tmp-edge_x[1]].astype(np.float32)
        plane2_init = self.imstack2.data[0, 0+edge_y[0]-pad_xy[0]:hw_xy+edge_y[0]+pad_xy[1], :].astype(np.float32)
        if normalize:
            plane1_init = NormImage(plane1_init) 
            plane2_init = NormImage(plane2_init) 
        ix_init, iy_init, res_init = Imagematch(plane2_init, plane1_init, subpixelmeth=subpixelmeth)

        if display:
            fig, h1, h2, h3, h4 = _initDisplay(plane1_init, plane2_init, res_init)

        for i in range(y_dim):
            if verbose: 
                if scandim == 'x':
                    _indicator(i, y_dim, comments = self._flag + ' technique in X direction')
                if scandim == 'y':
                    _indicator(i, y_dim, comments = self._flag + ' technique in Y direction')
            plane1 = self.imstack1.data[0, i+edge_y[0]:hw_xy+i+edge_y[0], edge_x[0]:x_dim_tmp-edge_x[1]].astype(np.float32)
            plane2 = self.imstack2.data[0, i+edge_y[0]-pad_xy[0]:hw_xy+i+edge_y[0]+pad_xy[1], :].astype(np.float32)
            if normalize:
                plane1 = NormImage(plane1)
                plane2 = NormImage(plane2)
            ix_tmp, iy_tmp, res_tmp = Imagematch(plane2, plane1, subpixelmeth=subpixelmeth) 
            ixs[i] = ix_tmp - edge_x[0]
            iys[i] = iy_tmp - pad_xy[0]
            resmax[i] = np.max(res_tmp)
            if display:
                _contiDisplay(fig, h1, h2, h3, h4, plane1, plane2, res_tmp)

        if display and self.dimension == '2D': plt.close('all')

        if scandim == 'y':
            self.delayY = iys
            self._delayX = ixs
            self.resY = resmax
        if scandim == 'x':
            self.delayX = np.flip(iys)
            self._delayY = np.flip(ixs)
            self.resX = resmax

        if _Resreturn:
            return ixs, iys, resmax

    def XST_self(self, edge_x, edge_y, pad_x, pad_y, hw_xy, window=None, normalize=False, display=False, verbose=True):
        """
        Speckle tracking for self-reference XST technique.
        Two image stacks are needed. Both are with test optic.
        One image stack consists only one image when the diffuser is at one position,
        another image stack consists another image when the diffuser
        is at another position.

        This technique has been described in [XST_selfpaper]_:
        
        .. [XST_selfpaper] Hu, L., Wang, H., Fox, O., & Sawhney, K. (2022). 
             Fast wavefront sensing for X-ray optics with an alternating speckle tracking technique. 
             Opt. Exp., 30(18), 33259-33273.
             https://doi.org/10.1364/OE.460163

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        pad_x : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in x direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'y', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        pad_y : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in y direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'x', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        hw_xy : int
            The height (when Tracking.scandim is 'y') 
            or the width (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
        window : int
            The width (when Tracking.scandim is 'y') 
            or the height (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
            Only used when Tracking.dimension is '2D'. (default None)
        normalize : bool
            To normalize the stitched image or not. (default False)
        display : bool
            To display or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        if self._flag is None: self._flag = "Self-reference XST"
        scandim = self.scandim
        if scandim not in ['x', 'y', 'xy']:
            print("Unrecognized scan mode. It should be 'x', 'y' or 'xy'.")
            sys.exit(0)
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if scandim == 'xy' and self.imstack3 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if scandim == 'xy' and self.imstack4 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()
        # For 'xy' case, ROI should be a square, for 2D integaration 
        if scandim == 'xy': 
            if self.imstack1.data.shape[1] != self.imstack1.data.shape[2] \
                    or self.imstack2.data.shape[1] != self.imstack2.data.shape[2]: 
                        print("For scandim is 'xy', the ROI should be a square, please re-select.")
                        sys.exit(0)
        if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
            verbose_tmp1 = self.imstack1.verbose
            verbose_tmp2 = self.imstack2.verbose
            self.imstack1.verbose = False
            self.imstack2.verbose = False
            self.imstack1.rot90deg()
            self.imstack2.rot90deg()
            self.imstack1.verbose = verbose_tmp1
            self.imstack2.verbose = verbose_tmp2

        if isinstance(edge_x, int):
            edge_x = (edge_x, edge_x)
        if isinstance(edge_y, int):
            edge_y = (edge_y, edge_y)
        if isinstance(pad_x, int):
            pad_x = (pad_x, pad_x)
        if isinstance(pad_y, int):
            pad_y = (pad_y, pad_y)
        if scandim == 'xy':
            if edge_x != edge_y or edge_x[0] != edge_x[1] or edge_y[0] != edge_y[1]:
                print("For scandim == 'xy', the edges should be symmetrical, edge_x should be the same as edge_y, and also the elements of each.")
                sys.exit(0)
        if scandim == 'xy':
            if pad_x != pad_y or pad_x[0] != pad_x[1] or pad_y[0] != pad_y[1]: 
                print("For scandim == 'xy', the pad should be symmetrical, pad_x should be the same as pad_y, and also the elements of each.")
                sys.exit(0)

        if self.dimension == '1D':
            if scandim == 'x':
                pad_y = None
                pad_xy = (pad_x[1], pad_x[0])
                edge_x = (edge_x[1], edge_x[0])
                self._XST_2images_1D(edge_y, edge_x, pad_xy, hw_xy, normalize, display, verbose)

            if scandim == 'y':
                pad_x = None
                pad_xy = pad_y
                self._XST_2images_1D(edge_x, edge_y, pad_xy, hw_xy, normalize, display, verbose)

            if self.delayX is not None: self.curvX = curv_XST(self.delayX, self.scanstep, self.dist, self.pixsize, self.mempos)
            if self.delayY is not None: self.curvY = curv_XST(self.delayY, self.scanstep, self.dist, self.pixsize, self.mempos)

        if self.dimension == '2D':
            if scandim == 'x' or scandim == 'xy': 
                edge_x_new = edge_y
                edge_y_new = (edge_x[1], edge_x[0])
                pad_x_new = pad_y
                pad_y_new = (pad_x[1], pad_x[0])
            if scandim == 'y':
                edge_x_new = edge_x
                edge_y_new = edge_y
                pad_x_new = pad_x
                pad_y_new = pad_y
            if pad_x_new[0] > edge_x_new[0] or pad_x_new[1] > edge_x_new[1]: 
                print("pad should not be greater than edge.")
            if pad_y_new[0] > edge_y_new[0] or pad_y_new[1] > edge_y_new[1]: 
                print("pad should not be greater than edge.")
            imNo, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
            jxs = np.arange(0, x_dim_tmp-edge_x_new[0]-edge_x_new[1]-window+1, 1)
            delayY_2D = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
            delayX_2D = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
            res_2D = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
            imstack1_data = copy.deepcopy(self.imstack1.data)
            imstack2_data = copy.deepcopy(self.imstack2.data)
            if self.scandim == 'xy': self.scandim = 'x'
            for index, jx in enumerate(jxs):
                if verbose:
                    if self.scandim == 'x':
                        _indicator(jx, len(jxs), comments = self._flag + 'technique in X direction')
                    if self.scandim == 'y':
                        _indicator(jx, len(jxs), comments = self._flag + 'technique in Y direction')
                y_ind_left = edge_y_new[0] - pad_y_new[0]
                y_ind_right = y_dim_tmp - edge_y_new[1] + pad_y_new[1]
                x_ind_left = jx + edge_x_new[0] - pad_x_new[0]
                x_ind_right = jx + edge_x_new[0] + window + pad_x_new[1]
                self.imstack1.data = imstack1_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                self.imstack2.data = imstack2_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                ix_new, iy_new, res_new = self._XST_2images_1D(pad_x_new, pad_y_new, pad_y_new, hw_xy, normalize, display, verbose=False, _Resreturn=True)
                delayX_2D[:, index] = ix_new #- pad_x_new[0]
                delayY_2D[:, index] = iy_new #- pad_y_new[0]
                res_2D[:, index] = res_new

            if scandim == 'y':
                self.delayY = delayY_2D
                self._delayX = delayX_2D
                self.resY = res_2D
                self.curvY = curv_XST(self.delayY, self.scanstep, self.dist, self.pixsize, self.mempos)
                self._curvX = None

            if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
                self.delayX = np.rot90(delayY_2D, k=-1)
                self._delayY = np.rot90(delayX_2D, k=-1)
                self.resX = np.rot90(res_2D, k=-1)
                self.curvX = curv_XST(self.delayX, self.scanstep, self.dist, self.pixsize, self.mempos)
                self._curvY = None

            if scandim == 'xy':
                if self.imstack3.rawdata is None:
                    self.imstack3.read_data()
                if self.imstack4.rawdata is None:
                    self.imstack4.read_data()
                # For 'xy' case, ROI should be a square, for 2D integaration 
                if self.imstack3.data.shape[1] != self.imstack3.data.shape[2] \
                        or self.imstack4.data.shape[1] != self.imstack4.data.shape[2]: 
                            print("For scandim is 'xy', the ROI should be a square, please re-select.")
                            sys.exit(0)

                edge_x_new = edge_x
                edge_y_new = edge_y
                pad_x_new = pad_x
                pad_y_new = pad_y
                if pad_x_new[0] > edge_x_new[0] or pad_x_new[1] > edge_x_new[1]: 
                    print("pad should not be greater than edge.")
                if pad_y_new[0] > edge_y_new[0] or pad_y_new[1] > edge_y_new[1]: 
                    print("pad should not be greater than edge.")
                imNo, y_dim_tmp, x_dim_tmp = self.imstack3.data.shape
                jxs = np.arange(0, x_dim_tmp-edge_x_new[0]-edge_x_new[1]-window+1, 1)
                delayY_2D_2 = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
                delayX_2D_2 = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
                res_2D_2 = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
                imstack1_data = copy.deepcopy(self.imstack1.data)
                imstack2_data = copy.deepcopy(self.imstack2.data)
                imstack3_data = copy.deepcopy(self.imstack3.data)
                imstack4_data = copy.deepcopy(self.imstack4.data)
                self.scandim = 'y'
                for index, jx in enumerate(jxs):
                    if verbose:
                        if self.scandim == 'x':
                            _indicator(jx, len(jxs), comments = self._flag + 'technique in X direction')
                        if self.scandim == 'y':
                            _indicator(jx, len(jxs), comments = self._flag + 'technique in Y direction')
                    y_ind_left = edge_y_new[0] - pad_y_new[0]
                    y_ind_right = y_dim_tmp - edge_y_new[1] + pad_y_new[1]
                    x_ind_left = jx + edge_x_new[0] - pad_x_new[0]
                    x_ind_right = jx + edge_x_new[0] - pad_x_new[0] + window + pad_x_new[1]
                    self.imstack1.data = imstack3_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                    self.imstack2.data = imstack4_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                    ix_new2, iy_new2, res_new2 = self._XST_2images_1D(pad_x_new, pad_y_new, pad_y_new, hw_xy, normalize, display, verbose=False, _Resreturn=True)
                    delayX_2D_2[:, index] = ix_new2 #- pad_x_new[0]
                    delayY_2D_2[:, index] = iy_new2 #- pad_y_new[0]
                    res_2D_2[:, index] = res_new2

                self.scandim = scandim
                self.delayY = delayY_2D_2
                self._delayX = delayX_2D_2
                self.resY = res_2D_2
                self.imstack1.data = imstack1_data
                self.imstack2.data = imstack2_data

                self.curvY = curv_XST(self.delayY, self.scanstep, self.dist, self.pixsize, self.mempos)
                self._curvX = None

        return

    def XST_self_multi(self, edge_x, edge_y, pad_x, pad_y, hw_xy, window, cpu_no, normalize=False, verbose=True):
        """
        Speckle tracking for self-reference XST technique.
        Two image stacks are needed. Both are with test optic.
        One image stack consists one image when the diffuser is at one position,
        another image stack consists another image when the diffuser
        is at another position.

        This technique has been described in [XST_selfpaper2]_:
        
        .. [XST_selfpaper2] Hu, L., Wang, H., Fox, O., & Sawhney, K. (2022). 
             Fast wavefront sensing for X-ray optics with an alternating speckle tracking technique. 
             Opt. Exp., 30(18), 33259-33273.
             https://doi.org/10.1364/OE.460163

        .. warning:: **BE CAREFUL** to check the available and safe cpu numbers before run this function!!

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        pad_x : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in x direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'y', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        pad_y : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in y direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'x', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        hw_xy : int
            The height (when Tracking.scandim is 'y') 
            or the width (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
        window : int
            The width (when Tracking.scandim is 'y') 
            or the height (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
            Only used when Tracking.dimension is '2D'. (default None)
        cpu_no : int
            The number of CPUs that is available.
        normalize : bool
            To normalize the stitched image or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        if self._flag is None: self._flag = "Self-reference XST"
        scandim = self.scandim
        if scandim not in ['x', 'y', 'xy']:
            print("Unrecognized scan mode. It should be 'x', 'y' or 'xy'.")
            sys.exit(0)
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if scandim == 'xy' and self.imstack3 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if scandim == 'xy' and self.imstack4 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()
        # For 'xy' case, ROI should be a square, for 2D integaration 
        if scandim == 'xy': 
            if self.imstack1.data.shape[1] != self.imstack1.data.shape[2] \
                    or self.imstack2.data.shape[1] != self.imstack2.data.shape[2]: 
                        print("For scandim is 'xy', the ROI should be a square, please re-select.")
                        sys.exit(0)
        if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
            verbose_tmp1 = self.imstack1.verbose
            verbose_tmp2 = self.imstack2.verbose
            self.imstack1.verbose = False
            self.imstack2.verbose = False
            self.imstack1.rot90deg()
            self.imstack2.rot90deg()
            self.imstack1.verbose = verbose_tmp1
            self.imstack2.verbose = verbose_tmp2

        if isinstance(edge_x, int):
            edge_x = (edge_x, edge_x)
        if isinstance(edge_y, int):
            edge_y = (edge_y, edge_y)
        if isinstance(pad_x, int):
            pad_x = (pad_x, pad_x)
        if isinstance(pad_y, int):
            pad_y = (pad_y, pad_y)
        if scandim == 'xy':
            if edge_x != edge_y or edge_x[0] != edge_x[1] or edge_y[0] != edge_y[1]:
                print("For scandim == 'xy', the edges should be symmetrical, edge_x should be the same as edge_y, and also the elements of each.")
                sys.exit(0)
        if scandim == 'xy':
            if pad_x != pad_y or pad_x[0] != pad_x[1] or pad_y[0] != pad_y[1]: 
                print("For scandim == 'xy', the pad should be symmetrical, pad_x should be the same as pad_y, and also the elements of each.")
                sys.exit(0)

        if self.dimension == '1D':
            print("No need to use multiprocessing for 1D analysis.")
            sys.exit(0)

        if self.dimension == '2D':
            if scandim == 'x' or scandim == 'xy': 
                edge_x_new = edge_y
                edge_y_new = (edge_x[1], edge_x[0])
                pad_x_new = pad_y
                pad_y_new = (pad_x[1], pad_x[0])
            if scandim == 'y':
                edge_x_new = edge_x
                edge_y_new = edge_y
                pad_x_new = pad_x
                pad_y_new = pad_y
            if pad_x_new[0] > edge_x_new[0] or pad_x_new[1] > edge_x_new[1]: 
                print("pad should not be greater than edge.")
            if pad_y_new[0] > edge_y_new[0] or pad_y_new[1] > edge_y_new[1]: 
                print("pad should not be greater than edge.")
            imNo, y_dim_tmp, x_dim_tmp = self.imstack1.data.shape
            jxs = np.arange(0, x_dim_tmp-edge_x_new[0]-edge_x_new[1]-window+1, 1)
            delayY_2D = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
            delayX_2D = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
            res_2D = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
            imstack1_data = copy.deepcopy(self.imstack1.data)
            imstack2_data = copy.deepcopy(self.imstack2.data)
            if self.scandim == 'xy': self.scandim = 'x'
            jxs1 = list(jxs)
            global process_tmp1
            def process_tmp1(jx):
                if verbose:
                    if self.scandim == 'x':
                        _indicator(jx, len(jxs1), comments = self._flag + 'technique in X direction')
                    if self.scandim == 'y':
                        _indicator(jx, len(jxs1), comments = self._flag + 'technique in Y direction')
                y_ind_left = edge_y_new[0] - pad_y_new[0]
                y_ind_right = y_dim_tmp - edge_y_new[1] + pad_y_new[1]
                x_ind_left = jx + edge_x_new[0] - pad_x_new[0]
                x_ind_right = jx + edge_x_new[0] + window + pad_x_new[1]
                self.imstack1.data = imstack1_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                self.imstack2.data = imstack2_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                ix_new1, iy_new1, res_new1 = self._XST_2images_1D(pad_x_new, pad_y_new, pad_y_new, hw_xy, normalize, display=False, verbose=False, _Resreturn=True)
                #ix_new1 -= pad_x_new[0]
                #iy_new1 -= pad_y_new[0]

                return ix_new1, iy_new1, res_new1

            with multiprocessing.Pool(cpu_no) as pool:
                results = pool.map(process_tmp1, jxs1)
            for i in range(len(results)):
                delayX_2D[:, i] = results[i][0]
                delayY_2D[:, i] = results[i][1]
                res_2D[:, i] = results[i][2]

            if scandim == 'y':
                self.delayY = delayY_2D
                self._delayX = delayX_2D
                self.resY = res_2D
                self.curvY = curv_XST(self.delayY, self.scanstep, self.dist, self.pixsize, self.mempos)
                self._curvX = None

            if scandim == 'x' or scandim == 'xy': #or scandim == 'diag':
                self.delayX = np.rot90(delayY_2D, k=-1)
                self._delayY = np.rot90(delayX_2D, k=-1)
                self.resX = np.rot90(res_2D, k=-1)
                self.curvX = curv_XST(self.delayX, self.scanstep, self.dist, self.pixsize, self.mempos)
                self._curvY = None

            if scandim == 'xy':
                if self.imstack3.rawdata is None:
                    self.imstack3.read_data()
                if self.imstack4.rawdata is None:
                    self.imstack4.read_data()
                # For 'xy' case, ROI should be a square, for 2D integaration 
                if self.imstack3.data.shape[1] != self.imstack3.data.shape[2] \
                        or self.imstack4.data.shape[1] != self.imstack4.data.shape[2]: 
                            print("For scandim is 'xy', the ROI should be a square, please re-select.")
                            sys.exit(0)

                edge_x_new = edge_x
                edge_y_new = edge_y
                pad_x_new = pad_x
                pad_y_new = pad_y
                if pad_x_new[0] > edge_x_new[0] or pad_x_new[1] > edge_x_new[1]: 
                    print("pad should not be greater than edge.")
                if pad_y_new[0] > edge_y_new[0] or pad_y_new[1] > edge_y_new[1]: 
                    print("pad should not be greater than edge.")
                imNo, y_dim_tmp, x_dim_tmp = self.imstack3.data.shape
                jxs = np.arange(0, x_dim_tmp-edge_x_new[0]-edge_x_new[1]-window+1, 1)
                delayY_2D_2 = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
                delayX_2D_2 = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
                res_2D_2 = np.empty((y_dim_tmp-edge_y_new[0]-edge_y_new[1]-hw_xy+1, len(jxs)))
                imstack1_data = copy.deepcopy(self.imstack1.data)
                imstack2_data = copy.deepcopy(self.imstack2.data)
                imstack3_data = copy.deepcopy(self.imstack3.data)
                imstack4_data = copy.deepcopy(self.imstack4.data)
                self.scandim = 'y'
                jxs2 = list(jxs)
                global process_tmp2
                def process_tmp2(jx):
                    if verbose:
                        if self.scandim == 'x':
                            _indicator(jx, len(jxs2), comments = self._flag + 'technique in X direction')
                        if self.scandim == 'y':
                            _indicator(jx, len(jxs2), comments = self._flag + ' technique in Y direction')
                    y_ind_left = edge_y_new[0] - pad_y_new[0]
                    y_ind_right = y_dim_tmp - edge_y_new[1] + pad_y_new[1]
                    x_ind_left = jx + edge_x_new[0] - pad_x_new[0]
                    x_ind_right = jx + edge_x_new[0] - pad_x_new[0] + window + pad_x_new[1]
                    self.imstack1.data = imstack3_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                    self.imstack2.data = imstack4_data[:, y_ind_left:y_ind_right, x_ind_left:x_ind_right]
                    ix_new2, iy_new2, res_new2 = self._XST_2images_1D(pad_x_new, pad_y_new, pad_y_new, hw_xy, normalize, display=False, verbose=False, _Resreturn=True)
                    #ix_new2 -= pad_x_new[0]
                    #iy_new2 -= pad_y_new[0]

                    return ix_new2, iy_new2, res_new2

                with multiprocessing.Pool(cpu_no) as pool:
                    results = pool.map(process_tmp2, jxs2)
                for i in range(len(results)):
                    delayX_2D_2[:, i] = results[i][0]
                    delayY_2D_2[:, i] = results[i][1]
                    res_2D_2[:, i] = results[i][2]

                self.scandim = scandim
                self.delayY = delayY_2D_2
                self._delayX = delayX_2D_2
                self.resY = res_2D_2
                self.imstack1.data = imstack1_data
                self.imstack2.data = imstack2_data

                self.curvY = curv_XST(self.delayY, self.scanstep, self.dist, self.pixsize, self.mempos)
                self._curvX = None

        return

    def XST_withrefer(self, edge_x, edge_y, pad_x, pad_y, hw_xy, window=None, normalize=False, display=False, verbose=True):
        """
        Speckle tracking for conventional XST technique, with reference beam.
        Two image stacks are needed. 
        The fisrt image stack consists one image when the diffuser is in the beam,
        another image stack consists one reference image without the tested optic.

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        pad_x : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in x direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'y', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        pad_y : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in y direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'x', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        hw_xy : int
            The height (when Tracking.scandim is 'y') 
            or the width (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
        window : int
            The width (when Tracking.scandim is 'y') 
            or the height (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
            Only used when Tracking.dimension is '2D'. (default None)
        normalize : bool
            To normalize the stitched image or not. (default False)
        display : bool
            To display or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        self._flag = "XST(with reference)"
        self.scandim = 'y'
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()

        self.XST_self(edge_x, edge_y, pad_x, pad_y, hw_xy, window, normalize, display, verbose)

        self.delayX = copy.deepcopy(self._delayX)
        self._delayX = None

        self.sloX = slope_pixel(self.delayX, self.pixsize, self.dist)
        self.sloY = slope_pixel(self.delayY, self.pixsize, self.dist)

        self.scandim = 'random'

        return


    def XST_withrefer_multi(self, edge_x, edge_y, pad_x, pad_y, hw_xy, window, cpu_no, normalize=False, verbose=True):
        """
        Speckle tracking for conventional XST technique, with reference beam.
        Two image stacks are needed. 
        The fisrt image stack consists one image when the diffuser is in the beam,
        another image stack consists one reference image without the tested optic.

        .. warning:: **BE CAREFUL** to check the available and safe cpu numbers before run this function!!

        Parameters
        ----------
        edge_x : int, or [int, int]
            Area needs to be cut in x dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='x' (scan in x direction), it is useless.
        edge_y : int, or [int, int]
            Area needs to be cut in y dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            If Tracking.scandim='y' (scan in y direction), it is useless.
        pad_x : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in x direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'y', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        pad_y : int, or [int, int]
            It defines the extra part the reference image needed to do 
            the tracking in y direction. If Tracking.dimension  is '1D' 
            **and** Tracking.scandim is is 'x', it is useless.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        hw_xy : int
            The height (when Tracking.scandim is 'y') 
            or the width (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
        window : int
            The width (when Tracking.scandim is 'y') 
            or the height (when Tracking.scandim is 'x') 
            of the subregion to be chosen from the template for cross-correlation.
            Only used when Tracking.dimension is '2D'. 
        cpu_no : int
            The number of CPUs that is available.
        normalize : bool
            To normalize the stitched image or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        self._flag = "XST(with reference)"
        self.scandim = 'y'
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()

        self.XST_self_multi(edge_x, edge_y, pad_x, pad_y, hw_xy, window, cpu_no, normalize, verbose)

        self.delayX = copy.deepcopy(self._delayX)
        self._delayX = None

        self.sloX = slope_pixel(self.delayX, self.pixsize, self.dist)
        self.sloY = slope_pixel(self.delayY, self.pixsize, self.dist)

        self.scandim = 'random'

        return

    def XSVT_withrefer(self, edge_xy, edge_z, hw_xy=None, pad_xy=None, normalize=False, display=False, verbose=True):
        """
        Speckle tracking for XSVT technique with reference beam.
        The fisrt image stack is the one with test optic.
        The second image stack is the reference image stack.

        Parameters
        ----------
        edge_xy : int, or [int, int]
            Area needs to be cut in x and y dimensions.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        edge_z : int, or [int, int]
            Area needs to be cut in scan number dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
        hw_xy : int
            The width and height of the image subregion. 
            Needed when do 2D data processing. (default None) 
        pad_xy : int, or [int, int]
            It defines the extra part the reference image needed to do the tracking.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            Needed when do 2D data processing. (default None)
        normalize : bool
            To normalize the stitched image or not. (default False)
        display : bool
            To display or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        self._flag = "XSVT(with reference)"
        self.scandim = 'random'
        self.scanstep = 1.    # Dummy
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.dimension == '1D':
            print("The 1D data processing is not supported for XSVT method. \
                    Instead, you can cut a small strip of data and do 2D \
                    data processing and extract the information you want.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()
        if isinstance(edge_xy, int):
            edge_xy = (edge_xy, edge_xy)
        if isinstance(edge_z, int):
            edge_z = (edge_z, edge_z)
        self.imstack3 = copy.deepcopy(self.imstack1)
        self.imstack4 = copy.deepcopy(self.imstack2)

        #Use the underscored delay in the XSS method, thus delayY is _delayX, delayX is _delayY.
        self.scandim = 'xy'
        self.XSS_withrefer(edge_xy, edge_xy, edge_z, hw_xy, pad_xy, normalize, display, verbose)
        delayX = copy.deepcopy(self._delayY)
        delayY = copy.deepcopy(self._delayX)
        self.delayX = delayX
        self.delayY = delayY
        self._delayX = None 
        self._delayY = None
        self._sloX = None 
        self._sloY = None
        self.sloX = slope_pixel(self.delayX, self.pixsize, self.dist)
        self.sloY = slope_pixel(self.delayY, self.pixsize, self.dist)


        self.scandim = 'random' 

        return

    def XSVT_withrefer_multi(self, edge_xy, edge_z, hw_xy, pad_xy, cpu_no, normalize=False, verbose=True):
        """
        Speckle tracking for XSVT technique with reference beam.
        The fisrt image stack is the one with test optic.
        The second image stack is the reference image stack.

        .. warning:: **BE CAREFUL** to check the available and safe cpu numbers before run this function!!

        Parameters
        ----------
        edge_xy : int, or [int, int]
            Area needs to be cut in x and y dimensions.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
        edge_z : int, or [int, int]
            Area needs to be cut in scan number dimension.
            If it is a single integer, it will be expanded automatically 
            to (int, int).
        hw_xy : int
            The width and height of the image subregion. 
            Needed when do 2D data processing.  
        pad_xy : int, or [int, int]
            It defines the extra part the reference image needed to do the tracking.
            If it is a single integer, it will be expanded automatically 
            to (int, int). 
            Needed when do 2D data processing. 
        cpu_no : int
            The number of CPUs that is available.
        normalize : bool
            To normalize the stitched image or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        self._flag = "XSVT(with reference)"
        self.scandim = 'random'
        self.scanstep = 1.    # Dummy
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.dimension == '1D':
            print("The 1D data processing is not supported for XSVT method. \
                    Instead, you can cut a small strip of data and do 2D \
                    data processing and extract the information you want.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()
        if isinstance(edge_xy, int):
            edge_xy = (edge_xy, edge_xy)
        if isinstance(edge_z, int):
            edge_z = (edge_z, edge_z)
        self.imstack3 = copy.deepcopy(self.imstack1)
        self.imstack4 = copy.deepcopy(self.imstack2)

        #Use the underscored delay in the XSS method, thus delayY is _delayX, delayX is _delayY.
        self.scandim = 'xy'
        self.XSS_withrefer_multi(edge_xy, edge_xy, edge_z, hw_xy, pad_xy, cpu_no, normalize, verbose)
        delayX = copy.deepcopy(self._delayY)
        delayY = copy.deepcopy(self._delayX)
        self.delayX = delayX
        self.delayY = delayY
        self._delayX = None 
        self._delayY = None
        self._sloX = None 
        self._sloY = None
        self.sloX = slope_pixel(self.delayX, self.pixsize, self.dist)
        self.sloY = slope_pixel(self.delayY, self.pixsize, self.dist)

        self.scandim = 'random' 

        return


    def Hartmann_XST(self, cen_xmesh, cen_ymesh, pad, size, normalize=False, display=False, verbose=True):
        """
        Hartmann-like data processing procedure.
        Two image stacks are needed.
        The fisrt image stack consists one sample image.
        The second image stack consists another reference image.

        .. note::
            For simplicity, unlike other mode of data processing, 
            we only provide ``Tracking.delayX`` and ``Tracking.delayY``
            for this method. Recovering the appropriate physical quantities 
            from the speckle shifts is left to the discretion of the users.
            

        Parameters
        ----------
        cen_xmsh : numpy.ndarray 
           Mesh grid of the x coordinate of the box centre. A 2D array.
        cen_ymsh : numpy.ndarray 
           Mesh grid of the y coordinate of the box centre. A 2D array.
        pad : int, or [int, int]
           It defines the extra part the reference image needed to do the tracking.
           If it is a single integer, it will be expanded automatically 
           to (int, int). 
        size : int, or [int, int]
           It defines the size of the box used for Hartmann-like tracking mode.
           If it is a single integer, it will be expanded automatically 
           to (int, int). size[0] is the half width of the chosen box, size[1] is 
           the half height of the chosen box.
        normalize : bool
            To normalize the stitched image or not. (default False)
        display : bool
            To display or not. (default False)
        verbose : bool
            To show the information or not. (default True)
        """
        if self._flag is None: self._flag = "Hartmann-like"
        if self.imstack2 == None:
            print("Please provide another image stack.")
            sys.exit(0)
        if self.imstack1.rawdata is None:
            self.imstack1.read_data()
        if self.imstack2.rawdata is None:
            self.imstack2.read_data()
        if isinstance(pad, int):
            pad = (pad, pad)
        if isinstance(size, int):
            size = (size, size)
        y_dim_tmp, x_dim_tmp = cen_xmesh.shape
        delayX_2D = np.empty((y_dim_tmp, x_dim_tmp))
        delayY_2D = np.empty((y_dim_tmp, x_dim_tmp))
        res_2D = np.empty((y_dim_tmp, x_dim_tmp))
        imstack1_data = copy.deepcopy(self.imstack1.data)
        imstack2_data = copy.deepcopy(self.imstack2.data)
        im_sam_init = imstack1_data[0][cen_ymesh[0, 0]-size[1]:cen_ymesh[0, 0]+size[1], cen_xmesh[0, 0]-size[0]:cen_xmesh[0, 0]+size[0]] 
        im_ref_init = imstack2_data[0][cen_ymesh[0, 0]-size[1]-pad[1]:cen_ymesh[0, 0]+size[1]+pad[1], cen_xmesh[0, 0]-size[0]-pad[0]:cen_xmesh[0, 0]+size[0]+pad[0]] 
        subpixelmeth = self.subpixelmeth 
        if normalize:
            im_sam_init = NormImage(im_sam_init) 
            im_ref_init = NormImage(im_ref_init) 
        ix_init, iy_init, res_init = Imagematch(im_ref_init, im_sam_init, subpixelmeth=subpixelmeth)
        if display:
            fig, h1, h2, h3, h4 = _initDisplay(im_sam_init, im_ref_init, res_init)
        for iy in range(y_dim_tmp):
            if verbose:
                _indicator(iy, y_dim_tmp, comments = self._flag)
            for ix in range(x_dim_tmp):
                cen_index_x = cen_xmesh[iy, ix]
                cen_index_y = cen_ymesh[iy, ix]
                im_sam = imstack1_data[0][cen_index_y-size[1]:cen_index_y+size[1], cen_index_x-size[0]:cen_index_x+size[0]] 
                im_ref = imstack2_data[0][cen_index_y-size[1]-pad[1]:cen_index_y+size[1]+pad[1], cen_index_x-size[0]-pad[0]:cen_index_x+size[0]+pad[0]] 
                subpixelmeth = self.subpixelmeth 
                if normalize:
                    im_sam = NormImage(im_sam) 
                    im_ref = NormImage(im_ref) 
                ix_tmp, iy_tmp, res_tmp = Imagematch(im_ref, im_sam, subpixelmeth=subpixelmeth)
                delayX_2D[iy, ix] = ix_tmp - pad[0]
                delayY_2D[iy, ix] = iy_tmp - pad[1]
                res_2D[iy, ix] = np.max(res_tmp)
                if display:
                    _contiDisplay(fig, h1, h2, h3, h4, im_sam, im_ref, res_tmp)
        
        self.delayY = delayY_2D
        self.delayX = delayX_2D
    
        return
