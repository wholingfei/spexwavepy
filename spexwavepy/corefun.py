import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
import cv2

def _indicator(num, length, comments=None):
    """
    This auxiliary function is used for counting.

    Parameters
    ----------
    num : int
        Index for each loop
    length : int
        Total loop length
    comments : str, optional, default is None
        Used as helper

    Returns
    -------
    None
    """
    progress = np.round(np.arange(0.1, 1.1, 0.1) * length)
    strings = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
    if num in progress:
        if comments is not None: print(comments)
        index = np.where(progress==num)[0][0]
        print(strings[index])

    return 

def read_one(filename, ShowImage=False):
    """
    Read one image

    Parameters
    ----------
    filename : str
        File name 
    ShowImage : bool, optional
        To show image or not, default is False

    Return
    ------
    numpy.ndarray
        One image data
    """
    data_tmp = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
    if ShowImage:
        plt.figure()
        plt.imshow(data_tmp, cmap='jet')
        plt.show()

    return data_tmp 

def crop_one(data, ROI, ShowImage=False):
    """
    Crop the input image according to the ROI.

    Parameters
    ----------
    data : numpy.ndarray
        The input image data
    ROI : [int, int, int, int]
        ROI is [y_begin, y_end, x_begin, x_end]
    ShowImage : bool, optional
        To show image or not, default is False

    Return
    ------
    numpy.ndarray
        The cropped image data
    """
    if ShowImage:
        plt.figure()
        plt.imshow(data[ROI[0]:ROI[1], ROI[2]:ROI[3]], cmap='jet')
        plt.show()

    return data[ROI[0]:ROI[1], ROI[2]:ROI[3]]

def NormImage(image_raw):
    """
    Image Normalization. This function is used to mitigate the low frequent structures appeared in the raw images. 

    Parameters
    ----------
    image_raw : numpy.ndarray
        The input image data

    Return
    ------
    numpy.ndarray
        The normalized image
    """
    row_sum_1 = np.sum(image_raw, axis=0)
    col_sum_1 = np.sum(image_raw, axis=1)
    col_sum_1 = col_sum_1.reshape(len(col_sum_1), 1)
    image_raw = image_raw / row_sum_1 / col_sum_1 
    image_raw = (image_raw - np.mean(image_raw)) / np.std(image_raw)

    return image_raw

def Imagematch(im1, im2, meth='cv2.TM_CCOEFF_NORMED', subpixelmeth='default', res=True):
    """
    Find the shifts between two images (im1 and im2) with subpixel accuracy.

    Parameters
    ----------
    im1 : numpy.ndarray
       Reference image.
    im2 : numpy.ndarray
       Image to be tracked. It must be **smaller** than im1.
    meth : str
       Method for cv2.matchTemplate (default 'cv2.TM_CCOEFF_NORMED'). Other methods need to be implemented.
    subpixelmeth : str
       The algorithm used with subpixel accuracy (default 'default'). 
       When it is None, no subpixel method used. Now only 'default'|'gausspeak'|'parapeak' is available. 
    res : bool
       Whether or not return the coefficient matrix (default True). 
    
    Returns
    -------
    delayX, delayY, res_mat : numpy.array
       Shifts in two dimensions, matching coefficient matrix if res is True.
    """
    if meth not in ['cv2.TM_CCOEFF_NORMED']:
        print("This tracking method needs to be implemented...")
        sys.exit(0)
    else:
        planeRef = im1.astype(np.float32)
        planetmp = im2.astype(np.float32)
    res_mat = cv2.matchTemplate(planeRef, planetmp, eval(meth))
    index_y, index_x = np.where(res_mat==np.max(res_mat))
    index_y = index_y[0]
    index_x = index_x[0]
    if subpixelmeth is None:
        delayX = index_x
        delayY = index_y
    if subpixelmeth not in ['default', 'gausspeak', 'parapeak']:
        print("No valid method for subpixel registration! Pixel accuracy values are used.")
        sys.exit(0)
    if subpixelmeth == 'default':
        try:
            delta_x, delta_y = subpix_default(res_mat)
        except IndexError as err:
            print("Potential tracking failure, no subpixel registration: \n")
            delta_x, delta_y = 10000., 10000.
    if subpixelmeth == 'gausspeak':
        try:
            delta_x, delta_y = subpix_gausspeak(res_mat)
        except IndexError as err:
            print("Potential tracking failure, no subpixel registration: \n")
            delta_x, delta_y = 10000., 10000.
    if subpixelmeth == 'parapeak':
        try:
            delta_x, delta_y = subpix_parapeak(res_mat)
        except IndexError as err:
            print("Potential tracking failure, no subpixel registration: \n")
            delta_x, delta_y = 10000., 10000.
    delayX = index_x + delta_x
    delayY = index_y + delta_y

    if res:
        return delayX, delayY, res_mat
    else:
        return delayX, delayY

def subpix_default(res, meth='cv2.TM_CCOEFF_NORMED'):
    """
    The default subpix registration method. 

    This subpixel registrition method can be found from [FLCT]_ and [QiaoWavelet]_. 

    .. [FLCT] Fisher, G. H., & Welsch, B.T. 
         "FLCT: a fast, efficient method for performing local correlation tracking." 
         Subsurface and Atmospheric Influences on Solar Activity. Vol. 383. 2008.

    .. [QiaoWavelet] Qiao, Zhi, et al. 
        "Wavelet-transform-based speckle vector tracking method for X-ray phase imaging." 
        Optics Express 28.22 (2020): 33053-33067. 

    Parameters
    ----------
    res : numpy.ndarray  
        The cross correlation results, 2D array
    meth : str
        Method for cv2.matchTemplate (default 'cv2.TM_CCOEFF_NORMED'). 
        Do not modify it, other methods haven't been implemented.

    Returns
    -------
    Deltax, Deltay : numpy.array
        1D array of subpixel registration results.

    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    max_y, max_x = top_left[1], top_left[0]
    ##Central Difference 
    dx = (res[max_y, max_x+1] - res[max_y, max_x-1]) / 2.
    dy = (res[max_y+1, max_x] - res[max_y-1, max_x]) / 2.
    dxx = res[max_y, max_x+1] + res[max_y, max_x-1] - 2*res[max_y, max_x]
    dyy = res[max_y+1, max_x] + res[max_y-1, max_x] - 2*res[max_y, max_x]
    dxy = (res[max_y+1, max_x+1] + res[max_y-1, max_x-1] - res[max_y+1, max_x-1] - res[max_y-1, max_x+1]) / 4.
    ##
    det = 1. / (dxy**2 - dxx * dyy)

    return (dyy*dx - dxy*dy) * det, (dxx*dy - dxy*dx) * det   #Deltax, Deltay

def subpix_gausspeak(res, meth='cv2.TM_CCOEFF_NORMED'):
    """
    Gaussian peak fitting for subpixel registration.

    Parameters
    ----------
    res : numpy.ndarray  
        The cross correlation results, 2D array
    meth : str
        Method for cv2.matchTemplate (default is 'cv2.TM_CCOEFF_NORMED'). 
        Do not modify it, other methods haven't been implemented.

    Returns
    -------
    Deltax, Deltay : numpy.array
        1D array of subpixel registration results.

    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    max_y, max_x = top_left[1], top_left[0]

    dx = (np.log(res[max_y, max_x-1]) - np.log(res[max_y, max_x+1])) \
            / (2 * np.log(res[max_y, max_x-1]) - 4 * np.log(res[max_y, max_x]) + 2 * np.log(res[max_y, max_x+1]))
    dy = (np.log(res[max_y-1, max_x]) - np.log(res[max_y+1, max_x])) \
           / (2 * np.log(res[max_y-1, max_x]) - 4 * np.log(res[max_y, max_x]) + 2 * np.log(res[max_y+1, max_x]))

    return dx, dy

def subpix_parapeak(res, meth='cv2.TM_CCOEFF_NORMED'):
    """
    Parabola peak fitting for subpixel registration.

    Parameters
    ----------
    res : numpy.ndarray  
        The cross correlation results, 2D array
    meth : str
        Method for cv2.matchTemplate (default is 'cv2.TM_CCOEFF_NORMED'). 
        Do not modify it, other methods haven't been implemented.

    Returns
    -------
    Deltax, Deltay : numpy.array
        1D array of subpixel registration results.

    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    max_y, max_x = top_left[1], top_left[0]

    dx = (res[max_y, max_x-1] - res[max_y, max_x+1]) \
            / (2 * res[max_y, max_x-1] - 4 * res[max_y, max_x] + 2 * res[max_y, max_x+1])
    dy = (res[max_y-1, max_x] - res[max_y+1, max_x]) \
           / (2 * res[max_y-1, max_x] - 4 * res[max_y, max_x] + 2 * res[max_y+1, max_x])

    return dx, dy

def _initDisplay(plane1_init, plane2_init, res_init):
    """
    Used for initialize figure layout for dispalaying.

    Parameters
    ----------
    plane1_init: numpy.ndarray
        Template image
    plane2_init: numpy.ndarray
        Original image
    res_init: numpy.ndarray
        2D cross-correlation results matrix

    Returns
    -------
    fig, h1, h2, h3, h4
        Auxiliaries for animation
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_init)
    top_left = max_loc
    index_y, index_x = top_left[1], top_left[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    y_dim_1, x_dim_1 = plane1_init.shape
    h1 = ax1.imshow(plane1_init, extent=[0, x_dim_1, 0, y_dim_1], aspect=x_dim_1/y_dim_1)
    ax2 = fig.add_subplot(222)
    y_dim_2, x_dim_2 = plane2_init.shape
    y_dim_res, x_dim_res = res_init.shape
    h2 = ax2.imshow(plane2_init, extent=[0, x_dim_2, 0, y_dim_2], aspect=x_dim_2/y_dim_2)
    ax3 = fig.add_subplot(223)
    h3 = ax3.imshow(res_init, extent=[0, x_dim_res, 0, y_dim_res], aspect=x_dim_res/y_dim_res)
    ax4 = fig.add_subplot(224)
    h4 = ax4.plot(res_init[:, index_x])[0]
    ax4.set_ylim(0.1, 1.)

    return fig, h1, h2, h3, h4

def _contiDisplay(fig, h1, h2, h3, h4, plane1t, plane2, res):
    """
    Used for dispalaying.

    .. note:: Double click on the image will terminate displaying!

    Parameters
    ----------
    fig, h1, h2, h3, h4
        Auxiliaries for animation
    plane1t : numpy.ndarray
        Template image
    plane2 : numpy.ndarray
        Original image
    res : numpy.ndarray
        2D cross-correlation results matrix
    """
    def _onclick(event):
        if event.dblclick:
            sys.exit(0)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    index_y, index_x = top_left[1], top_left[0]
    h1.set_data(plane1t)
    h2.set_data(plane2)
    h3.set_data(res)
    h4.set_ydata(res[:, index_x])
    plt.draw()
    plt.pause(1e-3)

    connection_id = fig.canvas.mpl_connect('button_press_event', _onclick)

    return
