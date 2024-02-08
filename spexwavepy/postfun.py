import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def slope_scan(delay, scanstep, dist):
    """
    Get wavefront slope (slope error) from tracked shift. 
    Used for scan-based technique with reference beam.

    Parameters
    ----------
    delay: numpy.ndarray
        Tracked shift in the scanned direction.
    scanstep : float
        scan step size. Unit in :math: `\mu m`.
    dist : float
        distance from diffusor to detector plane,
        if it's the downstream mode, unit in mm. 
        If the diffuser is placed in the upstream,
        usually it is set to be the distance between the centre 
        of the optic to the detector. 

    Returns
    -------
    numpy.ndarray
        wavefront slope (slope error). 
        Unit in :math:`\mu rad`.
    """

    return delay * scanstep / (dist * 1.e-3)     # [urad]

def slope_pixel(delay, pixsize, dist):
    """
    Get wavefront slope (slope error) from tracked shift. 
    Used when not scanned.

    Parameters
    ----------
    delay: numpy.ndarray
        Tracked shift in the scanned direction.
    pixsize : float
        detector pixel size. Unit in :math: `\mu m`.
    dist : float
        distance from diffusor to detector plane,
        if it's the downstream mode, unit in mm. 
        If the diffuser is placed in the upstream,
        usually it is set to be the distance between the centre 
        of the optic to the detector. 

    Returns
    -------
    numpy.ndarray
        wavefront slope (slope error). 
        Unit in :math:`\mu rad`.
    """

    return delay * pixsize / (dist * 1.e-3)     # [urad]

def _make_window_SCS(height, width):
    """
    Make a window for a normal integration method:
    the SCS (Simchony, Chellappa, and Shao) method.
    """
    ulist = 1.0 * np.arange(0, width) / width
    vlist = 1.0 * np.arange(0, height) / height
    u, v = np.meshgrid(ulist, vlist)
    sin_u = np.sin(2 * np.pi * u)
    sin_v = np.sin(2 * np.pi * v)
    sin_u2 = np.power(np.sin(np.pi * u), 2)
    sin_v2 = np.power(np.sin(np.pi * v), 2)
    window = (sin_u2 + sin_v2)
    window[0, 0] = 1.0
    window = 1 / (4 * 1j * window)
    window[0, 0] = 0.0

    return sin_u, sin_v, window

def Integration2D_SCS(slope_x, slope_y, neg_corr=True, pad=0, pad_mode="linear_ramp"):
    """
    Reconstruct a surface from the gradients in x and y-direction using the
    Simchony-Chellappa-Shao method. Note that the DC-component
    (average value of an image) of the reconstructed image is unidentified
    because the DC-component of the FFT-window is zero.
    Assuming the space of the mesh grid is 1 :math:`\mu m`.

    The SCS method can be found from [SCSIntegration]_.

    .. [SCSIntegration] Simchony T, Chellappa R, Shao, M.
            Direct analytical methods for solving Poisson equations in computer vision problems
            IEEE transactions on pattern analysis and machine intelligence 
            1990, 12(5), 435-446.
            https://doi.org/10.1109/34.55103

    Parameters
    ----------
    slope_x : numpy.ndarray 
        2D array. Wavefront slope in x-direction, in :math:`\mu rad`.
    slope_y : numpy.ndarray 
        2D array. Wavefront slope in y-direction, in :math:`\mu rad`.
    neg_corr : bool, optional
        Correct negative offset if True.
    pad : int
        Padding width.
    pad_mode : str
        Padding method. Full list can be found at numpy.pad documentation.

    Returns
    -------
    numpy.ndarray
        2D array. Reconstructed surface. In [pm].
    """
    if pad != 0:
        slope_x = np.pad(slope_x, pad, mode=pad_mode)
        slope_y = np.pad(slope_y, pad, mode=pad_mode)
    height, width = slope_x.shape
    sin_u, sin_v, win = _make_window_SCS(height, width)
    fmat_x = sin_u * np.fft.fft2(slope_x)
    fmat_y = sin_v * np.fft.fft2(slope_y)
    fmat = fmat_x + fmat_y
    rec_surf = np.real(np.fft.ifft2(fmat * win))
    if pad != 0:
        rec_surf = rec_surf[pad:-pad, pad:-pad]
    if neg_corr:
        nmin = np.min(rec_surf)
        if nmin < 0.0:
            rec_surf = rec_surf - 2 * nmin

    return np.float32(rec_surf)

def _double_image(mat):
    mat1 = np.hstack((mat, np.fliplr(mat)))
    mat2 = np.vstack((np.flipud(mat1), mat1))

    return mat2

def _make_window_FC(height, width):
    """
    Make a window for a normal integration method:
    the FC (Frankot and Chellappa) method.
    """
    xcenter = width // 2
    ycenter = height // 2
    ulist = (1.0 * np.arange(0, width) - xcenter) / width
    vlist = (1.0 * np.arange(0, height) - ycenter) / width
    u, v = np.meshgrid(ulist, vlist)
    window = u ** 2 + v ** 2
    window[ycenter, xcenter] = 1.0
    window = 1 / window
    window[ycenter, xcenter] = 0.0

    return u, v, window

def Integration2D_FC(slope_x, slope_y, neg_corr=True):
    """
    Reconstruct a surface from the gradients in x and y-direction using the
    Frankot-Chellappa method. Note that the DC-component
    (average value of an image) of the reconstructed image is unidentified
    because the DC-component of the FFT-window is zero.
    Assuming the space of the mesh grid is 1 :math:`\mu m`.

    Frankot-Chellappa method can be found from [FCIntegration]_.

    .. [FCIntegration] Frankot, R, & Chellappa, R.
             A method for enforcing integrability in shape from shading algorithms
             IEEE Transactions on pattern analysis and machine intelligence, 
             1998, 10(4), 439-451.
             https://doi.org/10.1109/34.3909

    Parameters
    ----------
    slope_x : numpy.ndarray 
        2D array. Wavefront slope in x-direction, in :math:`\mu rad`.
    slope_y : numpy.ndarray 
        2D array. Wavefront slope in y-direction, in :math:`\mu rad`.
    neg_corr : bool, optional
        Correct negative offset if True.

    Returns
    -------
    numpy.ndarray 
        2D array. Reconstructed surface. In [pm].
    """
    height, width = slope_x.shape
    slope2_x = _double_image(slope_x)
    slope2_y = _double_image(slope_y)
    height2, width2 = slope2_x.shape
    u, v, win = _make_window_FC(height2, width2)
    fmat_x = -1j * u * np.fft.fftshift(np.fft.fft2(slope2_x))
    fmat_y = -1j * v * np.fft.fftshift(np.fft.fft2(slope2_y))
    rec_surf = (0.5 / np.pi) * np.real(
        np.fft.ifft2(np.fft.ifftshift((fmat_x + fmat_y) * win)))[height:, 0:width]
    if correct_negative:
        nmin = np.min(rec_surf)
        if nmin < 0.0:
            rec_surf = rec_surf - 2 * nmin

    return np.float32(rec_surf)

def curv_scan(delay, scanstep, dist, pixsize, nstep, mempos):
    """
    Get wavefront local curvature (curvature error) from tracked shift.
    Used when scanned. The corresponding wavefront is **on the detector plane**.

    This function corresponds to Eq. (3) in [WangXSSOEpaper]_.

    .. [WangXSSOEpaper] Wang H, Sutter J, Sawhney K. 
               Advanced in situ metrology for x-ray beam shaping with super precision. 
               Opt Express 2015, 23(2): 1605-1614.
               https://doi.org/10.1364/OE.23.001605

    Parameters
    ----------
    delay: numpy.ndarray
        Tracked shift in the scanned direction.
        :math:`\epsilon` in the paper, it must be **positive**.
    scanstep : float
        Step size of piezo [:math:`\mu m`], :math:`\mu` in the paper.
    dist : float
        Distance from diffusor to detector plane,
        if it's the downstream mode [:math:`mm`]. 
        If the diffuser is placed in the upstream,
        usually it is set to be the distance between the centre 
        of the optic to the detector. :math:`d` in the paper.
    pixsize : float
        Detector pixel size [:math:`\mu m`], :math:`p` in the paper
    nstep : int
        ntep, :math:`(j-i)` in the paper
    mempos : str
        'downstream' or 'upstream'. Use this to define the position of 
        the diffusor in respect to the focus of the optics. 'downstream'
        means the diffuser is placed downstream of the focus. See 
        :ref:`Local curvature reconstruction <curvature>` for more 
        details of it. 

    Returns
    -------
    curvature : numpy.ndarray   
        Local curvature [m^-1]
    """
    dist *= 1.e-3           # [m]
    delay = np.abs(delay)
    mag = delay * scanstep / nstep / pixsize 
    if mempos == 'downstream':
        curvature = (1 - mag) / dist 
    if mempos == 'upstream':
        curvature = (1 + mag) / dist 

    return curvature              #[1/m]

def curv_XST(delay, scanstep, dist, pixsize, mempos):
    """
    Get wavefront local curvature (curvature error) from tracked shift.
    Used for self-reference XST technique. 
    The corresponding wavefront is **on the detector plane**.

    This function corresponds to Eq. (2) of [HuXSTOEpaper]_.

    .. [HuXSTOEpaper] Hu, L., Wang, H., Fox, O., & Sawhney, K. (2022). 
             Fast wavefront sensing for X-ray optics with an alternating speckle tracking technique. 
             Opt. Exp., 30(18), 33259-33273.
             https://doi.org/10.1364/OE.460163

    Parameters
    ----------
    delay: numpy.ndarray
        Tracked shift in the scanned direction.
        :math:`iy_{AST}` in the paper, it must be **positive**.
    scanstep : float
        Step size of piezo [:math:`\mu m`], :math:`s` in the paper.
    dist : float
        Distance from diffusor to detector plane,
        if it's the downstream mode [mm]. 
        If the diffuser is placed in the upstream,
        usually it is set to be the distance between the centre 
        of the optic to the detector. :math:`D` in the paper.
    pixsize : float
        Detector pixel size [:math:`\mu m`], :math:`p` in the paper.
    mempos : str
        'downstream' or 'upstream'. Use this to define the position of 
        the diffusor in respect to the focus of the optics. 'downstream'
        means the diffuser is placed downstream of the focus. See 
        :ref:`Local curvature reconstruction <curvature>` for more 
        details of it. 

    Returns
    -------
    curvature : numpy.ndarray   
        Local curvature [m^-1]
    """
    dist *= 1.e-3           # [m]
    delay = np.abs(delay)
    mag = scanstep / (delay * pixsize)
    if mempos == 'downstream':
        curvature = (1 - mag) / dist 
    if mempos == 'upstream':
        curvature = (1 + mag) / dist 

    return curvature              #[1/m]

def EllipseSlope(x, p, q, theta):
    """
    This function calculate the slope error at coordinate x. 
    The resulting slope error always starts with 0.

    Parameters
    ----------
    x : numpy.array
        Ellipse x coordinate, x starts from 0. [m]
    p : float
        Ellipse p, [m]
    q : float
        Ellipse q, [m]
    theta : float
        Ellipse theta, [rad]

    Returns
    -------
    Slo : numpy.array
        Ellipse slope error at x coordinate, starts from 0. [rad]

    """
    a = 0.5 * (p + q)
    b = np.sqrt(p * q) * np.sin(theta)
    f = np.sqrt(p**2 + q**2 + 2 * p * q * np.cos(2*theta))
    xc = (p**2 - q**2) / (2 * f)
    yc = -(p * q * np.sin(2*theta)) / f
    x_ellip = xc + x - 0.5 * (x[-1]-x[0])
    y_ellip = -b * np.sqrt(1 - x_ellip**2/a**2)
    Slo = - b**2 * x_ellip / (a**2 * y_ellip) 
    Slo = Slo - Slo[0]

    return Slo
