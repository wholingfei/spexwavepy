===============
Getting started
===============
*Description goes here ...*

.. _install:

Installing spexwavepy
=====================
*Say something...*

.. _tutorial:

Tutorial
========
The **spexwavepy** is a data processing package for speckle-based X-ray wavefront sensing techniques. 
In this tutorial, we try to help new users to get familiar with the basic concept and the usage of this package.
Let's do it step by step.

.. _tuimstack:

1. Read the image stack
-----------------------
An image stack is the first class you will create before any other operations. 
To create it, we need to import ``imstackfun`` module.

.. code-block:: Python

   from spexwavepy.imstackfun import Imagestack

After that, we can create an instance of Imagestack. 
Two mandatory parameters are needed to initialize an Imagestack. 
One is the data file folder, and the other one is the region of interest (ROI) in the raw images which will be cropped.  

.. code-block:: Python

   fileFolder = "/dls/b16/data/2023/cm33912-1/pixelsizestep10um/402724-pcoedge-files/"
   ROI = [0, 3500, 0, 4500]           #[y_start, y_end, x_start, x_end]
   Imstack_1 = Imagestack(fileFolder, ROI)

The above codes create a Imagestack instance ``Imstack_1``. 
The raw images are stored in the ``fileFolder``. 
The ``ROI`` in this case is larger than the real figure size,
thus, contains the whole image range.
Other parameters, such as start image number, total image number, etc., 
can be used to define how to load the images in the :py:class:`~spexwavepy.imstackfun.Imagestack`.

Until now, we have only defined one ``Imagestack`` instance, no real data has been loaded.
To load the raw data into the memory, method :py:meth:`~spexwavepy.imstackfun.Imagestack.read_data`
needs to be called.

.. code-block:: Python

   Imstack_1.read_data()

After the above operation, the raw data will be stored in the attribute 
:py:attr:`~spexwavepy.imstackfun.Imagestack.rawdata`.
The rawdata is read-only. Another attribute 
:py:attr:`~spexwavepy.imstackfun.Imagestack.data` 
is used to store the data to be processed in the future.

It is also possible that we only need to one image from the folder in the beginning.
In this case, we can use the :py:func:`~spexwavepy.corefun.read_one` function from 
the :py:mod:`~spexwavepy.corefun` module. If you want to crop the raw image, 
you need to import :py:func:`~spexwavepy.corefun.crop_one` function too.

.. code-block:: Python

   from spexwavepy.corefun import read_one, crop_one

To call :py:func:`~spexwavepy.corefun.read_one` function, 
you need to input the file path that you want to read.

.. code-block:: Python

   filepath = "/dls/b16/data/2023/cm33912-1/pixelsizestep10um/402724-pcoedge-files/00005.tif"
   im_raw = read_one(filepath, ShowImage=True)

If ``ShowImage`` is set to be True, then it will show the image.

.. image:: _static/readone.png
   :width: 80%

Usually we need to crop the raw image for future processing, so we provide ROI for 
:py:func:`~spexwavepy.corefun.crop_one` function.

.. code-block:: Python

   filepath = "/dls/b16/data/2023/cm33912-1/pixelsizestep10um/402724-pcoedge-files/00005.tif"
   ROI = [750, 1500, 500, 2000]    #[y_start, y_end, x_start, x_end]
   im_crop = crop_one(im_raw, ROI, ShowImage=True)

Again, if ``ShowImage`` is set to be True, then it will show the cropped image.

.. image:: _static/cropone.png
   :width: 80%

.. _tudetpix:

2. Determine the detector pixel size
------------------------------------
In many cases, e.g., B16 Test beamline at Diamond Light Source, 
the pixel size of the detector used for data acquisition is changeable. 
The first step is to determine the pixel size.

``Imstack_1`` has already loaded the data used for detector pixel size determination,
we use the :py:meth:`~spexwavepy.imstackfun.Imagestack.getpixsize` method to calculate the pixel size.
The parameters that needed are ``subROI``, ``dim`` and ``step``.
``subROI`` is the ROI used for image matching. 
``dim`` is either 'x' or 'y' to indicate in which direction the speckle generator was scanned.
``step`` is the scan step in unit of :math:`\mu m`.

.. note::
   Note that the image stack has been cropped according to ROI. 
   Thus, ``subROI`` is the region on the cropped images from the cropped image stack,
   **NOT** the coordinates on the raw images.  

.. code-block:: Python

   subROI = [1500, 2000, 500, 2000]      #[y_start, y_end, x_start, x_end]
   dim = 'x'
   step = 10.0                           #[um]
   pixsize = Imstack_1.getpixsize(subROI, dim, step)

Please refer to the :ref:`detector pixel size determination <usedetpix>` in the user guide to 
find out how we use speckle patterns to determine the pixel size.

.. _tustable:

3. Stability check
------------------
Using speckle patterns to monitor the stability of the beamline is 
a very simple use of the speckle-based technique. 
The images are acquired when all the hardware is fixed. 
The stability is monitored by comparing the images in the whole folder with the **first** one.

To enable the stability check, 
a class called :py:class:`~spexwavepy.trackfun.Tracking` needs to be initialized.

.. code-block:: Python 

   from spexwavepy.trackfun import Tracking 

Depending on the type of speckle-based technique that is used, 
one image stack or two image stacks are needed to initialize :py:class:`~spexwavepy.trackfun.Tracking`.
For stability checking, only one image stack is needed. 

.. code-block:: Python 

   from spexwavepy.imstackfun import Imagestack

   fileFolder = "/dls/science/groups/b16/SpeckleData/example_1/"
   ROI = [0, 3500, 0, 4500]           #[y_start, y_end, x_start, x_end]
   Imstack_1 = Imagestack(fileFolder, ROI)
   Imstack_1.fnum = 99   #File number to be used for stability check
   Imstack_1.fstart = 0   #File start number to be used for stability check
   Imstack_1.dim = 'both'

   track = Tracking(Imstack_1)

Usually, there will be plenty of raw images in one folder. 
There is no need to load all the data into memory for stability checking. 
The :py:meth:`~spexwavepy.trackfun.Tracking.stability` method is used for this purpose.
``edge_x`` and ``edge_y`` are the two parameters needed.
``edge_x`` and ``edge_y`` can be either a single integer or a list of two integers, 
like [int1, int2]. If input as a single integer int0, 
``edge_x`` and ``edge_y`` will be expanded as a list of two integers, 
the elements in the list are the same, i.e., [int0, int0]. 

.. code-block:: Python 
   
   edge_x, edge_y = 10, 10
   delayX, delayY, res = track.stability(edge_x, edge_y)

Please refer to :ref:`stability check using speckle patterns <trastable>` in the user guide
to see how to do the stability checking using speckle patterns. 

.. image:: _static/stableres.png
   :width: 80%

We can also use multi-cores to accelerate the calculation.

.. code-block:: Python 
   
   cpu_no = 16
   edge_x, edge_y = 10, 10
   delayX, delayY, res = track.stability_multi(edge_x, edge_y, cpu_no)

The :py:meth:`~spexwavepy.trackfun.Tracking.stability_multi` method uses the 
built-in ``multiprocessing`` package.

.. warning::
   Please check the available CPUs before calling :py:meth:`~spexwavepy.trackfun.Tracking.stability_multi` method. 

.. _tuCRL:

4. Single CRL measurement
-------------------------
*In this section we will show how to obtain a single CRL wavefront using X-ray Speckle Scanning (XSS) ... Then say more things...*

First, let us load and see the raw images.

.. code-block:: Python

   from spexwavepy.imstackfun import Imagestack
   from spexwavepy.trackfun import Tracking 
   from spexwavepy.corefun import read_one, crop_one

   ref_folder_x = "/dls/science/groups/b16/SpeckleData/CRLXSS/ReferX1D/402923-pcoedge-files/"
   sam_folder_x = "/dls/science/groups/b16/SpeckleData/CRLXSS/SampleX1D/402924-pcoedge-files/"
   ref_folder_y = "/dls/science/groups/b16/SpeckleData/CRLXSS/ReferY1D/402925-pcoedge-files/"
   sam_folder_y = "/dls/science/groups/b16/SpeckleData/CRLXSS/SampleY1D/402926-pcoedge-files/"

   im_sam_tmp = read_one(sam_folder_y+'00005.tif', ShowImage=True)

.. image:: _static/crlraw.png
   :width: 80%

The raw image of the single CRL shows that we only need to crop the central part of it 
for future processing. 
The reference image should use the same ROI as the CRL image.

.. code-block:: Python

   ROI_sam = [540, 1570, 750, 1800]
   ROI_ref = ROI_sam
   im_crop_tmp = crop_one(im_sam_tmp, ROI_sam, ShowImage=True)
   im_ref_tmp = read_one(ref_folder_y+'00005.tif', ShowImage=True)
   im_crop_tmp2 = crop_one(im_ref_tmp, ROI_sam, ShowImage=True)
 

.. image:: _static/crlcrop.png
   :width: 90%

.. code-block:: Python

   Imstack_sam_x = Imagestack(sam_folder_x, ROI_sam)
   Imstack_ref_x = Imagestack(ref_folder_x, ROI_ref)
   Imstack_sam_y = Imagestack(sam_folder_y, ROI_sam)
   Imstack_ref_y = Imagestack(ref_folder_y, ROI_ref)

   Imstack_sam_x.normalize = True
   Imstack_ref_x.normalize = True
   Imstack_sam_y.normalize = True
   Imstack_ref_y.normalize = True

Since we are going to use XSS technique with reference beam, 
we need at least two image stacks to initialize the :py:class:`~spexwavepy.trackfun.Tracking` class.
The first image stack to be loaded is the template image stack, i.e., the image stack with test optic.
The second image stack is the reference image stack.
In this example, we would like to obtain the 2D slope map in two directions from two
1D scans. Thus we need four image stacks. Two for references and two for samples, respectively. 
The first two image stacks are the template image stack and reference image stack 
in the x (horizontal) scan direction. The last two image stacks are those in the
y (vertical) direction. Also, we choose to :ref:`normalize <usenorm>` these image stacks in the beginning.

.. code-block:: Python

   track_XSS = Tracking(Imstack_sam_x, Imstack_ref_x, Imstack_sam_y, Imstack_ref_y)

Before we do the real tracking, there are some parameters to be set for ``track_XSS``. 
``dimension`` is set to be `'1D'` or `'2D'`. 
We use it to tell the code to do `1D` or `2D` data processing. ``scandim`` is used to tell
the code the `scan direction` of the loaded image stack. For XSS technique, it supports
`'x', 'y', 'xy' and 'diag'`. In this case, we use `'xy'`. That means we will obtain the
speckle shifts from both x (horizontal) 1D scan and y (vertical) 1D scan all together.
Thus, 4 image stacks are loaded. Besides, we need to provide
``dist``, ``pixsize`` and ``scanstep``. They are `distance between diffuser and detector plane` in 
mm, `detector pixel size` in :math:`\mu m` and `scan step size` in :math:`\mu m`, repectively.


.. code-block:: Python

   track_XSS.dimension = '2D'
   track_XSS.scandim = 'xy'
   track_XSS.dist = 623.    #[mm]
   track_XSS.pixsize = 1.03    #[um]
   track_XSS.scanstep = 1.0    #[um]


There are several compulsory input for 
:py:func:`~spexwavepy.trackfun.Tracking.XSS_withrefer` function before we call it. 
``edge_x``, ``edge_y`` and ``edge_z`` define the edges of the raw images in the image stack to be
cut in order to be trackable. ``width`` is the window width used for 1D tracking, 
``pad_xy`` determines how large the extra area needed for the reference image. 
See :ref:`User guide <traXSS>` for detailed description.

.. code-block:: Python

   edge_x = 20
   edge_y = 20
   edge_z = 8
   width = 30
   pad_xy = 20

Then we call :py:func:`~spexwavepy.trackfun.Tracking.XSS_withrefer` function. 
In the beginning, we can set the ``display`` to be ``True`` to have a check
of the settings of all the parameters.

.. code-block:: Python

   track_XSS.XSS_withrefer(edge_x, edge_y, edge_z, width, pad_xy, display=True)

.. note::
   
   Double click the mouse will terminate the display.

If every parameter is set appropriately, the following window will appear. 
The top-left is the template image, the top-right is the reference image,
the bottom-left is the tracking coefficient matrix, the bottom-right is the central cut 
of the matrix.

.. image:: _static/XSSdisplay.png
   :width: 80%

Switch off ``display`` if we want to do the real calculation.

.. code-block:: Python

   track_XSS.XSS_withrefer(edge_x, edge_y, edge_z, width, pad_xy, display=False)

If you have multicores, you can also use the multi-core version of this function,
:py:func:`~spexwavepy.trackfun.Tracking.XSS_withrefer_multi`. The only additional
parameter is ``cpu_no``.

.. code-block:: Python

   track_XSS.XSS_withrefer_multi(edge_x, edge_y, edge_z, width, pad_xy, cpu_no=16)

After calling the :py:func:`~spexwavepy.trackfun.Tracking.XSS_withrefer` or 
:py:func:`~spexwavepy.trackfun.Tracking.XSS_withrefer_multi` function,
the 2D shift map in both x and y direction are stored in the ``delayX`` and 
``delayY`` attribute. Likewise, the 2D slope map are stored in the 
``sloX`` and ``sloY`` attribute.

.. code-block:: Python

   plt.imshow(track_XSS.delayX, cmap='jet')
   plt.xlabel('x [pixel]')
   plt.ylabel('y [pixel]')
   plt.colorbar()
   plt.title('Shift in x direction')

   plt.figure()
   plt.imshow(track_XSS.delayY, cmap='jet')
   plt.xlabel('x [pixel]')
   plt.ylabel('y [pixel]')
   plt.colorbar()
   plt.title('Shift in y direction')

   plt.figure()
   plt.imshow(track_XSS.sloX, cmap='jet')
   plt.xlabel('x [pixel]')
   plt.ylabel('y [pixel]')
   plt.colorbar(label=r'$\mu$rad')
   plt.title('Slope in x direction')

   plt.figure()
   plt.imshow(track_XSS.sloY, cmap='jet')
   plt.xlabel('x [pixel]')
   plt.ylabel('y [pixel]')
   plt.colorbar(label=r'$\mu$rad')
   plt.title('Slope in y direction')

.. image:: _static/XSS_delayandslope.png
   :width: 90%

We know that the tested single CRL has a surface of paraboloid of revolution.
Thus, its 2D slope map will be a tilted plane. Let's calculate the slope of this plane.
We extract the central horizontal line from the slope map in horizontal direction.

.. code-block:: Python

   plt.figure()
   plt.plot(track_XSS.sloX[500, :], label='Raw data')

The curve of slope in the central part can be fitted with a straight line, 
if we cut the edge of this curve.

.. code-block:: Python

   sloX_cen = track_XSS.sloX[500, :]
   sloX_cen_fit = sloX_cen[200:800]
   sloX_coord = np.arange(200, 800, 1)
   fit_para_X = np.polyfit(sloX_coord, sloX_cen_fit, deg=1)

We plot the fitted line and the raw curve together.

.. code-block:: Python

   plt.plot(np.arange(200, 800, 1), track_XSS.sloX[500, 200:800], label='Partial data')
   x_plot = np.arange(1, len(track_XSS.sloX[500, :])+1, 1)
   plt.plot(x_plot, fit_para_X[0]*x_plot+fit_para_X[1], label='Fitted line')
   plt.legend()
   plt.xlabel('Pixel')
   plt.ylabel('Slope ['+'$\mu rad$'+']')
   plt.title('X slope')

.. image:: _static/XSS_fitx.png
   :width: 80%

Likewise, we do the same thing on y direction.

.. code-block:: Python

   sloY_cen = track_XSS.sloY[:, 450]
   sloY_cen_fit = sloY_cen[200:780]
   sloY_coord = np.arange(200, 780, 1)
   fit_para_Y = np.polyfit(sloY_coord, sloY_cen_fit, deg=1)

   plt.figure()
   plt.plot(track_XSS.sloY[:, 450], label='Raw data')
   plt.plot(np.arange(200, 780, 1), track_XSS.sloY[200:780, 450], label='Partial data')
   y_plot = np.arange(1, len(track_XSS.sloY[:, 450])+1, 1)
   plt.plot(y_plot, fit_para_Y[0]*y_plot+fit_para_Y[1], label='Fitted line')
   plt.legend()
   plt.xlabel('Pixel')
   plt.ylabel('Slope ['+'$\mu rad$'+']')
   plt.title('Y slope')

.. image:: _static/XSS_fity.png
   :width: 80%

Let's check the fitting parameter in both directions. 

.. code-block:: Python

   print("Fiiting parameters in x direction:", fit_para_X)
   print("Fiiting parameters in y direction:", fit_para_Y)
 
.. parsed-literal::
   Fitting parameters in x direction: [ 0.01470974 -6.86428282]
   Fitting parameters in y direction: [ 0.01475435 -7.04149456]

We can see the slope of the two fitted straight lines are very close.
The slope for the ideal single 2D CRL in both directions is a tilted plane.
We use the fitting parameters to generated this plane.

.. code-block:: Python

   y_dim_tmp, _ = track_XSS.sloX.shape
   planeXcoord = np.arange(1, len(sloX_cen)+1, 1)
   planeX = planeXcoord * fit_para_X[0] + fit_para_X[1]
   planeX = np.array([list(planeX)] * y_dim_tmp)

   _, x_dim_tmp = track_XSS.sloY.shape
   planeYcoord = np.arange(1, len(sloY_cen)+1, 1)
   planeY = planeYcoord * fit_para_Y[0] + fit_para_Y[1]
   planeY = np.rot90(np.array([list(planeY)] * x_dim_tmp), k=-1)

To estimate the slope error, we subtract the fitted plane.

.. code-block:: Python

   sloErr_x = track_XSS.sloX - planeX
   sloErr_y = track_XSS.sloY - planeY

After that, we plot the 2D map of the slope error.

.. code-block:: Python

   plt.figure()
   y_dim_tmp, x_dim_tmp = track_XSS.sloX.shape
   plt.imshow(sloErr_x, cmap='jet', vmin=-0.5, vmax=0.5, extent=[0, x_dim_tmp*track_XSS.pixsize, y_dim_tmp*track_XSS.pixsize, 0])
   plt.colorbar(label=r'$\mu rad$')
   plt.xlabel(r'$\mu m$')
   plt.ylabel(r'$\mu m$')
   plt.title('Slope error in X direction')

   plt.figure()
   y_dim_tmp, x_dim_tmp = track_XSS.sloY.shape
   plt.imshow(sloErr_y, cmap='jet', vmin=-0.5, vmax=0.5, extent=[0, x_dim_tmp*track_XSS.pixsize, y_dim_tmp*track_XSS.pixsize, 0])
   plt.colorbar(label=r'$\mu rad$')
   plt.xlabel(r'$\mu m$')
   plt.ylabel(r'$\mu m$')
   plt.title('Slope error in Y direction')

.. image:: _static/XSSsloerr.png
   :width: 90%

Next, we can do 2D integration to obtain the surface of the wavefront.

.. code-block:: Python

   surface = Integration2D_SCS(track_XSS.sloX, track_XSS.sloY) 

For the 2D integration, please see the :ref:`User guide <use2Dint>` for details.
The x and y coordinate in the 2D integrations are in the unit of :math:`\mu m`. 
The output height is in the unit of pm. 
Let's see the integrated surface.

.. code-block:: Python

   plt.figure()
   plt.imshow(surface, cmap='jet')

.. image:: _static/XSSintsurf.png
   :width: 60%

The surface should be cutted in order to be fitted.

.. code-block:: Python

   surface2fit = surface[200:750, 150:750]

   plt.figure()
   plt.imshow(surface2fit, cmap='jet')

.. image:: _static/XSSintsurfcut.png
   :width: 60%

The ideal wavefront after a single CRL is defined as: 

.. math::
   z = \frac{(x-x_0)^2+(y-y_0)^2}{2f} + z_0

We fit the measured wavefront to the above ideal function.

.. code-block:: Python

   def ideal_surf(data, x0, y0, R, z0):
       x = data[0]
       y = data[1]

       return ((x-x0)**2 + (y-y0)**2) / R + z0

   x_surf = np.arange(150, 750)
   y_surf = np.arange(200, 750)
   X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
   X = np.ravel(X_surf)
   Y = np.ravel(Y_surf)
   XY_data = [X, Y]
   Z_data = np.ravel(surface2fit)
   p_init = [(150+750)//2, (200+750)//2, 10, np.mean(Z_data)]
   popt, pcov = scipy.optimize.curve_fit(ideal_surf, XY_data, Z_data, p_init)

Since the real pixel size is 1.03 :math:`\mu m` rather than the assumed 1 :math:`\mu m`,
the real ``f`` should be 70.16 m.

.. code-block:: Python

   print("f is {:.4f} m.".format(popt[2]/2*track_XSS.pixsize))

.. parsed-literal::

   f is 70.1582 m.

The beam energy is 15.5 keV, the CRL is made of Be, the :math:`\delta` for
Be at 15.5 keV is around :math:`1.42\times 10^{-6}`.
According to the relation 

.. math::
   f = \frac{R}{2 \delta}

The ``R`` will be 199.25 :math:`\mu m`. It is close to 200 :math:`\mu m` 
which is the value the manufacturer provided. 

.. code-block:: Python

   delta=1.42 * 1.e-6
   print("R is {:.2f} um.".format(popt[2]*track_XSS.pixsize*delta*1.e6))

.. parsed-literal::

   R is 199.25 um.

From the parameters ``popt`` we can obtain the fitted surface.

.. code-block:: Python

   y_dim_tmp, x_dim_tmp = surface.shape
   x_plot = np.arange(0, x_dim_tmp)
   y_plot = np.arange(0, y_dim_tmp)
   X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
   surf_fit = (((X_plot-popt[0])**2+(Y_plot-popt[1])**2)/popt[2]+popt[3])     

The residual is ``surface``-``surf_fit``. Remeber that the real pixel size is 
1.03 :math:`\mu m` instead of 1 :math:`\mu m`, this factor should be multiplied.
Also, we cut the outside part of the CRL using a mask.

.. code-block:: Python

   residual = surface - surf_fit
   mask = 1 - (np.abs(residual)>20)*np.ones(residual.shape)
   residual = residual * mask * track_XSS.pixsize         #[pm]

Divide the residual with :math:`\delta`, we have the residual 
height error of single CRL. Also we can convert the wavefront 
surface to the CRL thinckness distribution.

.. code-block:: Python

   delta = 1.42 * 1.e-6
   T_residual = residual / (delta * 1.e6)                  #[um]
   T_crl = surface * track_XSS.pixsize / (delta * 1.e6) * 1.e-3  #[mm]

We display the 2D residual height error map.

.. code-block:: Python

   plt.figure()
   y_dim_tmp, x_dim_tmp = T_residual.shape
   plt.imshow(T_residual, cmap='jet', extent=[0, x_dim_tmp*track_XSS.pixsize, y_dim_tmp*track_XSS.pixsize, 0])
   plt.colorbar(label=r'$\mu m$')
   plt.xlabel(r'$\mu m$')
   plt.ylabel(r'$\mu m$')
   plt.title('Residual thickness error')

.. image:: _static/XSSheighterr.png
   :width: 80%

Likewise, we can also draw the 3D CRL height surface shape.

.. code-block:: Python

   from mpl_toolkits import mplot3d
   plt.figure()
   y_dim_tmp, x_dim_tmp = T_crl.shape
   ax = plt.axes(projection='3d')
   ax.plot_surface(X_plot*track_XSS.pixsize, Y_plot*track_XSS.pixsize, (T_crl-np.min(T_crl)), rstride=1, cstride=1, cmap='jet', edgecolor='none')
   ax.set_title('Be single CRL')
   ax.set_xlabel(r'$\mu$m')
   ax.set_ylabel(r'$\mu$m')
   ax.set_zlabel(r'mm')

.. image:: _static/XSScrlheight.png
   :width: 80%

