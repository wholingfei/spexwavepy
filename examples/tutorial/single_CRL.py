import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('../..'))
sys.path.append(os.path.join('../../..'))
from spexwavepy.imstackfun import Imagestack
from spexwavepy.trackfun import Tracking 
from spexwavepy.corefun import read_one, crop_one
from spexwavepy.postfun import Integration2D_SCS

if __name__ == "__main__":
    ref_folder_x = "/dls/science/groups/b16/SpeckleData/CRLXSS/ReferX1D/402923-pcoedge-files/"
    sam_folder_x = "/dls/science/groups/b16/SpeckleData/CRLXSS/SampleX1D/402924-pcoedge-files/"
    ref_folder_y = "/dls/science/groups/b16/SpeckleData/CRLXSS/ReferY1D/402925-pcoedge-files/"
    sam_folder_y = "/dls/science/groups/b16/SpeckleData/CRLXSS/SampleY1D/402926-pcoedge-files/"

    #im_sam_tmp = read_one(sam_folder_y+'00005.tif', ShowImage=True)
    #sys.exit(0)
    ROI_sam = [540, 1570, 750, 1800]
    ROI_ref = ROI_sam
    #im_crop_tmp = crop_one(im_sam_tmp, ROI_sam, ShowImage=True)
    #im_ref_tmp = read_one(ref_folder_y+'00005.tif', ShowImage=True)
    #im_crop_tmp2 = crop_one(im_ref_tmp, ROI_sam, ShowImage=True)

    #sys.exit(0)
    Imstack_sam_x = Imagestack(sam_folder_x, ROI_sam)
    Imstack_ref_x = Imagestack(ref_folder_x, ROI_ref)
    Imstack_sam_y = Imagestack(sam_folder_y, ROI_sam)
    Imstack_ref_y = Imagestack(ref_folder_y, ROI_ref)

    Imstack_sam_x.normalize = True
    Imstack_ref_x.normalize = True
    Imstack_sam_y.normalize = True
    Imstack_ref_y.normalize = True

    track_XSS = Tracking(Imstack_sam_x, Imstack_ref_x, Imstack_sam_y, Imstack_ref_y)

    track_XSS.dimension = '2D'
    track_XSS.scandim = 'xy'
    track_XSS.dist = 623.    #[mm]
    track_XSS.pixsize = 1.03    #[um]
    track_XSS.scanstep = 1.0    #[um]
    #sys.exit(0)

    edge_x = 20
    edge_y = 20
    edge_z = 8
    width = 30
    pad_xy = 20

    #track_XSS.XSS_withrefer(edge_x, edge_y, edge_z, width, pad_xy, display=False)
    track_XSS.XSS_withrefer_multi(edge_x, edge_y, edge_z, width, pad_xy, cpu_no=16)

    surface = Integration2D_SCS(track_XSS.sloX, track_XSS.sloY) 

    sloX_cen = track_XSS.sloX[500, :]
    sloX_cen_fit = sloX_cen[200:800]
    sloX_coord = np.arange(200, 800, 1)
    fit_para_X = np.polyfit(sloX_coord, sloX_cen_fit, deg=1)

    y_dim_tmp, _ = track_XSS.sloX.shape
    planeXcoord = np.arange(1, len(sloX_cen)+1, 1)
    planeX = planeXcoord * fit_para_X[0] + fit_para_X[1]
    planeX = np.array([list(planeX)] * y_dim_tmp)

    sloY_cen = track_XSS.sloY[:, 450]
    sloY_cen_fit = sloY_cen[200:780]
    sloY_coord = np.arange(200, 780, 1)
    fit_para_Y = np.polyfit(sloY_coord, sloY_cen_fit, deg=1)

    _, x_dim_tmp = track_XSS.sloY.shape
    planeYcoord = np.arange(1, len(sloY_cen)+1, 1)
    planeY = planeYcoord * fit_para_Y[0] + fit_para_Y[1]
    planeY = np.rot90(np.array([list(planeY)] * x_dim_tmp), k=-1)

    sloErr_x = track_XSS.sloX - planeX
    sloErr_y = track_XSS.sloY - planeY

    surface2fit = surface[200:750, 150:750]
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

    y_dim_tmp, x_dim_tmp = surface.shape
    x_plot = np.arange(0, x_dim_tmp)
    y_plot = np.arange(0, y_dim_tmp)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    surf_fit = (((X_plot-popt[0])**2+(Y_plot-popt[1])**2)/popt[2]+popt[3])     

    residual = surface - surf_fit
    mask = 1 - (np.abs(residual)>20)*np.ones(residual.shape)
    residual = residual * mask * track_XSS.pixsize         #[pm]

    delta = 1.42 * 1.e-6
    T_residual = residual / (delta * 1.e6)                  #[um]
    T_crl = surface * track_XSS.pixsize / (delta * 1.e6) * 1.e-3  #[mm]

    showimage = True
    if showimage:
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

        plt.figure()
        plt.plot(track_XSS.sloX[500, :], label='Raw data')
        plt.plot(np.arange(200, 800, 1), track_XSS.sloX[500, 200:800], label='Partial data')
        x_plot = np.arange(1, len(track_XSS.sloX[500, :])+1, 1)
        plt.plot(x_plot, fit_para_X[0]*x_plot+fit_para_X[1], label='Fitted line')
        plt.legend()
        plt.xlabel('Pixel')
        plt.ylabel('Slope ['+'$\mu rad$'+']')
        plt.title('X slope')

        plt.figure()
        plt.plot(track_XSS.sloY[:, 450], label='Raw data')
        plt.plot(np.arange(200, 780, 1), track_XSS.sloY[200:780, 450], label='Partial data')
        y_plot = np.arange(1, len(track_XSS.sloY[:, 450])+1, 1)
        plt.plot(y_plot, fit_para_Y[0]*y_plot+fit_para_Y[1], label='Fitted line')
        plt.legend()
        plt.xlabel('Pixel')
        plt.ylabel('Slope ['+'$\mu rad$'+']')
        plt.title('Y slope')

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

        plt.figure()
        plt.imshow(surface, cmap='jet')

        plt.figure()
        plt.imshow(surface2fit, cmap='jet')

        plt.figure()
        y_dim_tmp, x_dim_tmp = T_residual.shape
        plt.imshow(T_residual, cmap='jet', extent=[0, x_dim_tmp*track_XSS.pixsize, y_dim_tmp*track_XSS.pixsize, 0])
        plt.colorbar(label=r'$\mu m$')
        plt.xlabel(r'$\mu m$')
        plt.ylabel(r'$\mu m$')
        plt.title('Residual thickness error')

        from mpl_toolkits import mplot3d
        plt.figure()
        y_dim_tmp, x_dim_tmp = T_crl.shape
        ax = plt.axes(projection='3d')
        ax.plot_surface(X_plot*track_XSS.pixsize, Y_plot*track_XSS.pixsize, (T_crl-np.min(T_crl)), rstride=1, cstride=1, cmap='jet', edgecolor='none')
        ax.set_title('Be single CRL')
        ax.set_xlabel(r'$\mu$m')
        ax.set_ylabel(r'$\mu$m')
        ax.set_zlabel(r'mm')


        plt.show()
