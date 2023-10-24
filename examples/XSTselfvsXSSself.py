"""
Comparison between self-reference XSS technique and self-reference XST technique
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('../..'))
sys.path.append(os.path.join('../../..'))
from spexwavepy.imstackfun import Imagestack
from spexwavepy.trackfun import Tracking 
from spexwavepy.corefun import read_one, crop_one
from spexwavepy.postfun import curv_scan_XST 

_PLOT = False#True
if __name__ == "__main__":
    #data_folder = "/dls/science/groups/b16/SpeckleData/example_2/"
    data_folder = "/home/lingfei/spexwavepy/tmp/example_2/"
    ROI = [600, 1600, 740, 2040]

    imstack = Imagestack(data_folder, ROI) 

    track_XSS = Tracking(imstack)
    track_XSS.dimension = '2D' #'1D'
    track_XSS.scandim = 'x'
    track_XSS.dist = 833.   # [mm] 
    track_XSS.scanstep = 1.0  # [um]
    track_XSS.pixsize = 1.07  # [um]

    edge_x = 10
    edge_y = 10
    edge_z = 10
    nstep = 2

    if track_XSS.dimension == '1D':
        track_XSS.XSS_self(edge_x, edge_y, edge_z, nstep, display=False, normalize=True)
    if track_XSS.dimension == '2D':
        pad_xy = 10
        hw_xy = 20
        cpu_no = 30#16
        #track_XSS.XSS_self(edge_x, edge_y, edge_z, nstep, hw_xy, pad_xy, display=True, normalize=True)
        #track_XSS.XSS_self_multi(edge_x, edge_y, edge_z, nstep, hw_xy, pad_xy, cpu_no, normalize=True)

    imstack_1 = Imagestack(data_folder, ROI) 
    imstack_1.fnum = 1
    imstack_1.fstart = 0

    imstack_2 = Imagestack(data_folder, ROI) 
    imstack_2.fnum = 1
    imstack_2.fstart = 5 

    track_XST = Tracking(imstack_1, imstack_2)
    track_XST.dimension = '2D' #'1D'
    track_XST.scandim = 'x'
    track_XST.dist = 833.   # [mm] 
    track_XST.scanstep = 5.0  # [um]
    track_XST.pixsize = 1.07  # [um]

    edge_x = [20, 20]
    edge_y = [20, 25]
    pad_x = [20, 20]
    hw_xy = 30
    pad_y = [20, 25]

    if track_XST.dimension == '1D':
        edge_x = [20, 20]
        edge_y = 10
        pad_x = [20, 20]
        hw_xy = 15
        pad_y = 10

        #track_XST.XST_self(edge_x, edge_y, pad_x, pad_y, hw_xy, display=False, normalize=True)

    if track_XST.dimension == '2D':
        edge_x = [20, 20]
        edge_y = [20, 25]
        pad_x = [20, 20]
        hw_xy = 30
        pad_y = [20, 25]
        window = 60
        cpu_no = 30 #16

        #track_XST.XST_self(edge_x, edge_y, pad_x, pad_y, hw_xy, window, display=True, normalize=True)
        #track_XST.XST_self_multi(edge_x, edge_y, pad_x, pad_y, hw_xy, window, cpu_no, normalize=True)


    if _PLOT:
        if track_XSS.dimension == '1D' and track_XST.dimension == '1D':
            plt.figure()
            plt.plot(track_XSS.curvX[20:], label='XSS technique')
            plt.plot(track_XST.curvX, label='XST technique')
            plt.xlabel('CCD pixel')
            plt.ylabel('Wavefront curvature')
            plt.legend()
            plt.grid(True)

            plt.figure()
            plt.plot(track_XSS.resX)
            plt.ylabel('Tracking coeffcient')
            plt.title('XSS technique')
            plt.grid(True)

            plt.figure()
            plt.plot(track_XST.resX)
            plt.ylabel('Tracking coeffcient')
            plt.title('XST technique')
            plt.grid(True)

        if track_XSS.dimension == '2D' and track_XST.dimension == '2D':
            plt.figure()
            plt.imshow(track_XSS.curvX, cmap='jet', vmin=-1.9, vmax=1.2)
            plt.xlabel('CCD pixel')
            plt.ylabel('Wavefront curvature [m^-1]')
            plt.colorbar()

            plt.figure()
            plt.imshow(-track_XST.curvX, cmap='jet', vmin=-1.9, vmax=1.2)
            plt.xlabel('CCD pixel')
            plt.ylabel('Wavefront curvature [m^-1]')
            plt.colorbar()

        plt.show()

