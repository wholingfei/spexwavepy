"""
Plane mirror measurement with reference beam
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

if __name__ == "__main__":
    sam_folder = "/home/lingfei/spexwavepy/data/planeM2D/mirror/"  # The sample folder path
    ref_folder = "/home/lingfei/spexwavepy/data/planeM2D/reference/"  # The reference folder path
    
    #sam_im = read_one(sam_folder + '00001.tif', ShowImage=True)
    #ref_im = read_one(ref_folder + '00001.tif', ShowImage=True)
    
    ROI_sam = [10, 2150, 685, 2065]   # [y_start, y_end, x_start, x_end]
    ROI_ref = [10, 2150, 490, 2290]   # [y_start, y_end, x_start, x_end]

    #sam_im_crop = crop_one(sam_im, ROI_sam, ShowImage=True)
    #ref_im_crop = crop_one(ref_im, ROI_ref, ShowImage=True)

    imstack_sam = Imagestack(sam_folder, ROI_sam)
    imstack_ref = Imagestack(ref_folder, ROI_ref)
    imstack_ref.flip = 'x' 

    track_XSS = Tracking(imstack_sam, imstack_ref)
    track_XSS.dimension = '2D'
    track_XSS.scandim = 'x'
    track_XSS.dist = 833.     # [mm]
    track_XSS.scanstep = 1.0  # [um]
    track_XSS.pixsize = 1.07  # [um]
    
    track_XSS.collimate(10, 200)

    edge_x = 0
    edge_y = 30
    edge_z = [15, 30]
    width = 100
    pad_xy = 30

    #track_XSS.XSS_withrefer(edge_x, edge_y, edge_z, width, pad_xy, normalize=True, display=True)
    #track_XSS.XSS_withrefer_multi(edge_x, edge_y, edge_z, width, pad_xy, cpu_no=16, normalize=True)

    rotateang = -0.275       # [degree]
    imstack_sam.rotate(rotateang)
    cut = 20
    imstack_sam.data = imstack_sam.data[:,cut:-cut, cut:-cut]
    imstack_ref.data = imstack_ref.data[:,cut:-cut, cut:-cut]

    track_XSS = Tracking(imstack_sam, imstack_ref)
    track_XSS.dimension = '2D'
    track_XSS.scandim = 'x'
    track_XSS.dist = 833.     # [mm]
    track_XSS.scanstep = 1.0  # [um]
    track_XSS.pixsize = 1.07  # [um]

    #track_XSS.XSS_withrefer(edge_x, edge_y, edge_z, width, pad_xy, normalize=True, display=True)
    track_XSS.XSS_withrefer_multi(edge_x, edge_y, edge_z, width, pad_xy, cpu_no=16, normalize=True)

    imageplot = True
    if imageplot:
        plt.figure()
        plt.imshow(track_XSS.delayX, cmap='jet', vmin=-10, vmax=10)
        plt.colorbar()
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.title('X shift')

        plt.figure()
        #plt.imshow(track_XSS._delayY, cmap='jet', vmin=-4, vmax=4)
        plt.imshow(track_XSS._delayY, cmap='jet', vmin=-0.8, vmax=0.8)
        plt.colorbar()
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.title('Y shift')

        plt.figure()
        plt.plot(track_XSS._delayY[1000])
        plt.xlabel('Pixel')
        plt.ylabel('y shift [pixel]')
        plt.grid(True)


    if imageplot:
        plt.show()
