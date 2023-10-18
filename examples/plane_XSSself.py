"""
Measurement of the wavefront local curvature after a plane mirror
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
    folderName = "/dls/science/groups/b16/SpeckleData/planemXSSself/Xscan/354343-pcoedge-files/"
    ROI = [180, 1980, 690, 1270]   # [y_start, y_end, x_start, x_end]
    imstack = Imagestack(folderName, ROI)
    track_XSS = Tracking(imstack)
    track_XSS.dimension = '2D'
    track_XSS.scandim = 'x'
    track_XSS.dist = 1705.0    #[mm]
    track_XSS.pixsize = 3.0    #[um]
    track_XSS.scanstep = 1.0    #[um]

    edge_x = 0
    edge_y = 10
    edge_z = 10
    nstep = 2
    width = 30
    pad_xy = 10
    normalize = True
    #track_XSS.XSS_self(edge_x, edge_y, edge_z, nstep, width, pad_xy, normalize, display=True)
    cpu_no = 16
    track_XSS.XSS_self_multi(edge_x, edge_y, edge_z, nstep, width, pad_xy, cpu_no, normalize)

    flatFolder = "/dls/science/groups/b16/SpeckleData/planemXSSself/FlatField/354340-pcoedge-files/"
    imstack2 = Imagestack(flatFolder, ROI)
    imstack2.read_data()
    ffimage = np.mean(imstack2.data, axis=0)

    ShowImage = True
    if ShowImage:
        plt.figure()
        plt.imshow(track_XSS.curvX, vmin=-0.15, vmax=0.2)
        plt.colorbar(label='1/R [1/m]')
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')

        plt.figure()
        plt.imshow(ffimage, vmin=2000, vmax=3500)
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')

        plt.show()
