import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import copy
from matplotlib.patches import Rectangle

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('../..'))
sys.path.append(os.path.join('../../..'))
from spexwavepy.imstackfun import Imagestack
from spexwavepy.trackfun import Tracking 
from spexwavepy.corefun import Hartmann_mesh_show

if __name__ == "__main__":
    #ref_folder = "/home/lingfei/spexwavepy/data/Hartmann/CRLRefer/"
    #sam_folder = "/home/lingfei/spexwavepy/data/Hartmann/CRLSample/"
    ref_folder = "/YOUR/DATA/FOLDER/PATH/Hartmann/CRLRefer/"
    sam_folder = "/YOUR/DATA/FOLDER/PATH/Hartmann/CRLSample/"

    ROI_sam = [540, 1570, 750, 1800]
    ROI_ref = ROI_sam
    
    Imstack_sam = Imagestack(sam_folder, ROI_sam)
    Imstack_ref = Imagestack(ref_folder, ROI_ref)
    Imstack_sam.read_data()
    Imstack_ref.read_data()
    #print(Imstack_sam.data.shape) 
    #print(Imstack_ref.data.shape) 

    x_cens = np.arange(50, 1050, 50)
    y_cens = np.arange(60, 1000, 50)
    size = 15

    X_cens, Y_cens = np.meshgrid(x_cens, y_cens)
    #Hartmann_mesh_show(Imstack_ref.data[0], X_cens, Y_cens, size)
    #plt.show()

    Track_Hartmann = Tracking(Imstack_sam, Imstack_ref)
    pad = 20
    Track_Hartmann.Hartmann_XST(X_cens, Y_cens, pad, size)

    plt.figure()
    plt.imshow(Track_Hartmann.delayX, cmap='jet')
    plt.figure()
    plt.imshow(Track_Hartmann.delayY, cmap='jet')

    plt.show()
