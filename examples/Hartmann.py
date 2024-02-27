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

if __name__ == "__main__":
    ref_folder = "../tmp/CRLRefer/"
    sam_folder = "../tmp/CRLSample/"

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
