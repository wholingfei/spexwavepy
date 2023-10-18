"""
Align KB mirrors using Self-reference XST technique.
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

_PLOT = True #False 
if __name__ == "__main__":
    #folder_prefix_HKB = "/dls/science/groups/b16/SpeckleData/KBPitch/HKB/"
    #folder_prefix_VKB = "/dls/science/groups/b16/SpeckleData/KBPitch/HKB/"
    folder_prefix_HKB = "/home/lingfei/spexwavepy/tmp/HKB/"
    folder_prefix_VKB = "/home/lingfei/spexwavepy/tmp/VKB/"
    
    ###### HKB self-reference XST
    ROI_HKB = [45, 545, 60, 330]

    delayHKB_stack = np.zeros((13, 466))
    curvYHKB_stack = np.zeros((13, 466))

    for jc in range(1, 14, 1):
        imstack_tmp_1 = Imagestack(folder_prefix_HKB+'theta' + str(jc) + '/', ROI_HKB)
        imstack_tmp_1.fstart = 0
        imstack_tmp_1.fnum = 1

        imstack_tmp_2 = Imagestack(folder_prefix_HKB+'theta' + str(jc) + '/', ROI_HKB)
        imstack_tmp_2.fstart = 1
        imstack_tmp_2.fnum = 1

        track_tmp = Tracking(imstack_tmp_1, imstack_tmp_2)
        track_tmp.dimension = '1D'
        track_tmp.scandim = 'y'
        track_tmp.dist = 1650.0   # [mm]
        track_tmp.scanstep = 4.0   # [um]
        track_tmp.pixsize = 6.45   # [um]

        edge_x = 10
        edge_y = [5, 20]
        pad_x = 10
        pad_y = [5, 20]
        hw_xy = 10

        track_tmp.XST_self_dev(edge_x, edge_y, pad_x, pad_y, hw_xy, display=False, normalize=True)

        delayHKB_stack[jc-1] = track_tmp.delayY
        curvYHKB_stack[jc-1] = track_tmp.curvY

    ##### VKB self-reference XST
    ROI_VKB = [50, 540, 30, 350]

    delayVKB_stack = np.zeros((13, 286))
    curvYVKB_stack = np.zeros((13, 286))

    for jc in range(1, 11, 1):
        imstack_tmp_1 = Imagestack(folder_prefix_VKB+'theta' + str(jc) + '/', ROI_VKB)
        imstack_tmp_1.fstart = 0
        imstack_tmp_1.num = 1

        imstack_tmp_2 = Imagestack(folder_prefix_VKB+'theta' + str(jc) + '/', ROI_VKB)
        imstack_tmp_2.fstart = 1
        imstack_tmp_2.num = 1

        track_tmp = Tracking(imstack_tmp_1, imstack_tmp_2)
        track_tmp.dimension = '1D'
        track_tmp.scandim = 'x'
        track_tmp.dist = 1650.0   # [mm]
        track_tmp.scanstep = 4.0   # [um]
        track_tmp.pixsize = 6.45   # [um]

        edge_x = [20, 5]
        edge_y = 10 
        pad_x = [20, 5]
        pad_y = 10
        hw_xy = 10

        track_tmp.XST_self_dev(edge_x, edge_y, pad_x, pad_y, hw_xy, display=False, normalize=True)

        delayVKB_stack[jc-1] = track_tmp.delayX
        curvYVKB_stack[jc-1] = track_tmp.curvX

    if _PLOT:
        plt.figure(figsize=(10, 3))
        angles = [0, 3, 6, 9, 12]
        for ic, ang in enumerate(angles):
            plt.plot(curvYHKB_stack[ic], label='Angle '+str(ang+1))
        plt.legend()
        plt.xlabel('CCD pixel')
        plt.ylabel('Curvature ['+r'$m^{-1}$'+']')
        plt.tight_layout()

        plt.figure(figsize=(10, 3))
        angles = [0, 3, 6, 9, 12]
        for ic, ang in enumerate(angles):
            plt.plot(curvYHKB_stack[ic, 120:], label='Angle '+str(ang+1))
        plt.legend()
        plt.xlabel('CCD pixel')
        plt.ylabel('Curvature ['+r'$m^{-1}$'+']')
        plt.tight_layout()

        plt.figure(figsize=(10, 3))
        angles = [0, 2, 4, 6, 8]
        for ic, ang in enumerate(angles):
            plt.plot(curvYVKB_stack[ic], label='Angle '+str(ang+1))
        plt.legend()
        plt.xlabel('CCD pixel')
        plt.ylabel('Curvature ['+r'$m^{-1}$'+']')
        plt.tight_layout()

        plt.show()

    ###### Fitting
    Slos_HKB = np.zeros(13)
    p0s_HKB = np.zeros(13)
    for ic in range(13):
        R_tmp = curvYHKB_stack[ic, 120:]
        x_fit = np.arange(0, len(R_tmp), 1)
        param_tmp = np.polyfit(x_fit, R_tmp, 1)
        Slos_HKB[ic] = param_tmp[0]
        p0s_HKB[ic] = param_tmp[1]

    Slos_VKB = np.zeros(10)
    p0s_VKB = np.zeros(10)
    for ic in range(10):
        R_tmp = curvYVKB_stack[ic, 20:]
        x_fit = np.arange(0, len(R_tmp), 1)
        param_tmp = np.polyfit(x_fit, R_tmp, 1)
        Slos_VKB[ic] = param_tmp[0]
        p0s_VKB[ic] = param_tmp[1]

    ###### Plot the linear relation
    HKB_xs = np.arange(1, len(Slos_HKB)+1, 1)
    VKB_xs = np.arange(1, len(Slos_VKB)+1, 1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(HKB_xs, Slos_HKB*1.e6, '.')
    ax1.tick_params(axis='both', labelcolor='tab:blue', which='major')
    ax1.set_xlabel('Angle No.', color='tab:blue')
    ax1.set_ylabel('Fittied slope ['+r'$\times 10^{-6}$'+']', color='tab:blue')
    func_HKB = np.polyfit(HKB_xs, Slos_HKB*1.e6, 1)
    ax1.plot(HKB_xs, func_HKB[0]*HKB_xs+func_HKB[1], '--', color='tab:blue')
    ax2 = fig1.add_subplot(111, frame_on=False)
    ax2.plot(VKB_xs, Slos_VKB*1.e6, '.', color='tab:orange')
    func_VKB = np.polyfit(VKB_xs, Slos_VKB*1.e6, 1)
    ax2.plot(VKB_xs, func_VKB[0]*VKB_xs+func_VKB[1], '--', color='tab:orange')
    ax2.set_xlabel('Angle No.', color='tab:orange')
    ax2.set_ylabel('Fittied slope ['+r'$\times 10^{-6}$'+']', color='tab:orange')
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.invert_yaxis()
    ax2.tick_params(axis='both', labelcolor='tab:orange', which='major') 
    plt.tight_layout()

    plt.show()
