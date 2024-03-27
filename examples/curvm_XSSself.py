"""
Mirror slope error curve (1D) reconstructed from the dowmstream setup.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import copy

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('../..'))
sys.path.append(os.path.join('../../..'))
from spexwavepy.imstackfun import Imagestack
from spexwavepy.trackfun import Tracking 
from spexwavepy.corefun import read_one, crop_one
from spexwavepy.postfun import EllipseSlope

if __name__ == "__main__":
    ShowImage = True
    #folderName = "/home/lingfei/spexwavepy/data/curvmirrorDown/"
    folderName = "/YOUR/DATA/FOLDER/PATH/curvmirrorDown/"
    #im_sam = read_one(folderName + 'ipp_292770_1.TIF', ShowImage=ShowImage)
    ROI = [338, 643, 675, 825]          #[y_start, y_end, x_start, x_end]
    #crop_one(im_sam, ROI, ShowImage=ShowImage)
    #sys.exit(0)

    imstack = Imagestack(folderName, ROI)

    track_XSS = Tracking(imstack)
    track_XSS.dimension = '1D'
    track_XSS.scandim = 'y'
    track_XSS.mempos = 'downstream'
    track_XSS.dist = 1790.0    #[mm]
    track_XSS.pixsize = 6.45    #[um]
    track_XSS.scanstep = 0.25    #[um]

    edge_x = 15
    edge_y = 0
    edge_z = [5, 30] 
    nstep = 2

    track_XSS.XSS_self(edge_x, edge_y, edge_z, nstep, display=False)

    ######### Iterative algorithm for donwstream case
    iy = track_XSS.delayY
    loccurv_y = track_XSS.curvY
    theta = 3.7e-3                     #[rad], pitch angle
    mirror_L = 0.10                    #[m], mirror length
    dist_mc2det = 2.925                #[m]
    D = dist_mc2det + 0.5 * mirror_L * np.cos(theta)   #[m]
    pixsize = track_XSS.pixsize

    loccurvs = 0.5 * np.flip(loccurv_y)
    detPos = np.arange(0, len(loccurvs)) * pixsize * 1.e-6     #[m]           
    SloErr = scipy.integrate.cumtrapz(loccurvs, detPos)           #[rad]
    SloErr = np.concatenate((np.array([0.]), SloErr))                #[rad]
    #Inc_corr = np.linspace(-0.5*0.08*theta/41., 0.5*0.08*theta/41, len(SloErr))
    #SloErr -= Inc_corr
    x_init = np.linspace(0, mirror_L, len(SloErr))                #[m]
    y_init = scipy.integrate.cumtrapz(SloErr*0.+theta, x_init)             #[m]
    y_init = np.concatenate((np.array([0.]), y_init))          #[m]
    Y_det = y_init + 2 * (SloErr+theta) * (D-x_init)
    Y_det = Y_det[0] + detPos
    y_init2 = Y_det - 2 * (SloErr+theta) * (D-x_init)
    x = copy.deepcopy(x_init)
    y = copy.deepcopy(y_init)
    for i in range(50):
        y_prev = copy.deepcopy(y)
        x_prev = copy.deepcopy(x)
        x = D - (Y_det - y) / (2 * (SloErr + theta))                   #[m]
        #sys.exit(0)
        y = scipy.integrate.cumtrapz(SloErr+theta, x)                  #[m]
        y = np.concatenate((np.array([0.]), y))                        #[m]
        y_after = copy.deepcopy(y)
        x_after = copy.deepcopy(x)
        if i>0: 
            #plt.plot(x*1.e3, s*1.e6)
            print("Iteration time: " + str(i+1))
            print(np.sqrt(np.sum((y_prev-y_after)**2)))
            print(np.sqrt(np.sum((x_prev-x_after)**2)))
    #########

    ######### Fitting
    p = 46.      #[m]
    q = 0.4      #[m]
    theta = 3.e-3     #[rad]
    popt, pcov = scipy.optimize.curve_fit(EllipseSlope, x, SloErr, bounds=([p-1, 0., theta-0.3e-3], [p+1, 1., theta+0.3e-3]))
    SloFit = EllipseSlope(x, popt[0], popt[1], popt[2])
    SloRes = SloErr - SloFit

    #print(popt)
    #########

    ######### Exel data reading
    import pandas

    #exel_folder = "/home/lingfei/spexwavepy/data/NOM_data.xlsx"
    exel_folder = "/YOUR/DATA/FOLDER/PATH/NOM_data.xlsx"
    data_Fram = pandas.read_excel(exel_folder)
    data_array = np.array(data_Fram)
    x_lane1 = data_array[2:901, 1]
    slo_lane1 = data_array[2:901, 2]
    sloErr_lane1 = data_array[2:901, 3]
    x_lane2 = data_array[2:901, 5]
    slo_lane2 = data_array[2:901, 6]
    sloErr_lane2 = data_array[2:901, 7]
    x_lane3 = data_array[2:901, 9]
    slo_lane3 = data_array[2:901, 10]
    sloErr_lane3 = data_array[2:901, 11]

    plt.figure()
    plt.plot(x*1.e3-41, np.flip(-SloRes)*1.e6, label='At-wavelength measurement')
    plt.plot(x_lane3, sloErr_lane3, label='Off-line measurement')
    plt.xlabel('Mirror length [mm]')
    plt.ylabel('Slope error [' + r'$\mu$' + 'rad]')
    plt.legend()
    ######### 

    if ShowImage:
        plt.figure()
        plt.plot(track_XSS.curvY)
        plt.xlabel('Pixels')
        plt.ylabel('Local curvature [1/m]')

        plt.figure()
        plt.plot(x_lane1, sloErr_lane1)
        plt.plot(x_lane2, sloErr_lane2)
        plt.plot(x_lane3, sloErr_lane3)


    plt.show()
