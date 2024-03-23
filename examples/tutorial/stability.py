import os
import sys
import numpy as np

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('../..'))
sys.path.append(os.path.join('../../..'))
from spexwavepy.imstackfun import Imagestack
from spexwavepy.trackfun import Tracking 

showImage = True
usemulti = True #Using multicores or not
if showImage: import matplotlib.pyplot as plt
if __name__ == "__main__":
    fileFolder = "/YOUR/DATA/FOLDER/PATH/stabilitycheck/"
    #fileFolder = "/home/lingfei/spexwavepy/data/stabilitycheck/"
    ROI = [0, 3500, 0, 4500]           #[y_start, y_end, x_start, x_end]
    Imstack_1 = Imagestack(fileFolder, ROI)
    Imstack_1.fnum = 99   #File number to be used for stability check
    Imstack_1.fstart = 0   #File start number to be used for stability check
    Imstack_1.dim = 'both'

    track = Tracking(Imstack_1)
    edge_x, edge_y = 10, 10
    if not usemulti:
        delayX, delayY, res = track.stability(edge_x, edge_y)
    else:
        cpu_no = 16
        delayX, delayY, res = track.stability_multi(edge_x, edge_y, cpu_no)

    if showImage:
        x_plot = np.arange(1, len(delayX)+1)
        plt.plot(x_plot, delayX, label='X shift')
        plt.plot(x_plot, delayY, label='Y shift')
        plt.xlabel('Image number')
        plt.ylabel('Shift in pixels [pixel]')
        plt.legend()
        plt.show()
