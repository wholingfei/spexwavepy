import os
import sys

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('../..'))
sys.path.append(os.path.join('../../..'))
from spexwavepy.imstackfun import Imagestack

showImage = True
if __name__ == "__main__":
    fileFolder = "/dls/b16/data/2023/cm33912-1/pixelsizestep10um/402724-pcoedge-files/"
    ROI = [0, 3500, 0, 4500]           #[y_start, y_end, x_start, x_end]
    Imstack_1 = Imagestack(fileFolder, ROI)
    Imstack_1.read_data()
    subROI = [1500, 2000, 500, 2000]      #[y_start, y_end, x_start, x_end]
    dim = 'x'
    step = 10.0                           #[um]
    pixsize = Imstack_1.getpixsize(subROI, dim, step, display=showImage)
    print(r"Pixel size is {:.4f} um".format(pixsize))
