import os
import sys

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('../..'))
sys.path.append(os.path.join('../../..'))
from spexwavepy.imstackfun import Imagestack
from spexwavepy.corefun import read_one, crop_one

showImage = True
if __name__ == "__main__":
    fileFolder = "/YOUR/DATA/FOLDER/PATH/pixelsizestep10um/402724-pcoedge-files/"
    #fileFolder = "/home/lingfei/spexwavepy/data/pixelsizestep10um/402724-pcoedge-files/"
    ROI = [0, 3500, 0, 4500]           #[y_start, y_end, x_start, x_end]
    Imstack_1 = Imagestack(fileFolder, ROI)
    Imstack_1.read_data()

    filepath = "/home/lingfei/spexwavepy/data/pixelsizestep10um/402724-pcoedge-files/00005.tif"
    im_raw = read_one(filepath, ShowImage=True)
    ROI = [750, 1500, 500, 2000]    #[y_start, y_end, x_start, x_end]
    im_crop = crop_one(im_raw, ROI, ShowImage=True)
    
    subROI = [1500, 2000, 500, 2000]      #[y_start, y_end, x_start, x_end]
    dim = 'x'
    step = 10.0                           #[um]
    pixsize = Imstack_1.getpixsize(subROI, dim, step, display=showImage)
    print(r"Pixel size is {:.4f} um".format(pixsize))
