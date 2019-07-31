from data import srraw_test
import sys

sys.path.append('..')
import utils
import scipy
import numpy as np
import PIL.Image as Image
from option import args
# testDataset = srraw_test.SRRAW_TEST(args, 'SRRAW')
from PIL import Image
from data import srraw
testDataset = srraw.SRRAW(args, 'SRRAW', train=False)
print(len(testDataset))
psnrs = []
for data in testDataset:
    lr = data[0].numpy()
    hr = data[1][0].numpy()
    print(data[2])
    lr = np.transpose(lr, (1,2,0))
    hr = np.transpose(hr, (1,2,0))

    lr = np.array(Image.fromarray(lr.astype(np.uint8)).resize(hr.shape[:2], Image.BICUBIC))
    #lr = scipy.misc.imresize(lr, hr.shape, interp='bicubic')

    psnr = utils.calc_psnr(lr, hr, scale=0, rgb_range=255.)
    psnrs.append(psnr)
    #print(psnr)

print(np.mean(np.array(psnrs)))
print(len(testDataset))
