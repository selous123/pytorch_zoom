from data import srraw_test
import sys

sys.path.append('..')
import utils
import scipy
import numpy as np
import PIL.Image as Image
from option import args
#testDataset = srraw_test.SRRAW_TEST(args, 'SRRAW')
from PIL import Image
from data import srraw
from matplotlib import pyplot as plt
testDataset = srraw.SRRAW(args, 'SRRAW', train=False)
print(len(testDataset))
psnrs = []
for data in testDataset:
    lr = data[0].numpy()
    print(lr.shape)
    hr = data[1][0].numpy()
    print(hr.shape)





    print(data[2])
    lr = np.transpose(lr, (1,2,0))
    hr = np.transpose(hr, (1,2,0))

    lr = np.array(Image.fromarray(lr.astype(np.uint8)).resize(hr.shape[:2], Image.BICUBIC))
    #lr = scipy.misc.imresize(lr, hr.shape, interp='bicubic')



    psnr = utils.calc_psnr(lr, hr, scale=0, rgb_range=255.)
    psnrs.append(psnr)
    print(psnr)
    #print(psnr)

    # plt.subplot(121)
    # plt.imshow(lr.astype(np.uint8))
    #
    # plt.subplot(122)
    # plt.imshow(hr.astype(np.uint8))
    #
    # plt.show()

print(np.mean(np.array(psnrs)))
print(len(testDataset))
