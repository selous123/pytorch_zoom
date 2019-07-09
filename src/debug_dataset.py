import data
from option import args
loader = data.Data(args)

print(len(loader.loader_test))
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

import utils

def plot(LR, HR):
    lr = LR.detach().cpu().numpy()
    hr = HR.detach().cpu().numpy()

    lr = np.transpose(lr[0], (1,2,0))
    hr = np.transpose(hr[0], (1,2,0))

    #lr = utils.image_float(lr)
    #hr = utils.image_float(hr)
    lr = lr / 255.0
    hr = hr / 255.0
    lr = np.clip(lr, 0, 1)
    hr = np.clip(hr, 0 ,1)

    plt.subplot(221)
    plt.imshow(lr)

    plt.subplot(222)
    plt.imshow(hr)

    plt.show()

import torch
import torchvision
for data in loader.loader_test[0]:
    lr,labels,filename,idx = data
    hr = labels[0]
    print("lr shape: ", lr.shape)
    print("HR shape: ", hr.shape)

    # print(data[2])
    # print(data[3])

    plot(lr,hr)

    print(filename)
    print(idx)

    break;

    # torchvision.utils.save_image(data[1] / 255.0,"lr.png")
    # # print(data[0])
    # #
    # lr = data[0][0].numpy() / 255.0
    # lr = np.transpose(lr, (1,2,0))
    # plt.imshow(lr)
    # plt.show()
    # break;
