from config.option import args
import dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
from srresnet import _NetG
import torch.nn as nn
import torch.optim as optim

#print(args.scale)
print(args)
##Load Data
zoomTrainDataset = dataset.ZoomDataset(args, isTrain=False)
zoomTrainDataloader = torch.utils.data.DataLoader(zoomTrainDataset,batch_size=args.batchSize,shuffle=True)

#print(zoomTrainDataset)
#print(len(zoomTrainDataloader))


zoomTestDataset = dataset.ZoomDataset(args, isTrain=False)
zoomTestDataloader = torch.utils.data.DataLoader(zoomTestDataset,batch_size=args.testBatchSize,shuffle=False)

#LR_raw,LR,HR,_ = zoomDataset[40]

## Define Model
model = _NetG(args)
model = model.cuda()
criterion = nn.MSELoss()

## define Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    return lr

def train(epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    for batch_idx, data in enumerate(zoomTrainDataloader):
        LR_raw,LR,HR,_ = data
        HR = HR.type(torch.float32)

        LR_raw, HR = LR_raw.cuda(), HR.cuda()
        #plot(LR,HR)
        #print(LR_raw.shape)
        output = model(LR_raw)
        #print(output.shape)
        #plot(HR,output)

        loss = criterion(output, HR)
        loss.backward()
        optimizer.step()

        print("[%d|%d] Loss:%.4f" %(batch_idx, len(zoomTrainDataloader), loss))
        #break;
    return HR, output

def test():
    pass

def plot(LR, HR):
    lr = LR.detach().cpu().numpy()
    hr = HR.detach().cpu().numpy()

    lr = np.transpose(lr[0], (1,2,0))
    hr = np.transpose(hr[0], (1,2,0))

    lr = np.clip(lr, 0, 1)
    hr = np.clip(hr, 0 ,1)

    plt.subplot(221)
    plt.imshow(lr)

    plt.subplot(222)
    plt.imshow(hr)

    plt.show()

if __name__=="__main__":
    HR, output = train(1)
    plot(HR,output)



# #print(model)
# aligned_image = Image.fromarray(np.uint8(utils.clipped(LR)))
# aligned_image = aligned_image.resize((HR.shape[1], HR.shape[0]), Image.ANTIALIAS)
# LR = np.array(aligned_image)
#
# LR = utils.image_float(LR)
# HR = utils.image_float(HR)
# #sum_img_t, _ = utils_align.sum_aligned_image(images,images)
#
# min_img_t = np.abs(HR - LR)
# min_img_t_scale = (min_img_t - np.min(min_img_t)) / (np.max(min_img_t) - np.min(min_img_t))
# #print(min_img_t)
# #print(min_img_t_scale)
# #cv2.imwrite('aligned.jpg', np.uint8(sum_img_t * 255))
# #sum_img_t = np.uint8(255.*utils.clipped(sum_img_t))
# #
# plt.subplot(221)
# plt.imshow(LR)
#
# plt.subplot(222)
# plt.imshow(HR)
#
# plt.subplot(223)
# plt.imshow(output)
#
# plt.subplot(224)
# plt.imshow(min_img_t_scale)
#
#
# plt.show()
