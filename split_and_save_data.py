import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import utils
import utils_align

class ZoomDataset(data.Dataset):
    def __init__(self, isTrain, dir_data="/store/dataset/zoom", scale=4, transform=None):
        self.isTrain = isTrain
        self.up_ratio = int(scale)
        self.transform = transform

        if isTrain is True:
            self.dir_path = os.path.join(dir_data, "train")
        else:
            self.dir_path = os.path.join(dir_data, "test")

        dir_names = os.listdir(self.dir_path)

        #print(dir_names)
        dir_names.sort()
        self.file_names = []

        if self.up_ratio == 4:
            for dir_name in dir_names:
                d_path = os.path.join(self.dir_path, dir_name)
                for i in range(1,4):
                    self.file_name = []
                    lr_raw_path = os.path.join(d_path, "0000"+str(i+4)+".ARW")

                    if not os.path.exists(lr_raw_path):
                        continue
                    hr_path = os.path.join(d_path, "0000"+str(i)+'.JPG')
                    self.file_name.append(lr_raw_path)
                    self.file_name.append(hr_path)
                    self.file_name.append(d_path)

                    self.file_name.append(i+4)
                    self.file_name.append(i)

                    self.file_names.append(self.file_name)
        elif self.up_ratio == 8:
            for dir_name in dir_names:

                d_path = os.path.join(self.dir_path, dir_name)

                self.file_name = []
                lr_raw_path = os.path.join(d_path, "0000"+str(6)+".ARW")
                if not os.path.exists(lr_raw_path):
                    continue
                hr_path = os.path.join(d_path, "0000"+str(1)+'.JPG')
                self.file_name.append(lr_raw_path)
                self.file_name.append(hr_path)
                self.file_name.append(d_path)

                self.file_name.append(6)
                self.file_name.append(1)

                self.file_names.append(self.file_name)
        else:
            raise ValueError("arg.scale should be 4 or 8")
        # for i in range(len(self.file_names)):
        #     print(self.file_names[i][2])
        ## file_name : [lr_raw, HR, d_path, lr_id, hr_id]

    def __getitem__(self,i):

        file_name = self.file_names[i]
        # print(file_name)
        LRAW_path = file_name[0]
        #LRAW_path = "/store/dataset/zoom/test/00134/00005.ARW"
        #height = width = self.patch_size


        LR_path = LRAW_path.replace(".ARW",".JPG")
        HR_path = file_name[1]
        #HR_path = "/store/dataset/zoom/test/00134/00001.JPG"
        tform_txt = os.path.join(file_name[2],"tform.txt")
        #tform_txt = "/store/dataset/zoom/test/00134/tform.txt"

        white_lv, black_lv = utils.read_wb_lv("sony")
        #print(LRAW_path)
        input_bayer = utils.get_bayer(LRAW_path, black_lv, white_lv)
        #print(input_bayer.shape)
        #print(input_bayer.shape)

        LR_raw = utils.reshape_raw(input_bayer)
        LR_img =  np.array(Image.open(LR_path))


        height = int(LR_img.shape[0] * utils.readFocal_pil(LR_path) / 240.0 / 2 - 20)
        width = int(LR_img.shape[1] * utils.readFocal_pil(LR_path) / 240.0 / 2 - 20)
        #height = width = 128


        #with shape [self.patch_size, self.patch_size, 4]
        input_raw = utils.crop_center_wh(LR_raw, height, width)
        cropped_lr_hw = utils.crop_center_wh(LR_img, height*2, width*2)

        #ground truth
        #with shape [self.patch_size*2*self.up_ratio, self.patch_size*2*self.up_ratio, 3]

        #HR_path = file_name[1]
        ## crop and resize according 00001.JPG
        rgb_camera_hr = np.array(Image.open(HR_path))
        crop_ratio = 240.0 / utils.readFocal_pil(HR_path)
        cropped_input_rgb_hr = utils.crop_fov(rgb_camera_hr, 1./crop_ratio)
        #cropped_input_rgb_hr = utils.image_float(cropped_input_rgb_hr)
        input_camera_rgb_hr = Image.fromarray(np.uint8(utils.clipped(cropped_input_rgb_hr)))
        input_camera_rgb_naive = input_camera_rgb_hr.resize((int(input_camera_rgb_hr.width * crop_ratio),
                int(input_camera_rgb_hr.height * crop_ratio)), Image.ANTIALIAS)
        #input_camera_rgb_naive.save("align_arw_test/input_rgb_camera_HR.png")
        hr = np.array(input_camera_rgb_naive)

        ## Align HR Image to LR Image and crop the corresponding patches
        ### Resize to corresponding up_ratio size
        zoom_ratio = 240.0 / utils.readFocal_pil(LR_path)
        aligned_hr_hw, _ = utils_align.imgAlign(hr, tform_txt, file_name[4], file_name[3], True, int(height*2*zoom_ratio), int(width*2*zoom_ratio))
        aligned_image = Image.fromarray(np.uint8(utils.clipped(aligned_hr_hw)))
        aligned_image = aligned_image.resize((int(width *2 * self.up_ratio),
                        int(height * 2 * self.up_ratio)), Image.ANTIALIAS)
        # aligned_image.save("align_arw_test/input_rgb_camera_alignedLHR.png")
        aligned_img_np = np.array(aligned_image)

        #print(aligned_img_np.shape)

        # [H,W,C] => [C,H,W]
        # input_raw = np.transpose(input_raw, (2,0,1)) / 255.0
        # cropped_lr_hw = np.transpose(cropped_lr_hw, (2,0,1)) / 255.0
        # aligned_img_np = np.transpose(aligned_img_np, (2,0,1)) / 255.0

        ##ToTensor
        return input_raw, cropped_lr_hw, aligned_img_np, LRAW_path

    def __len__(self):
        return len(self.file_names)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/store/dataset/", help="root folder that contains the images")
parser.add_argument("--source_dir", type=str, default="/store/dataset/zoom", help="root folder that contains the images")
parser.add_argument("--test", action='store_true', help="train or test data")
parser.add_argument("--scale", type=int, default=4, choices=[4,8], help="up ratio of data")
ARGS = parser.parse_args()

if __name__=="__main__":
    images = []
    zoomData = ZoomDataset(isTrain=not ARGS.test, dir_data=ARGS.source_dir, scale=ARGS.scale)

    print(len(zoomData))
    ## destination directory
    if ARGS.test:
        data_dir = os.path.join(ARGS.data_dir,'SRRAW','X'+str(ARGS.scale),'test')
    else:
        data_dir = os.path.join(ARGS.data_dir,'SRRAW','X'+str(ARGS.scale),'train')
    sub_dirs = ["LR","HR","ARW"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for sub_dir in sub_dirs:
        s_dir = os.path.join(data_dir,sub_dir)
        if not os.path.exists(s_dir):
            os.makedirs(s_dir)
    for i in range(0, len(zoomData)):
        print("Index:%d" %i)
        LR_raw,LR,HR,LRAW_path = zoomData[i]


        file_name = os.path.dirname(LRAW_path).split("/")[-1]+"_"+os.path.basename(LRAW_path).split(".")[0]+'.png'
        LR = Image.fromarray(np.uint8(utils.clipped(LR)))
        HR = Image.fromarray(np.uint8(utils.clipped(HR)))

        LR.save(os.path.join(data_dir,"LR",file_name))
        HR.save(os.path.join(data_dir,"HR",file_name))

        file_name = os.path.dirname(LRAW_path).split("/")[-1]+"_"+os.path.basename(LRAW_path).split(".")[0]+'.npy'

        np.save(os.path.join(data_dir,"ARW",file_name), LR_raw)
    #
    #
    LR = LR.resize((HR.size), Image.ANTIALIAS)
    LR = np.array(LR)
    HR = np.array(HR)
    LR = utils.image_float(LR)
    HR = utils.image_float(HR)
    min_img_t = np.abs(HR - LR)
    min_img_t_scale = (min_img_t - np.min(min_img_t)) / (np.max(min_img_t) - np.min(min_img_t))
    plt.subplot(221)
    plt.imshow(LR)

    plt.subplot(222)
    plt.imshow(HR)

    plt.subplot(223)
    plt.imshow(min_img_t)

    plt.subplot(224)
    plt.imshow(min_img_t_scale)


    plt.show()
