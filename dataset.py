import torch.utils.data as data
from config.option import args
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import utils
import utils_align

class ZoomDataset(data.Dataset):
    def __init__(self, args, isTrain, transform=None):
        self.isTrain = isTrain
        self.up_ratio = int(args.scale)
        #self.patch_size = 128
        self.patch_size = args.patch_size

        self.transform = transform
        if self.isTrain:
            self.dir_path = os.path.join(args.dir_data, args.data_train)
        else:
            self.dir_path = os.path.join(args.dir_data, args.data_test)

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
                hr_path = os.path.join(d_path, "0000"+str(1)+'.JPG')
                self.file_name.append(lr_raw_path)
                self.file_name.append(hr_path)
                self.file_name.append(d_path)

                self.file_name.append(6)
                self.file_name.append(1)

                self.file_names.append(self.file_name)
            else:
                raise ValueError("arg.scale should be 4 or 8")

        ## file_name : [lr_raw, HR, d_path, lr_id, hr_id]

    def __getitem__(self,i):

        file_name = self.file_names[i]
        height = width = self.patch_size
        LRAW_path = file_name[0]
        #LRAW_path = "/store/dataset/zoom/test/00134/00005.ARW"

        LR_path = LRAW_path.replace(".ARW",".JPG")
        HR_path = file_name[1]
        #HR_path = "/store/dataset/zoom/test/00134/00001.JPG"
        tform_txt = os.path.join(file_name[2],"tform.txt")
        #tform_txt = "/store/dataset/zoom/test/00134/tform.txt"

        white_lv, black_lv = utils.read_wb_lv("sony")
        input_bayer = utils.get_bayer(LRAW_path, black_lv, white_lv)
        #print(input_bayer.shape)
        LR_raw = utils.reshape_raw(input_bayer)
        LR_img =  np.array(Image.open(LR_path))
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
        aligned_image = aligned_image.resize((int(height *2 * self.up_ratio),
                        int(width * 2 * self.up_ratio)), Image.ANTIALIAS)
        # aligned_image.save("align_arw_test/input_rgb_camera_alignedLHR.png")
        aligned_img_np = np.array(aligned_image)

        # [H,W,C] => [C,H,W]
        input_raw = np.transpose(input_raw, (2,0,1)) / 255.0
        cropped_lr_hw = np.transpose(cropped_lr_hw, (2,0,1)) / 255.0
        aligned_img_np = np.transpose(aligned_img_np, (2,0,1)) / 255.0

        ##ToTensor
        return input_raw, cropped_lr_hw, aligned_img_np, LRAW_path

    def __len__(self):
        return len(self.file_names)


if __name__=="__main__":
    images = []
    zoomData = ZoomDataset(args, isTrain=True)

    print(len(zoomData))

    LR_raw,LR,HR,_ = zoomData[3]


    LR = np.transpose(LR, (1,2,0))
    HR = np.transpose(HR, (1,2,0))

    aligned_image = Image.fromarray(np.uint8(utils.clipped(LR)*255))
    aligned_image = aligned_image.resize((HR.shape[1], HR.shape[0]), Image.ANTIALIAS)
    LR = np.array(aligned_image)

    LR = utils.image_float(LR)
    HR = utils.image_float(HR)

    images.append(LR)
    images.append(HR)

    #sum_img_t, _ = utils_align.sum_aligned_image(images,images)

    min_img_t = np.abs(HR - LR)
    min_img_t_scale = (min_img_t - np.min(min_img_t)) / (np.max(min_img_t) - np.min(min_img_t))
    #print(min_img_t)
    #print(min_img_t_scale)
    #cv2.imwrite('aligned.jpg', np.uint8(sum_img_t * 255))
    #sum_img_t = np.uint8(255.*utils.clipped(sum_img_t))
    #
    plt.subplot(221)
    plt.imshow(LR)

    plt.subplot(222)
    plt.imshow(HR)

    plt.subplot(223)
    plt.imshow(min_img_t)

    plt.subplot(224)
    plt.imshow(min_img_t_scale)


    plt.show()
