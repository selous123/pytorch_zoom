import tensorflow as tf
import yaml
import utils
import net
import os
import numpy as np
from PIL import Image
import utils_align
import cv2
import dataset
import torch
import time

gpu_options = tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)

config_file_path = "config/test.yaml"
with open(config_file_path, "r") as f:
    config_file = yaml.load(f)

# Model parameters
mode = config_file["mode"]
device = config_file["device"]
up_ratio = config_file["model"]["up_ratio"]
num_in_ch = config_file["model"]["num_in_channel"]
num_out_ch = config_file["model"]["num_out_channel"]
file_type = config_file["model"]["file_type"]
upsample_type = config_file["model"]["upsample_type"]

# Input / Output parameters
inference_root = [config_file["io"]["inference_root"]]
task_folder = config_file["io"]["task_folder"]
restore_path = config_file["io"]["restore_ckpt"]

# Remove boundary pixels with artifacts
raw_tol = 0

sess = tf.Session(config=config)
# read in white and black level to normalize raw sensor data for different devices
white_lv, black_lv = utils.read_wb_lv(device)

# set up the model
with tf.variable_scope(tf.get_variable_scope()):
    input_raw=tf.placeholder(tf.float32,shape=[1,None,None,num_in_ch], name="input_raw")
    out_rgb, out_d = net.SRResnet(input_raw, num_out_ch, up_ratio=up_ratio, reuse=False, up_type=upsample_type)

    if raw_tol != 0:
        out_rgb = out_rgb[:,int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),
            int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),:]  # add a small offset to deal with boudary case

    objDict = {}
    objDict['out_rgb'] = out_rgb
    objDict['out_d'] = out_d
saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])

sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("%s"%(restore_path))
print("Contain checkpoint: ", ckpt)
if not ckpt:
    print("No checkpoint found.")
    exit()
else:
    saver_restore.restore(sess,ckpt.model_checkpoint_path)




zoomDataset = dataset.ZoomDataset(isTrain=False)
zoomDataloader = torch.utils.data.DataLoader(zoomDataset,batch_size=1,shuffle=False)

exit(0)

file = r'log.txt'
#print(len(zoomDataset))
#exit(0)
ssims = []
psnrs = []
t = time.time()
for data in zoomDataloader:

    input_raw_img, lr, hr, inference_path = data
    #print(input_raw_img.shape)
    #input_raw_img = np.expand_dims(input_raw_img, 0)

    lr = lr[0].numpy()
    hr = hr[0].numpy()
    inference_path = inference_path[0]

    aligned_image = Image.fromarray(np.uint8(utils.clipped(lr)))
    aligned_image = aligned_image.resize((hr.shape[1], hr.shape[0]), Image.ANTIALIAS)
    lr = np.array(aligned_image)

    #out_objDict=sess.run(objDict,feed_dict={input_raw:input_raw_img})


    wb_txt = os.path.join(os.path.dirname(inference_path), 'wb.txt')
    out_wb = utils.read_wb(wb_txt, key=os.path.basename(inference_path).split('.')[0]+":")
    #out_wb = utils.compute_wb(inference_path)

    #wb_rgb = out_objDict["out_rgb"][0,...]
    # print(wb_rgb.shape)
    # wb_rgb[...,0] *= np.power(out_wb[0,0],1/2.2)
    # wb_rgb[...,1] *= np.power(out_wb[0,1],1/2.2)
    # wb_rgb[...,2] *= np.power(out_wb[0,3],1/2.2)

    # aligned_image = Image.fromarray(np.uint8(utils.clipped(wb_rgb)*255))
    # aligned_image = aligned_image.resize((hr.shape[1], hr.shape[0]), Image.ANTIALIAS)
    # wb_rgb = np.array(aligned_image)

    hr = utils.image_float(hr)
    lr = utils.image_float(lr)
    #wb_rgb = utils.image_float(wb_rgb)

    s2_time = time.time()
    psnr = utils.calc_psnr(lr, hr, scale=4, rgb_range=1.)
    #psnr_sr = utils.calc_psnr(wb_rgb, hr, scale=4, rgb_range=1.)
    ssim = utils.calc_ssim(lr,hr)
    #ssim_sr = utils.calc_ssim(wb_rgb,hr)
    e_time = time.time()

    psnrs.append(psnr)
    ssims.append(ssim)

    log_line = "[PSNR] [SSIM] [calc Time] [Loop Time] for lr and hr:%.4f %.4f %.4f %.4f" \
        %(psnr, ssim, e_time-s2_time, time.time() - t)
    print(log_line)

    #write log file
    with open(file, 'a+') as f:
         f.write(log_line+'\n')

    t = time.time()

print("PSNR for lr and hr:", np.mean(psnrs))
#print("PSNR for sr and hr:", psnr_sr)
print("SSIM for lr and hr:", np.mean(ssims))
#print("SSIM for sr and hr:", ssim_sr)

with open(file, 'a+') as f:
     f.write("mean PSNR and SSIM are: %.4f, %.4f" %(np.mean(psnrs),np.mean(ssims)) + '\n')

print(hr.shape)

vis = True
## Visualization Results.
if vis:
    min_img_t = np.abs(hr - lr)
    min_img_t_scale = (min_img_t - np.min(min_img_t)) / (np.max(min_img_t) - np.min(min_img_t))

    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.imshow(lr)

    plt.subplot(222)
    plt.imshow(hr)

    plt.subplot(223)
    plt.imshow(min_img_t_scale)

    # plt.subplot(224)
    # plt.imshow(wb_rgb)

    plt.show()
