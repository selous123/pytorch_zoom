import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--e", type=int, default=500)
parser.add_argument("--root_path", type=str, default="/home/ibrain/git/pytorch_zoom/experiment")
parser.add_argument("--dir_path", type=str)
ARGS = parser.parse_args()



apath = os.path.join(ARGS.root_path, 'edsr_paper_x4')
loss_log = torch.load(os.path.join(apath, 'loss_log.pt'))
psnr_log = torch.load(os.path.join(apath, 'psnr_log.pt'))
#print(psnr_log)

apath = os.path.join(ARGS.root_path, 'edsr_baseline_x4_dynamic_lr_epoch500')
loss_b_log = torch.load(os.path.join(apath, 'loss_log.pt'))
psnr_b_log = torch.load(os.path.join(apath, 'psnr_log.pt'))


apath = os.path.join(ARGS.root_path, 'ssl_addfusionreverse_x4')
loss_s_log = torch.load(os.path.join(apath, 'loss_SR_log.pt'))
psnr_s_log = torch.load(os.path.join(apath, 'psnr_log.pt'))

#apath = os.path.join(ARGS.root_path, 'ssl_addfusion_reverse_rlossv0.2_x4')
apath = os.path.join(ARGS.root_path, ARGS.dir_path)
loss_r_log = torch.load(os.path.join(apath, 'loss_SR_log.pt'))
psnr_r_log = torch.load(os.path.join(apath, 'psnr_log.pt'))


e = min(len(loss_s_log), len(loss_r_log), len(loss_b_log), len(loss_log), ARGS.e)
print("Epoch:",e)
loss_log = loss_log[0:e]
psnr_log = psnr_log[0:e]
loss_b_log = loss_b_log[0:e]
psnr_b_log = psnr_b_log[0:e]
loss_r_log = loss_r_log[0:e]
psnr_r_log = psnr_r_log[0:e]
loss_s_log = loss_s_log[0:e]
psnr_s_log = psnr_s_log[0:e]


axis = np.linspace(1, e, e)
fig = plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(axis, loss_log.numpy(),color = 'red' ,label= "EDSR_paper")
plt.plot(axis, loss_b_log.numpy(),color = 'blue' ,label= "EDSR_baseline_paper")
plt.plot(axis, loss_r_log.numpy(),color = 'green' ,label= "SSL_r_loss")
plt.plot(axis, loss_s_log.numpy(),color = 'black' ,label= "SSL")
plt.title("loss")
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.plot(axis, psnr_log.numpy().squeeze(),color = 'red' ,label= "EDSR_paper")
plt.plot(axis, psnr_b_log.numpy().squeeze(),color = 'blue' ,label= "EDSR_baseline_paper")
plt.plot(axis, psnr_r_log.numpy().squeeze(),color = 'green' ,label= "SSL_r_loss")
plt.plot(axis, psnr_s_log.numpy().squeeze(),color = 'black' ,label= "SSL")
plt.title("psnr")
plt.legend()
plt.grid(True)

plt.savefig(apath+"/result.pdf")

#plt.show()
