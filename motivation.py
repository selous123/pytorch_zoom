import os
data_root = '/store/dataset/SR/train_data/SRRAW/X4/train/'
file_name = '00006_00005.png'

hr_path = os.path.join(data_root, 'HR', file_name)
lr_path = os.path.join(data_root, 'LR', file_name)

from PIL import Image

hr_img = Image.open(hr_path)
lr_img = Image.open(lr_path)
lr_img = lr_img.resize(hr_img.size)
new_height = new_width = 1024
width, height = hr_img.size   # Get dimensions
left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2
hr_img = hr_img.crop((left, top, right, bottom))
lr_img = lr_img.crop((left, top, right, bottom))
import numpy as np
from PIL import ImageChops

res_img = ImageChops.difference(hr_img, lr_img)

# hr_img.save("hr.png")
# lr_img.save("lr.png")
from matplotlib import pyplot as plt
# ax = plt.subplot(131)
# plt.imshow(hr_img)
# plt.axis("off")
# title='(a)'
# ax.set_title(title, y=-0.2)
# #plt.axis('off')
#
# ax = plt.subplot(132)
# plt.imshow(lr_img)
# plt.axis("off")
# title='(b)'
# ax.set_title(title, y=-0.2)
# #plt.axis('off')
#
# ax = plt.subplot(133)
# plt.imshow(res_img)
# plt.axis("off")
# title='(c)'
# ax.set_title(title, y=-0.2)
#
# plt.savefig("res.png", dpi=512, bbox_inches='tight', pad_inches = 0)


# import numpy as np
# res_img1 = res_img.convert("L")
# res_img1 = np.array(res_img1) / 255.0
# print(res_img1)
# ax = plt.gca()
# plt.imshow(res_img1, cmap='jet')
# plt.axis("off")
# plt.colorbar()
# #title='(e)'
# #ax.set_title(title, y=-0.08)
# plt.savefig("res_img.png", dpi=512, bbox_inches='tight', pad_inches = 0)

#print(np.array(res_img))
#plt.show()
a = np.array(res_img) / np.array(hr_img)
a2 = np.array(hr_img) / np.array(res_img)
a[a==np.nan] = 0
a2[a2==np.nan] = 0

num1 = []
num2 = []
for i in range(10):
    num1.append(((a>=(i/10)) & (a<((i+1)/10))).sum())
    num2.append(((a2>=(i/10)) & (a2<((i+1)/10))).sum())

num = [num1[i] + num2[i] for i in range(10)]

print(num)
num = np.array(num)
print(num.sum(), a.size)
num = num / a.size
plt.bar(range(10), num)
plt.savefig("statistics.png", dpi=512, bbox_inches='tight', pad_inches = 0)
