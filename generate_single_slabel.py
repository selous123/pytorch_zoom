

HR_path = "/store2/dataset/SR/train_data/SRRAW/X4/train/HR/00006_00005.png"
LR_path = "/store2/dataset/SR/train_data/SRRAW/X4/train/LR/00006_00005.png"

from PIL import Image
from PIL import ImageFilter,ImageChops



hr_img = Image.open(HR_path)
lr_img = Image.open(LR_path)
u_img_lr = lr_img.resize(hr_img.size)

new_height = new_width = 1024
width, height = hr_img.size   # Get dimensions
left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2
# hr_img = hr_img.crop((left, top, right, bottom))
# lr_img = lr_img.crop((left, top, right, bottom))


d = ImageChops.difference(hr_img,u_img_lr)
d1 = d.crop((left, top, right, bottom))
d2 = d1.filter(ImageFilter.EDGE_ENHANCE_MORE)
d3 = d2.filter(ImageFilter.DETAIL)

import matplotlib.pyplot as plt
# d1.save("difference.png")
# d2.save("d_e.png")
# d3.save("d_e_detail.png")

# plt.figure(figsize=(16,8))
plt.axis("off")
ax = plt.subplot(131)
plt.imshow(d1)
plt.axis("off")
title= '(a)'
ax.set_title(title, y=-0.2,fontweight='bold')

ax = plt.subplot(132)
plt.imshow(d2)
plt.axis("off")
title="(b)"
ax.set_title(title, y=-0.20,fontweight='bold')

ax = plt.subplot(133)
plt.imshow(d3)
plt.axis("off")
title = "(c)"
ax.set_title(title, y=-0.20,fontweight='bold')

plt.savefig("slabel.png", dpi=512, bbox_inches='tight', pad_inches = 0)
plt.savefig("slabel.pdf", dpi=512, bbox_inches='tight', pad_inches = 0)
#plt.show()
