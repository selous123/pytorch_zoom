
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/store/dataset/SRRAW/X4/train", help="root folder that contains the images")
parser.add_argument("--labels", type=str, default="Edge+Diff", help="root folder that contains the images")
ARGS = parser.parse_args()

#traindataRootDir = "/store2/dataset/SR/train_data/SRRAW/X4/train"
traindataRootDir = ARGS.data_dir
subDirs = ["HR", "LR"]
## Generate Edge Information

import os
from PIL import Image, ImageFilter, ImageChops

file_names = os.listdir(os.path.join(traindataRootDir,subDirs[0]))
file_names.sort()

dir_label = ARGS.labels.split("+")

for d_label in dir_label:
    os.makedirs(os.path.join(traindataRootDir, d_label),exist_ok=True)


for i, file_name in enumerate(file_names):
    imgPath = os.path.join(traindataRootDir, subDirs[0], file_name)

    desPath = os.path.join(traindataRootDir,"Edge",file_name)

    img = Image.open(imgPath)
    img_edge = img.filter(ImageFilter.FIND_EDGES).filter(
                 ImageFilter.EDGE_ENHANCE_MORE)
    #.filter(ImageFilter.DETAIL)
    img_edge.save(desPath)


    print("file_name:%s, Index:%d" %(file_name,i))


for i, file_name in enumerate(file_names):
    imgPath = os.path.join(traindataRootDir, subDirs[0], file_name)
    imgLRPath = os.path.join(traindataRootDir, subDirs[1], file_name)

    desPath = os.path.join(traindataRootDir,"Diff",file_name)

    img = Image.open(imgPath)
    img_lr = Image.open(imgLRPath)

    u_img_lr = img_lr.resize(img.size)

    d = ImageChops.difference(img,u_img_lr).filter(
            ImageFilter.EDGE_ENHANCE_MORE).filter(ImageFilter.DETAIL)
    d.save(desPath)

    print("file_name:%s, Index:%d" %(file_name,i))
