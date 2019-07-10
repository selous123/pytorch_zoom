
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/store2/dataset/SR/zoom/train", help="root folder that contains the images")
ARGS = parser.parse_args()

rootPath = ARGS.data_dir
import os

## delete file in 00350
dirPath = os.path.join(rootPath, '00350')

delFiles = ["00005.JPG","00005.ARW","00006.ARW","00006.JPG","00007.ARW","00007.JPG"]
os.listdir(dirPath)

for delFile in delFiles:
    os.remove(os.path.join(dirPath, delFile))

## rename file in 00561
dirPath = os.path.join(rootPath, '00561')

for filename in sorted(os.listdir(dirPath), reverse=True):
    srPath = os.path.join(dirPath, filename)
    d_name = str(int(filename.split(".")[0])+1).zfill(5)+'.'+filename.split(".")[1]

    desPath = os.path.join(dirPath, d_name)
    print(desPath)
    os.rename(srPath, desPath)


# move 00006 file from 00560=>00561
srDirPath = os.path.join(rootPath, '00560')

srFilename = ["00006.ARW", "00006.JPG"]
desFilename = ["00001.ARW", "00001.JPG"]
import shutil
for filename,d_filename in zip(srFilename, desFilename):
    srPath = os.path.join(srDirPath, filename)
    desPath = os.path.join(dirPath, d_filename)
    shutil.move(srPath, desPath)
