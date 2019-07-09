#!/bin/bash

src_base=$1 # [YOUR TRAINING DATA PATH]
sfolder=1
efolder=585


for (( i=sfolder; i<=efolder; i++))
do
    dest_dir=$(printf "$src_base%0.5d/" $i)
    echo $dest_dir
    num=$(find $dest_dir -maxdepth 1 -name '*.JPG' | wc -l)
    echo $num $dest_dir
    if [ $num == 0 ]
    then
      continue
    fi
    rm -rf $dest_dir/cropped
    rm -rf $dest_dir/aligned
    rm -rf $dest_dir/compare

    python3 ./main_crop.py --path $dest_dir --num $num
    python3 ./main_align_camera.py --path $dest_dir --model ECC --rsz 3
done
