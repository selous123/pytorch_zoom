## in your local host seeing data_train

--dir_data /store2/dataset/SR/train_data
--wb_root /store/dataset/zoom/train/


## ibrain server
--dir_data /store/dataset/SR/train_data

## test
python main.py --model EDSR --scale 4  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train /home/ibrain/git/EDSR-PyTorch/experiment/model/EDSR_x4.pt --test_only --save_results
python main.py --model EDSR --scale 4  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train /home/ibrain/git/EDSR-PyTorch/experiment/model/EDSR_x4.pt --test_only --save_results

python main.py --model EDSR --scale 4 --pre_train ../models/edsr_baseline_x4-6b446fab.pt --test_only --save_results


## train baseline
python main.py --model EDSR --scale 4 --patch_size 256 --save edsr_baseline_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results
## train edsr in paper
python main.py --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --patch_size 256 --save edsr_paper_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --save_gt

## train baseline with epoch 300
python main.py --model EDSR --scale 4 --patch_size 256 --save edsr_baseline_x4_dynamic_lr --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --save_gt

## train baseline with epoch 500
python main.py --model EDSR --scale 4 --patch_size 256 --save edsr_baseline_x4_dynamic_lr_epoch500 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --save_gt

##debug SAN
python main.py --model SAN --scale 4 --patch_size 256 --save san_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --batch_size 8


## debug RCAN
python main.py --model RCAN --scale 4 --patch_size 256 --save rcan_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --batch_size 16

## debug srresnet
## ARW data
python main.py --model SRResNet --scale 4 --patch_size 256 --save srrsenet_x4_ARW --data_train SRRAW --data_test SRRAW --n_colors 4 --save_results --batch_size 16 --save_gt --labels HR --rgb_range 1
python main.py --model EDSR --scale 4 --patch_size 256 --save EDSR_x4_ARW --data_train SRRAW --data_test SRRAW --n_colors 4 --save_results --batch_size 16 --labels HR --rgb_range 1 --loss 100.0*L1
## PNG data
python main.py --model SRResNet --scale 4 --patch_size 256 --save srrsenet_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --batch_size 16


## debug outr model
python main.py --model SSL --scale 4 --patch_size 256 --save ssl_addfushion_x4 --data_train SRRAW --data_test SRRAW --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR+Diff
