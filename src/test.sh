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
