cd src
##baseline
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py mot --exp_id bdd100k --gpus '0,1' --print_iter 100 --batch_size 12 --num_epochs 40 --load_model '../models/ctdet_coco_dla_2x.pth'
## change backbone to hrnet_32, reid_dim 128
# CUDA_VISIBLE_DEVICES=0,1,2 python -W ignore train.py mot --exp_id bdd_hrnet --gpus 0,1,2 --print_iter 100 --batch_size 16 --reid_dim 128 --arch 'hrnet_32' --load_model '../models/hrnetv2_w32_imagenet_pretrained.pth'
## use bdd-pretrained ctdet detection model, dla34 architecture, reid_dim 128
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py mot --exp_id bdd100k_pretrain_ctdet --gpus '0,1,2,3' --print_iter 100 --batch_size 64 --num_epochs 40 --reid_dim 128 --lr 1e-3 --load_model '../models/ctdet_bdd_dla_30.pth'
cd ..
