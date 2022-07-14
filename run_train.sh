#!/bin/bash
data_root="/mnt/nasv2/MILab/zhulei/nasv1_dmp/datas/dataset"
algs=("cam" "has" "adl" "cutmix" "dawsol")
lrs=(0.002248447 0.003619965 0.012534383 0.002251717 0.001)
for (( i = 0 ; i < ${#algs[@]} ; i++ ))
do
    CUDA_VISIBLE_DEVICES=3 python main.py --data_root $data_root \
                        --experiment_name ${algs[$i]} \
                        --pretrained TRUE \
                        --num_val_sample_per_class 0 \
                        --large_feature_map TRUE \
                        --batch_size 32 \
                        --epochs 50 \
                        --lr ${lrs[$i]} \
                        --lr_decay_frequency 15 \
                        --weight_decay 1.00E-04 \
                        --override_cache FALSE \
                        --workers 16 \
                        --box_v2_metric True \
                        --iou_threshold_list 30 50 70 \
                        --save_dir 'train_logs' \
                        --seed 4 \
                        --has_grid_size 120 \
                        --has_drop_rate 0.66 \
                        --adl_threshold 0.76 \
                        --adl_drop_rate 0.24 \
                        --cutmix_beta 1.35 \
                        --cutmix_prob 0.34 \
                        --beta 0.3 \
                        --univer 2 \
                        --uda_method mmd \
                        --dataset_name CUB \
                        --architecture resnet50 \
                        --wsol_method ${algs[$i]} \
                        --check_path "" \
                        --eval_frequency 5 \
                        --post_method "CAM"
done
