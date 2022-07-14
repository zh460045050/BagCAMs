##Set as the path of the dataset
data_root="/mnt/nasv2/MILab/zhulei/nasv1_dmp/datas/dataset"

##Set as the path of the downloaded resnet checkpoint
check_path_cub="../BagCAMs-upload/checkpoints/CUB/resnet50/dawsol/last_checkpoint.pth.tar"
check_path_openimages="../BagCAMs-upload/checkpoints/OpenImages/resnet50/dawsol/last_checkpoint.pth.tar"
check_path_ilsvrc="../BagCAMs-upload/checkpoints/OpenImages/ILSVRC/dawsol/last_checkpoint.pth.tar"

###CUB-200
CUDA_VISIBLE_DEVICES=1 python main.py --data_root $data_root \
                --experiment_name "CUB_BagCAMs_DAWSOL_ResNet_Layer3" \
                --pretrained TRUE \
                --num_val_sample_per_class 0 \
                --large_feature_map TRUE \
                --save_dir 'test_logs' \
                --dataset_name "CUB" \
                --architecture "resnet50" \
                --wsol_method "dawsol" \
                --check_path $check_path_cub \
                --mode "test" \
                --post_method "BagCAMs" \
                --target_layer "layer3"


###OpenImages
CUDA_VISIBLE_DEVICES=1 python main.py --data_root $data_root \
                --experiment_name "OpenImages_BagCAMs_DAWSOL_ResNet_Layer3" \
                --pretrained TRUE \
                --num_val_sample_per_class 0 \
                --large_feature_map FALSE \
                --save_dir 'test_logs' \
                --dataset_name "OpenImages" \
                --architecture "resnet50" \
                --wsol_method "dawsol" \
                --check_path $check_path_openimages \
                --mode "test" \
                --post_method "BagCAMs" \
                --target_layer "layer3"


###ILSVRC
CUDA_VISIBLE_DEVICES=1 python main.py --data_root $data_root \
                --experiment_name "ILSVRC_BagCAMs_DAWSOL_ResNet_Layer3" \
                --pretrained TRUE \
                --num_val_sample_per_class 0 \
                --large_feature_map True \
                --save_dir 'test_logs' \
                --dataset_name "ILSVRC" \
                --architecture "resnet50" \
                --wsol_method "dawsol" \
                --check_path $check_path_ilsvrc \
                --mode "test" \
                --post_method "BagCAMs" \
                --target_layer "layer3"