data_root="/mnt/nasv2/MILab/zhulei/nasv1_dmp/datas/dataset"
post_methods=("BagCAMs" "PCS" "GradCAM" "GradCAM++" "CAM")
wsol_methods=("dawsol")
layers=("layer3" "layer4" "layer2" "layer1")

##Set as the path of the downloaded checkpoint
check_path=""

for (( j = 0 ; j < ${#wsol_methods[@]} ; j++ ))
do
    check_path="checkpoints/"${wsol_methods[$j]}"/last_checkpoint.pth.tar"
    for (( k = 0 ; k < ${#layers[@]} ; k++ ))
    do
        layer=${layers[$k]}
        for (( i = 0 ; i < ${#post_methods[@]} ; i++ ))
        do
            post_method=${post_methods[$i]}
            CUDA_VISIBLE_DEVICES=1 python main.py --data_root $data_root \
                            --experiment_name ${wsol_methods[$j]}"_"$layer"_"$post_method \
                            --pretrained TRUE \
                            --num_val_sample_per_class 0 \
                            --mode "test" \
                            --large_feature_map TRUE \
                            --save_dir 'test_logs_CUB_RES' \
                            --dataset_name "CUB" \
                            --architecture "resnet50" \
                            --wsol_method ${wsol_methods[$j]} \
                            --check_path $check_path \
                            --post_method $post_method \
                            --target_layer $layer
        done
    done
done
