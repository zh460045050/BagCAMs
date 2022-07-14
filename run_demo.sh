#Runing Demo with our checkpoint
check_path=""

CUDA_VISIBLE_DEVICES=1 python demo.py \
                --img_dir "./demo_birds" \
                --check_path $check_path \
                --save_path "./birds_results" \
                --post_method "BagCAMs" \
                --target_layer "layer3"

CUDA_VISIBLE_DEVICES=1 python demo.py \
                --img_dir "./demo_birds" \
                --check_path $check_path \
                --save_path "./birds_results" \
                --post_method "CAM" \
                --target_layer "layer3"

CUDA_VISIBLE_DEVICES=1 python demo.py \
                --img_dir "./demo_birds" \
                --check_path $check_path \
                --save_path "./birds_results" \
                --post_method "GradCAM" \
                --target_layer "layer3"

CUDA_VISIBLE_DEVICES=1 python demo.py \
                --img_dir "./demo_birds" \
                --check_path $check_path \
                --save_path "./birds_results" \
                --post_method "GradCAM++" \
                --target_layer "layer3"

CUDA_VISIBLE_DEVICES=1 python demo.py \
                --img_dir "./demo_birds" \
                --check_path $check_path \
                --save_path "./birds_results" \
                --post_method "PCS" \
                --target_layer "layer3"
