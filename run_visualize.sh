# #!/bin/bash

# conda deactivate
# source wonjun/dino/dino/bin/activate

# # multi_obj1.png에 대한 명령어
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj1.png --arch vit_small --patch_size 8
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj1.png --arch vit_small --patch_size 16
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj1.png --arch vit_base --patch_size 8
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj1.png --arch vit_base --patch_size 16

# # multi_obj2.jpg에 대한 명령어
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj2.jpg --arch vit_small --patch_size 8
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj2.jpg --arch vit_small --patch_size 16
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj2.jpg --arch vit_base --patch_size 8
# python visualize_attention_mean_attn_dist.py --image_path ./input/multi_obj2.jpg --arch vit_base --patch_size 16

#!/usr/bin/env sh
set -eu

# 기본값(필요하면 환경변수로 덮어쓰기 가능)
IMAGE_DIR="${IMAGE_DIR:-/home/work/wonjun/probing-vits/notebooks/1000_val_images_sampled}"
PY="${PY:-python}"
SCRIPT="${SCRIPT:-visualize_mean_attention_dist_multi.py}"

for ARCH in vit_small vit_base; do
  for PATCH in 8 16; do
    echo "==> Running: $PY $SCRIPT --image_dir \"$IMAGE_DIR\" --arch $ARCH --patch_size $PATCH"
    "$PY" "$SCRIPT" --image_dir "$IMAGE_DIR" --arch "$ARCH" --patch_size "$PATCH"
  done
done
