# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import vision_transformer as vits  # Assuming this is the DINO ViT implementation

def apply_mask(image, mask, color, alpha=0.5):
    # Apply a colored mask to the image with specified transparency (alpha)
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    # Generate N random colors in RGB format using HSV color space
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    # Create a figure without axes for visualizing masked image
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]  # Add singleton dimension for mask
    colors = random_colors(N)  # Generate random color for mask

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)  # Set y-axis limits
    ax.set_xlim(-margin, width + margin)  # Set x-axis limits
    ax.axis('off')  # Hide axes
    masked_image = image.astype(np.uint32).copy()  # Copy image for modification
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))  # Apply blur to mask if specified
        masked_image = apply_mask(masked_image, _mask, color, alpha)  # Apply mask to image
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))  # Pad mask for contour detection
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)  # Find contours in mask
            for verts in contours:
                verts = np.fliplr(verts) - 1  # Adjust coordinates for plotting
                p = Polygon(verts, facecolor="none", edgecolor=color)  # Create polygon for contour
                ax.add_patch(p)  # Add contour to plot
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')  # Display masked image
    fig.savefig(fname)  # Save figure to file
    print(f"{fname} saved.")
    return


# Mean Attention Distance Functions
def compute_distance_matrix(patch_size, num_patches, length, cache={}):
    # 거리 행렬 캐싱: 동일한 patch_size/num_patches/length 조합이면 재계산하지 않음
    key = (patch_size, num_patches, length)
    if key in cache:
        return cache[key]
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:
                continue
            xi, yi = (int(i / length)), (i % length)  # Grid coordinates for patch i
            xj, yj = (int(j / length)), (j % length)  # Grid coordinates for patch j
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])  # Scaled Euclidean distance
    cache[key] = distance_matrix
    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights, num_cls_tokens=1):
    # CLS 토큰 제외
    attention_weights = attention_weights[:, num_cls_tokens:, num_cls_tokens:]  # Remove CLS token
    num_patches = attention_weights.shape[-1]  # Number of patches
    length = int(np.sqrt(num_patches))  # Grid side length
    assert length ** 2 == num_patches, "Num patches is not perfect square"

    # 거리 행렬 캐싱 사용
    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)  # Compute distance matrix
    distance_matrix = torch.from_numpy(distance_matrix).float().to(attention_weights.device)  # Convert to PyTorch tensor
    distance_matrix = distance_matrix.view(1, num_patches, num_patches)  # Reshape for broadcasting
    mean_distances = attention_weights * distance_matrix  # Weight distances by attention
    mean_distances = torch.sum(mean_distances, dim=-1)  # Sum over patches
    mean_distances = torch.mean(mean_distances, dim=-1)  # Average over patches
    return mean_distances.cpu().numpy()  # Return as NumPy array



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps and Mean Attention Distances')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./attn-output/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)


    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights found at {args.pretrained_weights} and loaded with msg: {msg}')
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    if args.image_path is None:
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("이미지 경로가 제공되지 않아 기본 이미지를 사용합니다.")
        
        # 기본 이미지 경로 (프로젝트 디렉토리 내 assets 폴더)
        default_img_path = os.path.join(
            os.path.dirname(__file__), 
            "assets", 
            "default.jpg"
        )
        
        if os.path.isfile(default_img_path):
            img = Image.open(default_img_path).convert('RGB')
        else:
            print(f"기본 이미지를 찾을 수 없습니다: {default_img_path}")
            print("다음과 같이 기본 이미지를 준비해주세요:")
            print("1. assets 디렉토리 생성: mkdir -p assets")
            print("2. 기본 이미지 복사: cp your_default_image.jpg assets/default.jpg")
            sys.exit(1)
        
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"제공된 이미지 경로가 유효하지 않습니다: {args.image_path}")
        sys.exit(1)

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]

    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()


    attentions_head = attentions.reshape(nh, h_featmap, w_featmap)
    attentions_head = nn.functional.interpolate(attentions_head.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    
    attentions_mean = torch.mean(attentions, dim=0).reshape(-1, h_featmap, w_featmap)  # [h_featmap, w_featmap]
    # breakpoint()
    attentions_mean = nn.functional.interpolate(attentions_mean.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()



    img_basename = os.path.splitext(os.path.basename(args.image_path if args.image_path else "img.png"))[0]
    model_tag = f"{args.arch}_patch{args.patch_size}"


    # Save attentions heatmaps and masks
    base_out_subdir = os.path.join(args.output_dir, f"{img_basename}_{model_tag}")  # Base directory: ./attn-dist/{img_basename}_{model_tag}/
    attn_map_subdir = os.path.join(base_out_subdir, "attention_maps")  # Subdirectory for attention maps
    mean_dist_subdir = os.path.join(base_out_subdir, "mean_distances")  # Subdirectory for mean attention distances
    os.makedirs(base_out_subdir, exist_ok=True)  # Create base directory
    os.makedirs(attn_map_subdir, exist_ok=True)  # Create attention maps directory
    os.makedirs(mean_dist_subdir, exist_ok=True)  # Create mean distances directory


    resize_img_name = f"{img_basename}_{model_tag}_resized.png"  # Resized image filename
    resize_img_path = os.path.join(base_out_subdir, resize_img_name)  # Resized image path
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        resize_img_path
    )  # Save resized image
    print(f"{resize_img_path} saved.")


    # Attention heatmap 저장
    for j in range(nh):
        fname = os.path.join(
            attn_map_subdir,
            f"{img_basename}_{model_tag}_attn-head{j}.png"
        )
        plt.imsave(fname=fname, arr=attentions_head[j], format='png')
        plt.close()
        print(f"{fname} saved.")


    fname2 = os.path.join(attn_map_subdir, f"{img_basename}_{model_tag}_attn-mean.png")
    plt.imsave(fname=fname2, arr=attentions_mean[0], format='png')
    plt.close()
    print(f"{fname2} saved.")


    if args.threshold is not None:
        image = skimage.io.imread(resize_img_path)
        for j in range(nh):
            mask_fname = os.path.join(attn_map_subdir, f"{img_basename}_{model_tag}_mask-th{args.threshold}_head{j}.png")  # Mask filename
            display_instances(image, th_attn[j], fname=mask_fname, blur=False)  # Save thresholded mask



    # Compute Mean Attention Distances for All Blocks
    mean_distances = {}
    num_cls_tokens = 1
    num_blocks = len(model.blocks)
    # 기존: for block_idx in range(num_blocks): ... (매번 forward)
    # 최적화: 한 번의 forward에서 모든 블록의 attention을 추출하도록 vision_transformer.py를 수정하면 속도 대폭 향상
    # 아래는 기존 방식(느림), get_all_selfattention이 있으면 위처럼 사용

    with torch.no_grad():
        for block_idx in range(num_blocks):
            # torch.no_grad()로 불필요한 그래디언트 계산 방지
            attentions = model.get_selfattention(img.to(device), block_idx)

            mean_distance = compute_mean_attention_dist(
                patch_size=args.patch_size,
                attention_weights=attentions[0],
                num_cls_tokens=num_cls_tokens
            )
            mean_distances[f"transformer_block_{block_idx}_att_mean_dist"] = mean_distance[None, :]


    # Visualize Mean Attention Distances
    plt.figure(figsize=(10, 10))  # 10x10 크기의 플롯 생성
    colors = plt.cm.tab10(np.linspace(0, 1, nh))  # 각 헤드에 고유 색상 할당 (tab10 팔레트)
    for head_idx in range(nh):  # 각 주의 헤드에 대해 반복
        x = list(range(len(mean_distances)))  # X축: 트랜스포머 블록 인덱스 (0, 1, ..., num_blocks-1)
        y = [mean_distances[f"transformer_block_{idx}_att_mean_dist"][0, head_idx] for idx in range(len(mean_distances))]  # Y축: 해당 헤드의 블록별 평균 거리
        plt.scatter(x, y, color=colors[head_idx], label=f"head_{head_idx}")  # 산점도, 동일 헤드 동일 색상
        for i, txt in enumerate(x):  # 각 점에 블록 번호 주석 추가
            plt.annotate(head_idx, (x[i] + 0.1, y[i] + 0.1))  # 헤드 번호 주석
    plt.xlabel("Transformer Blocks")  # X축 레이블: 트랜스포머 블록
    plt.ylabel("Mean Attention Distance")  # Y축 레이블: 평균 주의 거리
    plt.legend(loc="lower right")  # 범례: 오른쪽 하단
    plt.title(f"{model_tag} - Mean Attention Distance per Head")  # 제목: 모델명과 "Mean Attention Distance per Head"
    mean_dist_fname = os.path.join(mean_dist_subdir, f"{img_basename}_{model_tag}_mean_attention_distance1.png")  # 저장 경로: mean_distances 디렉토리
    plt.savefig(mean_dist_fname)  # 플롯을 PNG로 저장
    plt.close()  # 플롯 닫기
    print(f"{mean_dist_fname} saved.")  # 저장 완료 메시지 출력



    # Visualize Mean Attention Distances 2
    plt.figure(figsize=(10, 10))
    num_blocks = len(model.blocks)
    for idx in range(len(mean_distances)):
        if idx in {0, (num_blocks-1)//2 - 1, (num_blocks-1)//2 , num_blocks-2, num_blocks-1}:
            mean_distance = mean_distances[f"transformer_block_{idx}_att_mean_dist"]
            x = list(range(nh))
            y = mean_distance[0, :]
            plt.plot(x, y, marker='o', label=f"block_{idx}")
            for i, txt in enumerate(range(nh)):
                plt.annotate(txt, (x[i] + 0.1, y[i] + 0.1))
    plt.xlabel("Attention Heads")
    plt.ylabel("Mean Attention Distance")
    plt.legend(loc="lower right")
    plt.title(f"{model_tag} - Mean Attention Distance per Head")
    mean_dist_fname = os.path.join(mean_dist_subdir, f"{img_basename}_{model_tag}_mean_attention_distance2.png")
    plt.savefig(mean_dist_fname)
    plt.close()
    print(f"{mean_dist_fname} saved.")