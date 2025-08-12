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

import utils
import vision_transformer as vits


def create_concatenated_image(original_img):
    """
    원본 이미지와 좌우반전 이미지를 가로로 이어붙인 하나의 이미지 생성
    
    Args:
        original_img: 원본 PIL Image 객체
    
    Returns:
        concatenated_img: 가로로 이어붙인 PIL Image 객체 (원본 | 좌우반전)
    """
    # 좌우반전 이미지 생성
    flipped_img = original_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 두 이미지를 가로로 이어붙이기
    width, height = original_img.size
    concatenated_img = Image.new('RGB', (width * 2, height))
    
    # 왼쪽에 원본, 오른쪽에 반전된 이미지 붙이기
    concatenated_img.paste(original_img, (0, 0))
    concatenated_img.paste(flipped_img, (width, 0))
    
    return concatenated_img


def analyze_concatenated_attention(original_img, model, device, args):
    """
    이어붙인 이미지에서 원본 부분의 최대 attention 패치가 
    전체 이미지(원본+반전)의 다른 패치들에 주는 attention 분석
    
    Args:
        original_img: 원본 PIL 이미지
        model: DINO 모델
        device: torch device
        args: 명령행 인자
    
    Returns:
        dict: attention 분석 결과 데이터
    """
    
    # 이어붙인 이미지 생성
    concat_img = create_concatenated_image(original_img)
    print(f"이어붙인 이미지 크기: {concat_img.size}")
    
    # 이미지 전처리 transform 정의
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 원본 이미지와 이어붙인 이미지 전처리
    orig_tensor = transform(original_img)
    concat_tensor = transform(concat_img)
    
    # 패치 사이즈에 맞게 이미지 크기 조정
    orig_w, orig_h = (orig_tensor.shape[1] - orig_tensor.shape[1] % args.patch_size, 
                      orig_tensor.shape[2] - orig_tensor.shape[2] % args.patch_size)
    concat_w, concat_h = (concat_tensor.shape[1] - concat_tensor.shape[1] % args.patch_size,
                          concat_tensor.shape[2] - concat_tensor.shape[2] % args.patch_size)
    
    orig_tensor = orig_tensor[:, :orig_w, :orig_h].unsqueeze(0)
    concat_tensor = concat_tensor[:, :concat_w, :concat_h].unsqueeze(0)
    
    # feature map 크기 계산
    orig_w_featmap = orig_tensor.shape[-2] // args.patch_size
    orig_h_featmap = orig_tensor.shape[-1] // args.patch_size
    concat_w_featmap = concat_tensor.shape[-2] // args.patch_size
    concat_h_featmap = concat_tensor.shape[-1] // args.patch_size
    
    print(f"원본 feature map 크기: {orig_w_featmap} x {orig_h_featmap}")
    print(f"이어붙인 feature map 크기: {concat_w_featmap} x {concat_h_featmap}")
    
    # 각 이미지에서 attention 추출
    orig_attentions = model.get_last_selfattention(orig_tensor.to(device))
    concat_attentions = model.get_last_selfattention(concat_tensor.to(device))
    
    nh = orig_attentions.shape[1]  # number of heads
    
    # 원본 이미지에서 CLS 토큰의 최대 attention 패치 위치 찾기
    orig_cls_attentions = orig_attentions[0, :, 0, 1:].reshape(nh, -1)
    orig_max_attention_idx = torch.max(orig_cls_attentions, dim=-1)[1].cpu()
    orig_patch_pos = (orig_max_attention_idx // orig_h_featmap, orig_max_attention_idx % orig_h_featmap)
    
    # 이어붙인 이미지에서 원본 부분의 최대 attention 패치에 해당하는 위치 계산
    # 이어붙인 이미지에서 원본 부분은 왼쪽 절반에 해당
    concat_patch_positions = []
    for head_idx in range(nh):
        orig_row = orig_patch_pos[0][head_idx].item()
        orig_col = orig_patch_pos[1][head_idx].item()
        
        # 이어붙인 이미지에서의 해당 패치 인덱스 계산
        # 이어붙인 이미지는 가로가 2배이므로 열의 비율을 조정
        concat_col = int(orig_col * concat_h_featmap / orig_h_featmap)
        concat_row = int(orig_row * concat_w_featmap / orig_w_featmap)
        
        concat_patch_idx = concat_row * concat_h_featmap + concat_col + 1  # +1 for CLS token
        concat_patch_positions.append((concat_row, concat_col, concat_patch_idx))
    
    return {
        'original_img': original_img,
        'concat_img': concat_img,
        'orig_attentions': orig_attentions,
        'concat_attentions': concat_attentions,
        'orig_patch_pos': orig_patch_pos,
        'concat_patch_positions': concat_patch_positions,
        'orig_w': orig_w, 'orig_h': orig_h,
        'concat_w': concat_w, 'concat_h': concat_h,
        'orig_w_featmap': orig_w_featmap, 'orig_h_featmap': orig_h_featmap,
        'concat_w_featmap': concat_w_featmap, 'concat_h_featmap': concat_h_featmap,
        'nh': nh
    }


def visualize_concatenated_attention(data, args, img_basename):
    """
    이어붙인 이미지에서의 attention 분석 결과 시각화
    
    Args:
        data: analyze_concatenated_attention에서 반환된 데이터
        args: 명령행 인자
        img_basename: 이미지 파일 베이스네임
    """
    
    nh = data['nh']
    orig_w, orig_h = data['orig_w'], data['orig_h']
    concat_w, concat_h = data['concat_w'], data['concat_h']
    
    # 결과 저장 디렉토리 생성
    model_tag = f"{args.arch}_patch{args.patch_size}"
    out_dir = os.path.join(args.output_dir, f"{img_basename}_{model_tag}_concatenated")
    os.makedirs(out_dir, exist_ok=True)
    
    # 각 헤드별로 시각화 생성
    for head_idx in range(nh):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 원본 이미지에서의 CLS attention과 최대 attention 패치 표시
        orig_cls_attn = data['orig_attentions'][0, head_idx, 0, 1:].reshape(
            data['orig_w_featmap'], data['orig_h_featmap']).cpu()
        orig_cls_attn_resized = nn.functional.interpolate(
            orig_cls_attn.unsqueeze(0).unsqueeze(0),
            size=(orig_w, orig_h), mode="nearest"
        )[0, 0].numpy()
        
        # 원본 이미지 표시
        axes[0, 0].imshow(np.asarray(data['original_img'].resize((orig_h, orig_w))))
        
        # 최대 attention 패치에 빨간 테두리 추가
        orig_row_start = int(data['orig_patch_pos'][0][head_idx].item()) * args.patch_size
        orig_col_start = int(data['orig_patch_pos'][1][head_idx].item()) * args.patch_size
        orig_rect = plt.Rectangle(
            (orig_col_start, orig_row_start), args.patch_size, args.patch_size,
            fill=False, color='red', linewidth=3
        )
        axes[0, 0].add_patch(orig_rect)
        axes[0, 0].set_title(f'Head {head_idx} - Original Image\n(Red: Max Attended Patch)')
        axes[0, 0].axis('off')
        
        # 2. 원본 이미지에서의 CLS attention map
        im1 = axes[0, 1].imshow(orig_cls_attn_resized, cmap='viridis')
        axes[0, 1].set_title(f'Head {head_idx} - Original CLS Attention')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. 이어붙인 이미지 표시
        concat_img_display = np.asarray(data['concat_img'].resize((concat_h, concat_w)))
        axes[1, 0].imshow(concat_img_display)
        
        # 이어붙인 이미지에서 원본의 최대 attention 패치 위치에 빨간 테두리
        concat_row, concat_col, _ = data['concat_patch_positions'][head_idx]
        concat_row_start = concat_row * args.patch_size
        concat_col_start = concat_col * args.patch_size
        concat_rect = plt.Rectangle(
            (concat_col_start, concat_row_start), args.patch_size, args.patch_size,
            fill=False, color='red', linewidth=3
        )
        axes[1, 0].add_patch(concat_rect)
        
        # 중간 구분선 표시 (원본과 반전 경계)
        middle_col = concat_h // 2
        axes[1, 0].axvline(x=middle_col, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
        axes[1, 0].set_title(f'Head {head_idx} - Concatenated Image\n(Red: Query Patch, Yellow: Original|Flipped)')
        axes[1, 0].axis('off')
        
        # 4. 핵심: 원본의 최대 attention 패치가 이어붙인 이미지의 다른 패치들에 주는 attention
        _, _, concat_patch_idx = data['concat_patch_positions'][head_idx]
        
        # 이어붙인 이미지에서 해당 패치의 attention map 추출
        patch_to_others_attn = data['concat_attentions'][0, head_idx, concat_patch_idx, 1:].reshape(
            data['concat_w_featmap'], data['concat_h_featmap']).cpu()
        patch_to_others_resized = nn.functional.interpolate(
            patch_to_others_attn.unsqueeze(0).unsqueeze(0),
            size=(concat_w, concat_h), mode="nearest"
        )[0, 0].numpy()
        
        im2 = axes[1, 1].imshow(patch_to_others_resized, cmap='viridis')
        axes[1, 1].set_title(f'Head {head_idx} - Attention from Max Patch\n(Original → Original|Flipped)')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 중간 구분선도 attention map에 표시
        axes[1, 1].axvline(x=concat_h // 2, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
        
        # 전체 제목 설정
        fig.suptitle(f'Concatenated Image Attention Analysis - Head {head_idx}\n'
                    f'({args.arch}_patch{args.patch_size})', 
                    fontsize=16, y=0.95)
        
        # 여백 조정
        plt.tight_layout()
        
        # 저장
        fname = os.path.join(out_dir, f"{img_basename}_head{head_idx}_concat_analysis.png")
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Head {head_idx} 이어붙인 이미지 분석 저장: {fname}")
    
    return out_dir


def create_attention_summary(data, args, img_basename):
    """
    모든 헤드의 이어붙인 이미지 attention을 요약하여 표시
    """
    nh = data['nh']
    concat_w, concat_h = data['concat_w'], data['concat_h']
    
    # 결과 저장 디렉토리
    model_tag = f"{args.arch}_patch{args.patch_size}"
    out_dir = os.path.join(args.output_dir, f"{img_basename}_{model_tag}_concatenated")
    
    # 모든 헤드의 패치별 attention을 한 번에 표시
    fig, axes = plt.subplots(1, nh, figsize=(6*nh, 6))
    if nh == 1:
        axes = [axes]
    
    for head_idx in range(nh):
        # 각 헤드에서 원본의 최대 attention 패치가 이어붙인 이미지에 주는 attention
        _, _, concat_patch_idx = data['concat_patch_positions'][head_idx]
        
        patch_to_others_attn = data['concat_attentions'][0, head_idx, concat_patch_idx, 1:].reshape(
            data['concat_w_featmap'], data['concat_h_featmap']).cpu()
        patch_to_others_resized = nn.functional.interpolate(
            patch_to_others_attn.unsqueeze(0).unsqueeze(0),
            size=(concat_w, concat_h), mode="nearest"
        )[0, 0].numpy()
        
        im = axes[head_idx].imshow(patch_to_others_resized, cmap='viridis')
        axes[head_idx].set_title(f'Head {head_idx}\nMax Patch → All')
        axes[head_idx].axis('off')
        
        # 구분선 표시
        axes[head_idx].axvline(x=concat_h // 2, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
        
        plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)
    
    fig.suptitle(f'All Heads: Attention from Original Max Patch to Concatenated Image\n'
                f'({args.arch}_patch{args.patch_size}) - Yellow line: Original|Flipped boundary', 
                fontsize=14, y=0.95)
    plt.tight_layout()
    
    # 저장
    fname = os.path.join(out_dir, f"{img_basename}_all_heads_concat_summary.png")
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"전체 헤드 요약 저장: {fname}")


if __name__ == '__main__':
    # 인자 파서 정의 (기존과 동일)
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
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

    # 디바이스 설정 (GPU 우선)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 모델 생성 및 준비 (기존과 동일)
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # 사전학습 가중치 로드 (기존과 동일)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
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

    # 이미지 로드 (기존과 동일)
    if args.image_path is None:
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img_basename = "sample_img"
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img_basename = os.path.splitext(os.path.basename(args.image_path))[0]
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)

    print("=== DINO Concatenated Image Attention Experiment ===")
    print(f"원본 이미지 크기: {img.size}")
    print(f"모델: {args.arch}, 패치 크기: {args.patch_size}")
    
    # === 이어붙인 이미지 실험 ===
    
    print("이어붙인 이미지 attention 분석 시작...")
    
    # 1. 이어붙인 이미지에서 attention 분석
    attention_data = analyze_concatenated_attention(img, model, device, args)
    
    # 2. 결과 시각화
    output_dir = visualize_concatenated_attention(attention_data, args, img_basename)
    
    # 3. 요약 시각화 생성
    create_attention_summary(attention_data, args, img_basename)
    
    print(f"\n=== 실험 완료 ===")
    print(f"결과 저장 위치: {output_dir}")
    print("생성된 시각화:")
    print("- 각 헤드별 상세 분석 (head{N}_concat_analysis.png)")
    print("- 전체 헤드 요약 (all_heads_concat_summary.png)")
    print("\n실험 내용:")
    print("- 원본 이미지 | 좌우반전 이미지를 가로로 이어붙임")
    print("- 원본에서 가장 높은 attention을 받은 패치 (빨간 테두리)")
    print("- 해당 패치가 전체 이어붙인 이미지의 다른 모든 패치들에 주는 attention 분석")
    print("- 노란 점선: 원본과 반전 이미지의 경계")