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

def apply_mask(image, mask, color, alpha=0.5):
    """
    이미지에 컬러 마스크를 alpha 투명도로 적용
    """
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    N개의 랜덤 RGB 색상 생성 (HSV 기반)
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    """
    마스크가 적용된 이미지를 저장 (contour, blur 옵션 지원)
    """
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    colors = random_colors(N)

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

if __name__ == '__main__':
    # 인자 파서 정의
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
    # 모델 생성 및 준비
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # 사전학습 가중치 로드
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # 멀티 GPU 및 백본 키 제거
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        # 사전학습 가중치가 없으면 공식 DINO 가중치 다운로드
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


    # 이미지 로드
    if args.image_path is None:
        # 이미지 경로 미지정 시 샘플 이미지 다운로드
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)

    tmp_img = img  # 원본 이미지 백업

    # 이미지 전처리 transform 정의
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # 패치 사이즈에 맞게 이미지 크기 조정 (오른쪽/아래쪽 잘림)
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    # 마지막 self-attention 맵 추출
    attentions = model.get_last_selfattention(img.to(device))   # (1, 6, 60*60, 60*60)
    # attentions = model.get_selfattention(img.to(device), 0)


    nh = attentions.shape[1] # number of head


    # CLS 토큰에서 각 패치로의 attention만 추출 및 최대값 위치 계산
    cls_attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  # (num_heads, num_patches)
    max_attention_val_idx = torch.max(cls_attentions, dim=-1)[1].cpu()  # CPU로 이동


    # second_attention_val_idx = torch.topk(cls_attentions, k=2, dim=-1)[1][:, 1].cpu()  # 2번째로 큰 패치 인덱스



    # 해당 패치의 (row, col) 좌표 계산
    patch_pos = (max_attention_val_idx//w_featmap, max_attention_val_idx%w_featmap)
    # patch_pos = (second_attention_val_idx//w_featmap, second_attention_val_idx%w_featmap)


    for i in range(nh):
        # 결과 저장 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        img_basename = os.path.splitext(
            os.path.basename(args.image_path if args.image_path else "img.png")
        )[0]
        model_tag = f"head{i}_{args.arch}_patch{args.patch_size}"
        out_subdir = os.path.join(args.output_dir, f"{img_basename}_{model_tag}")
        os.makedirs(out_subdir, exist_ok=True)

        # # 1. 원본 이미지에 최고 어텐션값 패치 테두리 표시
        plt.figure(figsize=(10, 10))
        img_display = np.asarray(tmp_img.resize((w, h)))
        plt.imshow(img_display)
        
        # 최대 어텐션값을 가지는 패치 위치 계산
        row_start = int(patch_pos[0][i].item()) * args.patch_size
        col_start = int(patch_pos[1][i].item()) * args.patch_size
        
        # 빨간색 테두리 추가
        rect = plt.Rectangle(
            (col_start, row_start),
            args.patch_size, 
            args.patch_size,
            fill=False,
            color='red',
            linewidth=2
        )
        plt.gca().add_patch(rect)
        plt.axis('off')
        plt.title(f'Head {i} Most Attended Patch')
        
        # 저장
        fname = os.path.join(out_subdir, f"{img_basename}_head{i}_patch_highlight.png")
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"{fname} saved.")

        # 2. 최대 어텐션 패치를 쿼리로 하는 어텐션맵
        plt.figure(figsize=(10, 10))
        # 해당 패치의 어텐션맵 추출 (패치 인덱스 = row * w_featmap + col)
        patch_idx = int(patch_pos[0][i].item() * w_featmap + patch_pos[1][i].item()) + 1  # +1은 CLS 토큰 때문
        patch_attn = attentions[0, i, patch_idx, 1:].reshape(h_featmap, w_featmap).cpu()
        
        # 원본 이미지 크기로 업샘플링
        patch_attn = nn.functional.interpolate(
            patch_attn.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="nearest"
        )[0, 0].numpy()
        
        plt.imshow(patch_attn, cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.axis('off')
        plt.title(f'Head {i} Attention from Most Attended Patch')
        
        fname = os.path.join(out_subdir, f"{img_basename}_head{i}_patch_to_others.png")
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"{fname} saved.")

    # 각 헤드별로 3개의 시각화를 하나의 행으로 표시
    plt.figure(figsize=(15, 5*nh))  # 가로 15, 세로는 헤드 수에 비례
    
    for i in range(nh):
        # 1. CLS 토큰의 어텐션맵
        plt.subplot(nh, 3, 3*i + 1)
        cls_attn = attentions[0, i, 0, 1:].reshape(h_featmap, w_featmap).cpu()
        cls_attn = nn.functional.interpolate(
            cls_attn.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="nearest"
        )[0, 0].numpy()
        plt.imshow(cls_attn, cmap='viridis')
        plt.colorbar(label='Attention')
        plt.axis('off')
        plt.title(f'Head {i} - CLS Token Attention')
        
        # 2. 원본 이미지 + 최대 어텐션 패치 표시
        plt.subplot(nh, 3, 3*i + 2)
        img_display = np.asarray(tmp_img.resize((w, h)))
        plt.imshow(img_display)
        
        # 최대 어텐션 패치 위치에 빨간색 테두리
        row_start = int(patch_pos[0][i].item()) * args.patch_size
        col_start = int(patch_pos[1][i].item()) * args.patch_size
        rect = plt.Rectangle(
            (col_start, row_start),
            args.patch_size, 
            args.patch_size,
            fill=False,
            color='red',
            linewidth=2
        )
        plt.gca().add_patch(rect)
        plt.axis('off')
        plt.title(f'Head {i} - Most Attended Patch')
        
        # 3. 최대 어텐션 패치의 어텐션맵
        plt.subplot(nh, 3, 3*i + 3)
        patch_idx = int(patch_pos[0][i].item() * w_featmap + patch_pos[1][i].item()) + 1
        patch_attn = attentions[0, i, patch_idx, 1:].reshape(h_featmap, w_featmap).cpu()
        patch_attn = nn.functional.interpolate(
            patch_attn.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="nearest"
        )[0, 0].numpy()
        plt.imshow(patch_attn, cmap='viridis')
        plt.colorbar(label='Attention')
        plt.axis('off')
        plt.title(f'Head {i} - Attention from Most Attended Patch')

    
    # 전체 figure 제목 추가
    plt.suptitle(f'Attention Analysis ({args.arch}_patch{args.patch_size})', 
                fontsize=16, y=0.95)
    
    # 여백 조정
    plt.tight_layout()
    
    # 저장
    model_tag = f"{args.arch}_patch{args.patch_size}"
    out_dir = os.path.join(args.output_dir, f"{img_basename}_{model_tag}")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{img_basename}_attention_analysis.png")
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n시각화 저장 완료: {fname}")

    # # 4. 각 헤드의 최대 어텐션 패치가 모든 헤드에서 어떻게 보이는지 시각화
    # for i in range(nh):  # 각 헤드별로
    #     plt.figure(figsize=(20, 15))
    #     n_cols = min(3, nh)
    #     n_rows = (nh + n_cols - 1) // n_cols

    #     # i번째 헤드의 최대 어텐션 패치 위치
    #     patch_idx = int(patch_pos[0][i].item() * w_featmap + patch_pos[1][i].item()) + 1
        
    #     # 원본 이미지 크기 확인 (tmp_img 사용)
    #     W, H = tmp_img.size  # PIL Image에서 크기 가져오기
        
    #     # 패치 크기로 나누어떨어지게 조정
    #     H = H - (H % args.patch_size)
    #     W = W - (W % args.patch_size)

    #     # 모든 헤드에서의 어텐션맵 시각화
    #     for h in range(nh):
    #         plt.subplot(n_rows, n_cols, h + 1)
            
    #         # h번째 헤드에서 i번째 헤드의 최대 어텐션 패치의 어텐션맵
    #         cross_attn = attentions[0, h, patch_idx, 1:].reshape(h_featmap, w_featmap).cpu()
            
    #         # 원본 이미지 크기로 업샘플링 (크기 명시적 지정)
    #         cross_attn = nn.functional.interpolate(
    #             cross_attn.unsqueeze(0).unsqueeze(0),
    #             size=(H, W),  # 명시적으로 조정된 크기 사용
    #             mode="nearest"
    #         )[0, 0].numpy()
            
    #         plt.imshow(cross_attn, cmap='viridis')
    #         plt.colorbar(label='Attention')
    #         plt.axis('off')
            
    #         # 현재 보고 있는 헤드가 원본 헤드인 경우 강조
    #         if h == i:
    #             title = f'Head {h} (Source)'
    #             plt.title(title, color='red', fontweight='bold')
    #         else:
    #             title = f'Head {h}'
    #             plt.title(title)

    #     # 전체 figure 제목
    #     plt.suptitle(
    #         f'Head {i} Most Attended Patch (row={patch_pos[0][i].item()}, col={patch_pos[1][i].item()})\n'
    #         f'Attention Patterns across All Heads', 
    #         fontsize=16, y=0.95
    #     )
        
    #     # 여백 조정
    #     plt.tight_layout()
        
    #     # 저장
    #     fname = os.path.join(out_dir, f"{img_basename}_head{i}_cross_head_attention.png")
    #     plt.savefig(fname, bbox_inches='tight', dpi=300)
    #     plt.close()
    #     print(f"Cross-head attention analysis for Head {i} saved: {fname}")

    print("\n모든 시각화 완료:")
    print("1. *_attention_analysis.png: 각 헤드별 기본 어텐션 분석 (CLS/패치/어텐션맵)")
    print("2. *_head{i}_cross_head_attention.png: 각 헤드의 최대 어텐션 패치가 다른 헤드들에서 보이는 패턴")

