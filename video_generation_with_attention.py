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
from datetime import datetime

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
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
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
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return



def extract_frames(video_path):
    """
    영상 파일에서 프레임들을 추출하여 리스트로 반환.
    반환: frames(리스트), fps(실수), width(정수), height(정수)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Cannot open video file')
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, width, height


def visualize_attention(frames, args):
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
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
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


    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    attn_frames = []
    for idx, frame in enumerate(frames):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = transform(img_pil)
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size  # 패치사이즈의 배수가 되게끔 조정
        img = img[:, :w, :h].unsqueeze(0)
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size
        with torch.no_grad():
            attentions = model.get_last_selfattention(img.to(device))
        # 마지막 레이어 마지막 헤드의 attention
        last_layer_attention = attentions[0, -1, 0, 1:].reshape(w_featmap, h_featmap).cpu().numpy()
        last_layer_attention = (last_layer_attention - last_layer_attention.min()) / (np.ptp(last_layer_attention) + 1e-6)
        last_layer_attention = cv2.resize(last_layer_attention, tuple(args.image_size), interpolation=cv2.INTER_NEAREST)
        last_layer_attention = (last_layer_attention * 255).astype(np.uint8)
        color_attn = cv2.applyColorMap(last_layer_attention, cv2.COLORMAP_JET)
        attn_frames.append(color_attn)


        del attentions, img, img_pil, last_layer_attention, color_attn
        torch.cuda.empty_cache()

        # 진행상황 프린트 (대용량 비디오 처리시)
        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1} / {len(frames)} frames...")

    return attn_frames


def save_video(frames, out_path, fps, size):
    """
    프레임 리스트(frames)를 비디오 파일로 저장.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--video_path",  default='/home/work/wonjun/dino/input2.mp4', type=str, help="Path of the input video file.")
    parser.add_argument("--image_size", default=None, type=int, nargs=2, help="Resize image (h w).")
    parser.add_argument('--output_dir', default='/home/work/wonjun/dino/attention_map', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # 입력 비디오 → 프레임 리스트
    frames, fps, width, height = extract_frames(args.video_path)


    # 비디오 프레임 크기로 image_size 덮어쓰기
    if args.image_size is None:
        # transforms.Resize expects (h, w)
        args.image_size = (int(height*0.8), int(width*0.8))
    else:
        # 사용자가 --image_size 를 지정했으면 (h, w) 형태로 강제 변환
        args.image_size = tuple(args.image_size)


    # Attention map 프레임 생성
    attn_frames = visualize_attention(frames, args)


    # 비디오로 저장
    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.now().strftime("%H%M")   # 시분(예: 1423)
    output_video_path = os.path.join(
        args.output_dir,
        f'output_{now}.mp4'
    )
    save_video(attn_frames, output_video_path, fps, tuple(args.image_size))
    print(f"Attention map video saved to {output_video_path}")