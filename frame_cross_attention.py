#!/usr/bin/env python3
"""
frame_cross_attention.py

프레임 간 Cross-Attention map을 계산하여 비디오로 저장하는 스크립트입니다.
각 프레임 t의 Query와 t+1의 Key/Value를 이용해 attention map을 생성하며,
이를 컬러맵으로 시각화한 뒤 연속 영상으로 출력합니다.
"""
import os
import argparse
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as pth_transforms

import vision_transformer as vits


def extract_frames(video_path):
    """
    비디오에서 프레임과 메타정보(fps, width, height)를 추출
    반환: frames(list of BGR np.ndarray), fps(float), width(int), height(int)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
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


def build_model(args, device):
    """
    ViT 모델과 Cross-Attention 레이어, 이미지 전처리(transform)를 생성
    """
    # ViT 모델 생성 및 weight 로드
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # 사전학습 가중치 로드 (원본 스크립트와 동일한 로직)
    if os.path.isfile(args.pretrained_weights):
        state = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key and args.checkpoint_key in state:
            state = state[args.checkpoint_key]
        # prefix 제거
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Pretrained weights not found: {args.pretrained_weights}")

    # Cross-Attention 레이어 정의
    # embed_dim, num_heads 추출 시도
    if hasattr(model, 'embed_dim') and hasattr(model, 'num_heads'):
        embed_dim = model.embed_dim
        num_heads = model.num_heads
    else:
        # transformer block에서 추출
        block = model.blocks[0]
        embed_dim = block.attn.embed_dim if hasattr(block.attn, 'embed_dim') else block.embed_dim
        num_heads = block.attn.num_heads
    cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)
    cross_attn.to(device)

    # 이미지 전처리
    transform = pth_transforms.Compose([
        pth_transforms.Resize(tuple(args.image_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return model, cross_attn, transform


def get_patch_embeddings(model, img_tensor):
    """
    ViT 모델의 patch embedding + cls_token + pos_embed 반환
    img_tensor: [1, 3, H, W]
    반환: [1, 1+num_patches, embed_dim]
    """
    # Patch Embedding
    x = model.patch_embed(img_tensor)
    B, N, C = x.shape
    # CLS 토큰 추가
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls, x), dim=1)
    # Positional Embedding
    x = x + model.pos_embed
    return x


def compute_cross_attention(embed_q, embed_kv, cross_attn):
    """
    Query(embed_q)와 Key/Value(embed_kv) 사이의 Cross-Attention 계산
    embed_*: [1, L, C]
    반환: attention weights [batch, tgt_len, src_len]
    """
    # MHA 입력: [seq_len, batch, embed_dim]
    q = embed_q.permute(1, 0, 2)
    k = embed_kv.permute(1, 0, 2)
    v = k
    _, attn_w = cross_attn(q, k, v, need_weights=True)
    # attn_w: [batch, tgt_len, src_len]
    return attn_w


def visualize_cross_attention(attn_w, args):
    """
    Cross-Attention map을 컬러맵 이미지로 변환
    """
    # CLS 토큰 쿼리 → 패치 키 간 attention
    w_f = args.image_size[0] // args.patch_size
    h_f = args.image_size[1] // args.patch_size
    attn = attn_w[0, 0, 1:]  # [num_patches]
    attn = attn.reshape(w_f, h_f).cpu().numpy()
    attn = (attn - attn.min()) / (attn.ptp() + 1e-6)
    attn_img = (attn * 255).astype(np.uint8)
    attn_img = cv2.resize(attn_img, tuple(args.image_size)[::-1], interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(attn_img, cv2.COLORMAP_JET)


def save_video(frames, out_path, fps):
    """
    BGR 프레임 리스트를 비디오로 저장
    """
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute frame-to-frame Cross-Attention maps")
    parser.add_argument('--video_path', type=str, required=True, help='입력 비디오 파일 경로')
    parser.add_argument('--arch', type=str, default='vit_small', choices=['vit_tiny','vit_small','vit_base'], help='ViT 아키텍처')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch 크기')
    parser.add_argument('--pretrained_weights', type='', required=True, help='사전학습 가중치 경로')
    parser.add_argument('--checkpoint_key', type=str, default='teacher', help='체크포인트 키')
    parser.add_argument('--image_size', type=int, nargs=2, default=None, help='리사이즈 이미지 크기 (h w)')
    parser.add_argument('--output_dir', type=str, default='./cross_attention_map', help='출력 디렉토리')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 프레임 추출
    frames, fps, w0, h0 = extract_frames(args.video_path)
    # image_size 설정
    if args.image_size is None:
        # patch_size 배수로 crop
        H = (h0 // args.patch_size) * args.patch_size
        W = (w0 // args.patch_size) * args.patch_size
        args.image_size = (H, W)
    else:
        args.image_size = tuple(args.image_size)

    # 모델 및 transform, cross_attn 레이어 준비
    model, cross_attn, transform = build_model(args, device)

    # Cross-Attention map 생성
    cross_frames = []
    for i in range(len(frames)-1):
        # 전처리 및 텐서 변환
        img0 = transform(Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        img1 = transform(Image.fromarray(cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        # patch_size 단위로 크롭
        H, W = img0.shape[2], img0.shape[3]
        Hc = H - (H % args.patch_size)
        Wc = W - (W % args.patch_size)
        img0 = img0[:, :, :Hc, :Wc]
        img1 = img1[:, :, :Hc, :Wc]

        # Embedding 추출
        emb0 = get_patch_embeddings(model, img0)
        emb1 = get_patch_embeddings(model, img1)
        # Cross-Attention 계산
        attn_w = compute_cross_attention(emb0, emb1, cross_attn)
        # 시각화
        vis = visualize_cross_attention(attn_w, args)
        cross_frames.append(vis)

        if (i+1) % 50 == 0:
            print(f"Processed cross-attn {i+1}/{len(frames)-1} frames...")


    # 비디오 저장
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"cross_attn_{datetime.now().strftime('%H%M%S')}.mp4")
    save_video(cross_frames, out_path, fps)
    print(f"Saved cross-attention video to {out_path}")
