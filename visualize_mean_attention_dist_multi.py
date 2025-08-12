import os
import sys
import argparse
from PIL import Image

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# DINO ViT: vision_transformer.py 를 vits 별칭으로 사용
import vision_transformer as vits

# =========================
# 유틸: 거리 행렬 / MAD 계산
# =========================
def compute_distance_matrix(patch_size: int, w_featmap: int, h_featmap: int) -> np.ndarray:
    """
    (w_featmap x h_featmap) 패치 그리드에서 패치 간 유클리드 거리 행렬 (N x N) 생성.
    거리 단위는 픽셀(= patch_size * 그리드 거리).
    """
    N = w_featmap * h_featmap
    xs = np.arange(N) // h_featmap
    ys = np.arange(N) %  h_featmap
    coords = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2)

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1) * float(patch_size)  # (N,N)
    np.fill_diagonal(dists, 0.0)
    return dists


def mean_attention_distance_from_map(attn_map: torch.Tensor,
                                     distance_matrix: np.ndarray) -> np.ndarray:
    """
    단일 어텐션 텐서에서 헤드별 Mean Attention Distance 계산.

    attn_map: (B, H, T, T)  (T=1+N, CLS 포함, 마지막 dim softmax)
    distance_matrix: (N, N) numpy
    return: (B, H) numpy
    """
    attn = attn_map[:, :, 1:, 1:]  # (B,H,N,N)  # CLS 제거
    B, H, N, _ = attn.shape

    D = torch.from_numpy(distance_matrix).to(attn.device).view(1, 1, N, N)  # (1,1,N,N)
    exp_dist_per_token = (attn * D).sum(dim=-1)  # (B,H,N)
    mad = exp_dist_per_token.mean(dim=-1)        # (B,H)
    return mad.detach().cpu().numpy()


# =========================
# 데이터 / 모델 로딩
# =========================
def gather_image_paths(image_dir: str, exts: str):
    """
    디렉토리 내 1-depth 파일을 확장자(대/소문자 무시)로 필터링.
    """
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"{image_dir} 디렉토리를 찾을 수 없습니다.")
    ext_set = {e.strip().lower() for e in exts.split(",") if e.strip()}
    paths = []
    for name in os.listdir(image_dir):
        full = os.path.join(image_dir, name)
        if not os.path.isfile(full):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in ext_set:
            paths.append(full)
    paths.sort()
    if len(paths) == 0:
        raise FileNotFoundError(f"{image_dir} 에서 {sorted(ext_set)} 확장자의 이미지를 찾지 못했습니다.")
    return paths


def build_transform(image_size):
    if isinstance(image_size, (list, tuple)):
        H, W = int(image_size[0]), int(image_size[1])
    else:
        H = W = int(image_size)
    return T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_model(arch: str, patch_size: int, pretrained_weights: str, checkpoint_key: str, device):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key and checkpoint_key in state_dict:
            print(f"checkpoint key '{checkpoint_key}' 사용")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"사전학습 가중치 로드 완료: {pretrained_weights} (msg={msg})")
    else:
        print("--pretrained_weights 를 찾지 못했습니다. DINO 레퍼런스 가중치 시도.")
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        if url is not None:
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
            print("DINO 레퍼런스 가중치 로드 완료.")
        else:
            print("레퍼런스 가중치가 없는 구성입니다. 랜덤 가중치로 진행합니다.")
    return model


# =========================
# 메인 로직
# =========================
@torch.no_grad()
def compute_mad_for_dir(arch: str,
                        patch_size: int,
                        pretrained_weights: str,
                        checkpoint_key: str,
                        image_dir: str,
                        image_size,
                        exts: str,
                        use_last_only: bool,
                        output_dir: str):
    # 출력 디렉토리 구성 (arch와 patch size 반영)
    model_tag = f"{arch}_patch{patch_size}"
    output_dir = os.path.join(output_dir, model_tag)
    os.makedirs(output_dir, exist_ok=True)
    print(f"결과 저장 경로: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(arch, patch_size, pretrained_weights, checkpoint_key, device)
    img_paths = gather_image_paths(image_dir, exts)
    transform = build_transform(image_size)

    # 첫 이미지로 그리드/토큰 수 파악
    sample = Image.open(img_paths[0]).convert("RGB")
    x = transform(sample)  # (C,H,W)
    H = x.shape[1] - (x.shape[1] % patch_size)
    W = x.shape[2] - (x.shape[2] % patch_size)
    x = x[:, :H, :W].unsqueeze(0).to(device)
    w_featmap = H // patch_size
    h_featmap = W // patch_size
    N = w_featmap * h_featmap

    # 헤드/레이어 수 파악
    if use_last_only:
        attn = model.get_last_selfattention(x)  # (1,Hh,1+N,1+N)
        used_layers = [ (len(model.blocks)-1) if hasattr(model, "blocks") else 11 ]
    else:
        attn = model.get_selfattention(x, 0)    # (1,Hh,1+N,1+N)
        used_layers = list(range(len(model.blocks))) if hasattr(model, "blocks") else list(range(12))

    _, num_heads, T, _ = attn.shape
    assert T == 1 + N, f"토큰 수 불일치: {T} != {1+N}"
    print(f"그리드: {w_featmap}x{h_featmap} (N={N}), 헤드: {num_heads}, 블록: {len(used_layers)}")

    # 거리 행렬
    base_D = compute_distance_matrix(patch_size, w_featmap, h_featmap)

    # 레이어별 MAD 누적
    per_layer_mads = {li: [] for li in used_layers}

    for p in tqdm(img_paths, desc="이미지 처리"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            print(f"이미지 로드 실패: {p} (skip)")
            continue

        t = transform(img)
        H = t.shape[1] - (t.shape[1] % patch_size)
        W = t.shape[2] - (t.shape[2] % patch_size)
        t = t[:, :H, :W].unsqueeze(0).to(device)

        w_fm = H // patch_size
        h_fm = W // patch_size
        if (w_fm != w_featmap) or (h_fm != h_featmap):
            D = compute_distance_matrix(patch_size, w_fm, h_fm)
        else:
            D = base_D

        if use_last_only:
            attn_map = model.get_last_selfattention(t)    # (1,Hh,1+N,1+N)
            mad = mean_attention_distance_from_map(attn_map, D)  # (1,Hh)
            per_layer_mads[used_layers[0]].append(mad[0])
        else:
            for li in used_layers:
                attn_map = model.get_selfattention(t, li) # (1,Hh,1+N,1+N)
                mad = mean_attention_distance_from_map(attn_map, D)
                per_layer_mads[li].append(mad[0])

    # 평균 집계
    per_layer_mean = {}
    for li, arr in per_layer_mads.items():
        if len(arr) == 0:
            per_layer_mean[li] = np.full((num_heads,), np.nan, dtype=np.float32)
        else:
            m = np.stack(arr, axis=0)  # (num_imgs, Hh)
            per_layer_mean[li] = m.mean(axis=0)

    # -----------------
    # 시각화 저장
    # -----------------
    # (1) 헤드별 블록 산점도 (탭10 컬러맵, 헤드 번호 주석)
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_heads))  # tab10 컬러맵 사용
    
    for h in range(num_heads):
        x_vals = sorted(used_layers)  # 블록 인덱스
        y_vals = [float(per_layer_mean[li][h]) for li in x_vals]
        
        # 헤드별 산점도 (동일 헤드는 같은 색상)
        plt.scatter(x_vals, y_vals, 
                   color=colors[h], 
                   label=f"head_{h}", 
                   alpha=0.8)
        
        # 각 점에 헤드 번호 주석
        for x, y in zip(x_vals, y_vals):
            plt.annotate(str(h), (x + 0.1, y + 0.1), fontsize=8)
    
    plt.xlabel("Transformer Blocks")
    plt.ylabel("Mean Attention Distance (pixels)")
    plt.title(f"{arch}_patch{patch_size} - Mean Attention Distance per Head")
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc="lower right", 
              fontsize=8,
              ncol=2 if num_heads > 6 else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_tag}_mad_per_head.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

    # (2) 선택된 블록들의 헤드별 MAD (첫/중간/마지막 블록)
    blocks = []
    if len(used_layers) >= 5:
        blocks = [
            used_layers[0],                    # 첫 블록
            used_layers[len(used_layers)//4],  # 1/4 지점
            used_layers[len(used_layers)//2],  # 중간
            used_layers[3*len(used_layers)//4],# 3/4 지점
            used_layers[-1]                    # 마지막 블록
        ]
    else:
        blocks = used_layers  # 블록이 5개 미만이면 전체 사용

    plt.figure(figsize=(10, 10))
    heads = list(range(num_heads))
    for b in blocks:
        y = per_layer_mean[b]
        plt.plot(heads, y, marker='o', label=f"block_{b}")
        # 각 점에 블록 번호 주석
        for h, v in enumerate(y):
            plt.annotate(f"{b}", (heads[h] + 0.1, v + 0.1), fontsize=8)

    plt.xlabel("Head Index")
    plt.ylabel("Mean Attention Distance (pixels)")
    plt.title(f"{arch}_patch{patch_size} - MAD across Heads (selected blocks)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_tag}_mad_selected_blocks.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

    # 원시 값 저장 (TSV)
    tsv = os.path.join(output_dir, f"{model_tag}_mad_values.tsv")
    with open(tsv, "w") as f:
        f.write("block\thead\tmad\n")
        for b in sorted(used_layers):  # xs를 used_layers로 수정
            for h, v in enumerate(per_layer_mean[b]):
                f.write(f"{b}\t{h}\t{float(v):.6f}\n")
    print(f"결과 저장 완료: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser("여러 장의 이미지에 대한 Mean Attention Distance 시각화")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'])
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--image_dir', required=True, type=str, help="이미지들이 들어있는 디렉토리 경로")
    parser.add_argument('--image_size', default=(480, 480), type=int, nargs="+")
    # 소문자 비교이므로 기본값에 소문자만 두면 .JPEG(대문자)도 자동 매칭됩니다.
    parser.add_argument('--exts', default='.jpg,.jpeg,.png,.bmp', type=str)
    parser.add_argument('--use_last_only', action='store_true', help='마지막 레이어만 사용')
    parser.add_argument('--output_dir', default='./dist-output', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compute_mad_for_dir(
        arch=args.arch,
        patch_size=args.patch_size,
        pretrained_weights=args.pretrained_weights,
        checkpoint_key=args.checkpoint_key,
        image_dir=args.image_dir,
        image_size=args.image_size,
        exts=args.exts,
        use_last_only=args.use_last_only,
        output_dir=args.output_dir
    )
