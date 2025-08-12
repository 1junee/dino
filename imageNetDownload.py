import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 데이터셋 저장 경로 설정
data_dir = './imagenet_data'

# # ImageNet 데이터셋 다운로드 (훈련 데이터셋)
# train_dataset = datasets.ImageNet(
#     root=data_dir,  # 데이터 저장 경로
#     split='train',  # 'train' 또는 'val' (검증 데이터셋)
#     download=True,  # 데이터가 없으면 다운로드
#     transform=transforms.ToTensor()  # 이미지 텐서로 변환
# )

# 검증 데이터셋
val_dataset = datasets.ImageNet(
    root=data_dir,
    split='val',
    download=True,
    # transform=transforms.ToTensor()
)