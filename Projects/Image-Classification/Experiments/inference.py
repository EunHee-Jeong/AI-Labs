import sys
sys.path.append(".")  # 현재 디렉토리를 모듈 검색 경로에 추가

import argparse
import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from scripts.dataloader import CustomImageDataset
from scripts.model import BaseModel

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='resnet18', help='모델 이름 (출력 파일명용)')
parser.add_argument('--model_path', type=str, required=True, help='저장된 모델 경로 (.pth)')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

# Config
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': args.batch_size
}

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 경로
# TODO:- 수정하기
train_root = './data/train'
test_root = './data/test'

# Transform
val_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 클래스 이름 추출
full_dataset = CustomImageDataset(train_root, transform=None)
class_names = full_dataset.classes

# 테스트 데이터 로드
test_dataset = CustomImageDataset(test_root, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# 모델 로딩
model = BaseModel(model_name=args.model_name, num_classes=len(class_names))
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# 추론
results = []
with torch.no_grad(), torch.cuda.amp.autocast():
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        for prob in probs.cpu():
            result = {class_names[i]: prob[i].item() for i in range(len(class_names))}
            results.append(result)

# 결과 저장
pred = pd.DataFrame(results)
submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig') # TODO:- 수정하기
class_columns = submission.columns[1:]
pred = pred[class_columns]
submission[class_columns] = pred.values

# 자동 파일명 생성
now = datetime.now().strftime("%Y%m%d_%H%M")
submission_name = f"submission_{args.model_name}_{now}.csv"
submission.to_csv(f"./submission/{submission_name}", index=False, encoding='utf-8-sig') # TODO:- 수정하기

print(f"✅ Inference finished. Submission file saved as '{submission_name}'")


# 사용 예시
"""
python scripts/inference.py \
  --model_name resnet18 \
  --model_path ./model_weights/efficientnet_b0_e10_b64_lr0.0003_20250601_0420.pth \
  --batch_size 64
"""