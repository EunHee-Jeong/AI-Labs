import sys
sys.path.append(".") # 현재 디렉토리 모듈 경로에 추가

import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm.auto import tqdm
import torchvision.transforms as transforms

from scripts.dataloader import CustomImageDataset
from scripts.model import BaseModel
from torch_ema import ExponentialMovingAverage

import json
from sklearn.metrics import f1_score
import numpy as np

# 로그 저장
def save_log(metrics_dict, save_dir="./logs"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"log_{metrics_dict['id']}.json")
    with open(log_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✅ 로그 저장 완료: {log_path}")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='resnet18')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
args = parser.parse_args()

# Config
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': args.batch_size, # 64
    'EPOCHS': args.epochs, # 10
    'LEARNING_RATE': args.lr, # 3e-4
    'SEED': 42
}

# seed 고정
def seed_everything(seed=42):
    import random, os, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 경로
# TODO:- 수정하기
train_root = './data/train'
MODEL_SAVE_PATH = './models/best_model.pth'
os.makedirs('./models', exist_ok=True)

# 저장할 모델 경로 구성
now = datetime.now().strftime("%Y%m%d_%H%M")
model_filename = f"{args.model_name}_e{args.epochs}_b{args.batch_size}_lr{args.lr}_{now}.pth"
MODEL_SAVE_PATH = os.path.join('./model_weights', model_filename) # TODO:- 수정하기

# Transform 정의
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transform = train_transform

# 전체 데이터셋 로드 + 클래스 추출
full_dataset = CustomImageDataset(train_root, transform=None)
class_names = full_dataset.classes
targets = [label for _, label in full_dataset.samples]

# Stratified Split
train_idx, val_idx = train_test_split(
    range(len(targets)), test_size=0.2, stratify=targets, random_state=CFG['SEED']
)

train_dataset = Subset(CustomImageDataset(train_root, transform=train_transform), train_idx)
val_dataset = Subset(CustomImageDataset(train_root, transform=val_transform), val_idx)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# 모델 준비
model = BaseModel(model_name=args.model_name, num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.0) # softmax 확률분포를 더 sharp하게 만듦
optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=CFG['LEARNING_RATE'],
                       steps_per_epoch=len(train_loader), epochs=CFG['EPOCHS'])
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
scaler = GradScaler()

best_logloss = float('inf')

patience = args.patience
early_stop_counter = 0

# 학습 루프
for epoch in range(CFG['EPOCHS']):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad(), autocast(), ema.average_parameters():
        for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))
    macro_f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='macro')

    print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.2f}%")

    if val_logloss < best_logloss:
        best_logloss = val_logloss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"📦 Best model saved: {MODEL_SAVE_PATH} (logloss: {val_logloss:.4f})")
        early_stop_counter = 0  # 성능 좋아졌으므로 초기화

        # 로그 저장
        metrics = {
            "id": now,
            "model": args.model_name,
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "valid_loss": round(avg_val_loss, 4),
            "log_loss": round(val_logloss, 4),
            "top1_acc": round(val_accuracy, 2),
            "macro_f1": round(macro_f1, 4),
            "submission": f"submission_{now}.csv",
            "path": MODEL_SAVE_PATH
        }
        save_log(metrics)

    else:
        early_stop_counter += 1
        print(f"⚠️ EarlyStopping counter: {early_stop_counter} / {patience}")
        if early_stop_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

# 사용예시
# python scripts/train.py --model_name resnet18 --epochs 12 --lr 0.0002 --batch_size 32