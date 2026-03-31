from random import random
from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader
from data.BaseDataset import BaseDataset, load_features_and_labels_task1, TASK1_QS
import tdqm
import numpy as np
from utils import save_model, train_one_epoch, evaluate
from src.moe.MoeCustom import MoeCustom
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况

    # 保证 cuDNN 可复现（会略微降低速度，但 baseline 完全可以接受）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8
    torch.use_deterministic_algorithms(True)

# 设置全局随机种子
set_seed(42)
TRAIN_FEATURE_DIR = "/home/orisu/avi2026/dataset/autodl-tmp/train_feature_token"
TRAIN_BASE_DIR = "/home/orisu/avi2026/dataset/train_data"
TRAIN_LABEL_FILE = "/home/orisu/avi2026/dataset/train_data.csv"

VAL_FEATURE_DIR = "/home/orisu/avi2026/dataset/autodl-tmp/val_feature_token"
VAL_LABEL_FILE = "/home/orisu/avi2026/dataset/val_data.csv"
VAL_BASE_DIR = "/home/orisu/avi2026/dataset/val_data"

train_data, train_labels = load_features_and_labels_task1(TRAIN_FEATURE_DIR, TRAIN_BASE_DIR, TRAIN_LABEL_FILE)
val_data, val_labels = load_features_and_labels_task1(VAL_FEATURE_DIR, VAL_BASE_DIR, VAL_LABEL_FILE)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
EPOCHS = 100

for idx, q in enumerate(TASK1_QS):
    train_dataset = BaseDataset(train_data[q], train_labels[q])
    val_dataset = BaseDataset(val_data[q], val_labels[q])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MoeCustom().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    logger = SummaryWriter(log_dir=f"./logs/task1_{q}_moe")

    for epoch in tdqm(range(EPOCHS), desc=f"Processing"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}")
        logger.add_scalar("Loss/Train", train_loss, epoch)
        logger.add_scalar("Loss/Val", val_loss, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f"./trained_models/task1_{q}.pth")