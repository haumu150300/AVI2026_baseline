
import torch
from torch import device
from src.moe.MoeCustom import MoeCustom
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def save_model(model, path):
    torch.save(model.state_dict(), path)

def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: Optimizer, criterion, scheduler, device: device):
    model.train()
    total_loss = 0
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        scaler = torch.amp.GradScaler()
        with torch.amp.autocast(device_type=device.type):
            outputs = model(features['visual'], features['audio'], features['text'])
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model: torch.nn.Module, dataloader: DataLoader, criterion, device: device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features['visual'], features['audio'], features['text'])
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)