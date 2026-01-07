import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import timm

#!/usr/bin/env python3
# File: /home/mas/Bureau/Cours MAS/M2-MAS/S1-Projet/LittleBugTaxonomist/Code/Test_code_ViT.py
# Purpose: Train a FasterViT (fallback to ResNet if timm/FasterViT unavailable)
# Usage example:
#   python Test_code_ViT.py --data_dir /path/to/dataset --output_dir ./outputs --epochs 20


import torch.nn as nn

# optional dependency
try:
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False

def get_args():
    p = argparse.ArgumentParser(description="Train FasterViT (or fallback) on beetle images with taxonomic labels")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dataset directory. Prefer structure: train/ <class>/images & val/ <class>/images. If no train/val subfolders, it will split automatically.")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--model_name", type=str, default="fastervit_t0",
                   help="timm model name for FasterViT. If timm not available or model not found, fallback to resnet50.")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Used when data_dir doesn't have train/ and val/ subfolders.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    return train_tf, val_tf

def prepare_datasets(data_dir, train_tf, val_tf, val_split, seed):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and val_dir.exists():
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    else:
        # use a single folder with subfolders per class
        full_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
        n = len(full_ds)
        val_len = int(n * val_split)
        train_len = n - val_len
        torch.manual_seed(seed)
        train_ds, val_ds = random_split(full_ds, [train_len, val_len])
        # Ensure transforms: replace val dataset transform with val_tf
        val_ds.dataset.transform = val_tf

    classes = train_ds.dataset.classes if isinstance(train_ds, torch.utils.data.Subset) else train_ds.classes
    return train_ds, val_ds, classes

def build_model(num_classes, model_name="fastervit_t0"):
    if HAVE_TIMM:
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f"Using timm model: {model_name}")
            return model
        except Exception:
            print(f"timm available but failed to create {model_name}, falling back to ResNet50.")
    else:
        print("timm not available, falling back to ResNet50.")

    # fallback
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs.detach(), 1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

def save_checkpoint(state, out_dir, name):
    torch.save(state, os.path.join(out_dir, name))

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, val_tf = make_transforms(args.img_size)
    train_ds, val_ds, classes = prepare_datasets(args.data_dir, train_tf, val_tf, args.val_split, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_classes = len(classes)
    model = build_model(num_classes, args.model_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": classes,
            "args": vars(args)
        }
        save_checkpoint(checkpoint, args.output_dir, f"checkpoint_epoch_{epoch}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint, args.output_dir, "best_model.pth")

    # Save final artifacts
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    print("Training finished. Best val acc: {:.4f}. Artifacts saved to: {}".format(best_val_acc, args.output_dir))

if __name__ == "__main__":
    main()