import os
import sys
import json
import tarfile
import tempfile
import argparse
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# File: Test_vit_comparison.py
# Purpose: fine-tune CLIP ViT-L/14 vision tower as an image classifier for an insect dataset
# Usage example:
#   python Test_vit_comparison.py --data /path/to/database.tar --metadata /path/to/metadata_images.json --out_dir ./output --epochs 5


# try to ensure required packages are present
required = ("torch", "transformers", "torchvision", "Pillow", "tqdm", "scikit-learn")
for pkg in required:
    try:
        __import__(pkg if pkg != "Pillow" else "PIL")
    except Exception:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import torch.nn as nn

def extract_tar_if_needed(path):
    if os.path.isdir(path):
        return path
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tmp = tempfile.mkdtemp(prefix="insect_data_")
    print(f"Extracting {path} -> {tmp}")
    with tarfile.open(path, "r:*") as tar:
        tar.extractall(path=tmp)
    return tmp

def infer_image_and_label_from_metadata(meta, base_dir):
    """
    Build list of (image_path, label_str) from metadata content.
    Supports common metadata formats:
    - list of dicts with keys like 'image_path','file_name','file','path' and
      label keys like 'scientific_name','species','label','class','taxon'
    - dict mapping image filename -> metadata dict
    """
    items = []
    def get_image_key(d):
        for k in ("image_path", "file_name", "file", "path", "image"):
            if k in d:
                return d[k]
        # fallback: look for a string value ending with an image extension
        for v in d.values():
            if isinstance(v, str) and v.lower().endswith((".jpg",".jpeg",".png")):
                return v
        return None
    def get_label_key(d):
        for k in ("scientific_name","species","label","class","taxon","category"):
            if k in d:
                return d[k]
        # fallback: any small string field
        for k,v in d.items():
            if isinstance(v,str) and len(v) < 100 and not v.lower().endswith((".jpg",".png")):
                return v
        return None

    if isinstance(meta, dict):
        # either mapping or single metadata
        # if values are dicts, treat as mapping
        if all(isinstance(v, dict) for v in meta.values()):
            for k,v in meta.items():
                img = get_image_key(v) or k
                lbl = get_label_key(v)
                items.append((os.path.join(base_dir, img) if not os.path.isabs(img) else img, lbl))
        else:
            # not mapping of dicts: try to parse single entry
            img = get_image_key(meta)
            lbl = get_label_key(meta)
            items.append((os.path.join(base_dir, img) if not os.path.isabs(img) else img, lbl))
    elif isinstance(meta, list):
        for entry in meta:
            if isinstance(entry, dict):
                img = get_image_key(entry)
                lbl = get_label_key(entry)
                if img is None:
                    continue
                items.append((os.path.join(base_dir, img) if not os.path.isabs(img) else img, lbl))
            elif isinstance(entry, str):
                # plain list of paths -> label not present
                items.append((os.path.join(base_dir, entry) if not os.path.isabs(entry) else entry, None))
    else:
        raise ValueError("Unsupported metadata format")
    # normalize and drop missing images
    normalized = []
    for p,l in items:
        if not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        if os.path.exists(p):
            normalized.append((p, l))
        else:
            # try relative to base_dir top-level
            p2 = os.path.join(base_dir, os.path.basename(p))
            if os.path.exists(p2):
                normalized.append((p2, l))
    return normalized

class InsectDataset(Dataset):
    def __init__(self, samples, processor, transform=None):
        self.samples = samples  # list of (image_path, label_idx)
        self.processor = processor
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, label

def collate_batch(batch, processor, device):
    images, labels = zip(*batch)
    # processor returns a torch tensor when return_tensors="pt"
    proc = processor(images=list(images), return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return pixel_values, labels

class ViTClassifier(nn.Module):
    def __init__(self, clip_model_name, num_classes, freeze_clip=False):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        hidden = self.clip.config.vision_hidden_size if hasattr(self.clip.config, "vision_hidden_size") else self.clip.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
    def forward(self, pixel_values):
        # feed through vision tower only
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        # try pooler_output, else use first token
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            feat = vision_outputs.pooler_output
        else:
            feat = vision_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(feat)
        return logits

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_dir = extract_tar_if_needed(args.data) if args.data else args.data_dir
    if not os.path.isdir(data_dir):
        # if data_dir not provided explicitly, try default Data path nearby script
        raise FileNotFoundError(data_dir)
    # load metadata
    with open(args.metadata, "r") as f:
        meta = json.load(f)
    samples = infer_image_and_label_from_metadata(meta, data_dir)
    if len(samples) == 0:
        # try scanning data_dir for images and use parent folder name as label
        print("No samples from metadata - scanning folder tree to build dataset.")
        samples = []
        for root, dirs, files in os.walk(data_dir):
            for fn in files:
                if fn.lower().endswith((".jpg",".jpeg",".png")):
                    label = os.path.basename(root)
                    samples.append((os.path.join(root, fn), label))
    labels = [lbl for (_, lbl) in samples]
    # build mapping
    unique = sorted(set(labels))
    label2idx = {l:i for i,l in enumerate(unique)}
    idx_samples = [(p, label2idx[l]) for (p,l) in samples]
    train_samples, val_samples = train_test_split(idx_samples, test_size=args.val_split, stratify=[l for (_,l) in idx_samples], random_state=42)
    print(f"Samples: total={len(idx_samples)} train={len(train_samples)} val={len(val_samples)} classes={len(unique)}")

    # model and processor
    clip_name = args.clip_name
    processor = CLIPProcessor.from_pretrained(clip_name)
    model = ViTClassifier(clip_name, num_classes=len(unique), freeze_clip=args.freeze_clip).to(device)

    # dataloaders
    train_ds = InsectDataset(train_samples, processor)
    val_ds = InsectDataset(val_samples, processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=lambda b: collate_batch(b, processor, device))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=lambda b: collate_batch(b, processor, device))

    # optimizer and loss
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")
        for pixel_values, labels in pbar:
            opt.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * labels.size(0)
            pbar.set_postfix(loss=running_loss / ((pbar.n+1) * args.batch_size))
        # validation
        model.eval()
        total = 0
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for pixel_values, labels in tqdm(val_loader, desc="Validation"):
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        val_loss = val_loss / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f} val_acc={acc:.4f}")
        # save best
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "label2idx": label2idx,
                "clip_name": clip_name,
            }, save_path)
            print(f"Saved best model ({best_acc:.4f}) -> {save_path}")
    # final save
    torch.save({
        "model_state": model.state_dict(),
        "label2idx": label2idx,
        "clip_name": clip_name,
    }, os.path.join(args.out_dir, "final_model.pt"))
    print("Training finished.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/home/mas/Bureau/Cours MAS/M2-MAS/S1-Projet/LittleBugTaxonomist/Data/database.tar",
                   help="path to dataset tar or directory")
    p.add_argument("--data_dir", type=str, default="/home/mas/Bureau/Cours MAS/M2-MAS/S1-Projet/LittleBugTaxonomist/Data",
                   help="path to dataset directory (used if --data not a tar)")
    p.add_argument("--metadata", type=str, default="/home/mas/Bureau/Cours MAS/M2-MAS/S1-Projet/LittleBugTaxonomist/Data/metadata_images.json",
                   help="path to metadata json")
    p.add_argument("--out_dir", type=str, default="./output", help="where to save models")
    p.add_argument("--clip_name", type=str, default="openai/clip-vit-large-patch14", help="CLIP model name")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--freeze_clip", action="store_true", help="freeze CLIP weights and only train classifier")
    p.add_argument("--no_cuda", action="store_true", help="disable CUDA")
    args = p.parse_args()
    main(args)