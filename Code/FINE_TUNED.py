import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip
import json
import time 
import copy
from sklearn.model_selection import train_test_split
import multiprocessing

# --- CONFIGURATION ---
DATA_DIR = os.path.join('Data', 'data_new_few_shot') 
METADATA_PATH = os.path.join('Data', 'metadata_images.json')

OUTPUT_DIR = "Data"
TIMING_FILE = os.path.join(OUTPUT_DIR, "training_summary_finetuning_optimized.json")

# Hyperparamètres
SEED = 123
EPOCHS_WARMUP = 4       # Phase 1 : Chauffe du classifier
EPOCHS_FINETUNE = 4     # Phase 2 : Ajustement fin du modèle complet

# Learning Rates
LR_WARMUP = 1e-3        # Rapide pour la tête
LR_BACKBONE = 1e-6      # Très lent pour BioClip (chirurgical)
LR_HEAD_FT = 1e-5       # Lent pour la tête en phase 2

# Optimisation Mémoire & Vitesse
PHYSICAL_BATCH_SIZE = 8       # Ce qui rentre réellement dans la VRAM (6GB friendly)
GRADIENT_ACCUMULATION = 4     # On accumule pour simuler un batch de 32 (8*4)
HIDDEN_DIM = 512        

# Optimisation CPU
# On utilise les coeurs CPU pour charger les images en parallèle du GPU
NUM_WORKERS = min(8, os.cpu_count()) 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use of {device} with {NUM_WORKERS} CPU workers for data loading.")

# --- 1. CHARGEMENT MÉTADONNÉES ---
print("Chargement taxonomie...")
TAXONOMY_MAP = {}
try:
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        meta_list = json.load(f)
        for entry in meta_list:
            gbif = entry.get('gbif')
            if not gbif: continue
            species = gbif.get('species')
            if species:
                clean_name = species.replace(" ", "_").replace(".", "")
                TAXONOMY_MAP[clean_name] = {
                    "family": gbif.get("family"),
                    "folder_number": entry.get("folder_number")
                }
except FileNotFoundError:
    print("Metadata introuvable.")

# --- 2. DATASET & MODÈLE ---

class LazyBugDataset(Dataset):
    def __init__(self, file_paths, labels, preprocess_fn):
        self.file_paths = file_paths
        self.labels = labels
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.preprocess(image)
            return tensor, label, path 
        except Exception:
            return torch.zeros((3, 224, 224)), label, path

class BioClipMLP(nn.Module):
    def __init__(self, bioclip_model, num_classes, hidden_dim=512):
        super(BioClipMLP, self).__init__()
        self.visual = bioclip_model.visual 
        
        # Le Classifier demandé : 2 couches cachées ReLU
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout un peu plus fort pour éviter l'overfitting
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        features = self.visual(x) 
        # On normalise souvent les features CLIP avant classification
        features = features / features.norm(dim=-1, keepdim=True)
        return self.classifier(features)
    
    def freeze_backbone(self):
        print("-> Freezing BioClip backbone")
        for param in self.visual.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        print("-> Unfreezing BioClip backbone (Deep Training enabled)")
        for param in self.visual.parameters():
            param.requires_grad = True

# --- 3. PRÉPARATION DONNÉES ---
def prepare_data_split(root_dir, preprocess_fn):
    random.seed(SEED)
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_paths = []
    all_labels = []
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        cls_idx = class_to_idx[cls_name]
        for f in files:
            all_paths.append(os.path.join(cls_path, f))
            all_labels.append(cls_idx)
            
    # Stratify split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        all_paths, all_labels, test_size=0.20, random_state=SEED, stratify=all_labels
    )
    
    print(f"Data Split: {len(X_train)} Train / {len(X_test)} Test")

    train_ds = LazyBugDataset(X_train, y_train, preprocess_fn)
    test_ds = LazyBugDataset(X_test, y_test, preprocess_fn)
    
    return train_ds, test_ds, classes

# --- 4. FONCTION D'ENTRAINEMENT OPTIMISÉE ---
def run_training_phase(model, loader, criterion, optimizer, scaler, epochs, phase_name):
    print(f"\n--- STARTING {phase_name} ({epochs} Epochs) ---")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        # Loader itère en utilisant le CPU en parallèle
        for i, (batch_imgs, batch_lbls, _) in enumerate(loader):
            # non_blocking=True permet de transférer pendant que le GPU calcule autre chose
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_lbls = batch_lbls.to(device, non_blocking=True)
            
            # --- MIXED PRECISION (AMP) ---
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_lbls)
                # Normalisation pour l'accumulation de gradients
                loss = loss / GRADIENT_ACCUMULATION 
            
            # Backprop avec scaling (pour éviter underflow float16)
            scaler.scale(loss).backward()
            
            # --- GRADIENT ACCUMULATION ---
            # On met à jour les poids uniquement tous les X batchs
            if (i + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Stats (on remultiplie la loss pour l'affichage correct)
            total_loss += loss.item() * GRADIENT_ACCUMULATION
            _, predicted = outputs.max(1)
            total += batch_lbls.size(0)
            correct += predicted.eq(batch_lbls).sum().item()
        
        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total
        print(f"[{phase_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

# --- 5. EVALUATION ---
def evaluate_and_generate_json(model, loader, class_names):
    model.eval()
    session_results = []
    print("\nStarting evaluation on Test set...")
    
    with torch.no_grad():
        for batch_imgs, batch_lbls, batch_paths in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            
            # Même en inférence, on utilise AMP pour aller plus vite
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                outputs = model(batch_imgs)
                probs = F.softmax(outputs, dim=1)
            
            top5_probs, top5_indices = probs.topk(5, dim=1)
            
            # Transfert CPU une seule fois par batch
            top5_probs = top5_probs.cpu()
            top5_indices = top5_indices.cpu()
            batch_lbls = batch_lbls.cpu()
            
            for i in range(len(batch_imgs)):
                path = batch_paths[i]
                real_label_idx = batch_lbls[i].item()
                real_name = class_names[real_label_idx]
                info_real = TAXONOMY_MAP.get(real_name, {})
                
                top5_list = []
                for k in range(5):
                    pred_idx = top5_indices[i][k].item()
                    prob = top5_probs[i][k].item()
                    pred_name = class_names[pred_idx]
                    info_pred = TAXONOMY_MAP.get(pred_name, {})
                    
                    top5_list.append({
                        "label": pred_name,
                        "prob": prob,
                        "genus": pred_name.split('_')[0],
                        "family": info_pred.get("family")
                    })
                
                session_results.append({
                    "archive_path": path,
                    "folder_number": info_real.get("folder_number"),
                    "real_name": real_name,
                    "real_genus": real_name.split('_')[0],
                    "real_family": info_real.get("family"),
                    "top5": top5_list
                })
                
    return session_results

# --- MAIN ---
if __name__ == "__main__":
    # Optimisation Windows/Linux pour DataLoader
    torch.backends.cudnn.benchmark = True # Accélère si la taille des images est constante
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_timings = []
    current_timing = {"experiment": "TwoStage_FineTuning_Optimized"}
    
    t_global_start = time.time()
    
    # 1. Loading
    print("Loading BioClip backbone...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    
    # 2. Data Preparation
    train_ds, test_ds, class_names = prepare_data_split(DATA_DIR, preprocess)
    
    # OPTIMISATION CPU : num_workers > 0 et pin_memory=True
    train_loader = DataLoader(
        train_ds, 
        batch_size=PHYSICAL_BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True # Garde les workers vivants entre les epochs (gain de temps)
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=PHYSICAL_BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # 3. Init Model
    full_model = BioClipMLP(clip_model, len(class_names), hidden_dim=HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))
    
    # ==========================================
    # PHASE 1 : WARMUP (Classifier Only)
    # ==========================================
    full_model.freeze_backbone() 
    
    optimizer_warmup = optim.AdamW(full_model.classifier.parameters(), lr=LR_WARMUP)
    
    t_p1_start = time.time()
    run_training_phase(full_model, train_loader, criterion, optimizer_warmup, scaler, EPOCHS_WARMUP, "PHASE 1 - WARMUP")
    current_timing["phase1_seconds"] = round(time.time() - t_p1_start, 4)
    
    # ==========================================
    # PHASE 2 : FULL FINE-TUNING (Backbone + Head)
    # ==========================================
    full_model.unfreeze_backbone() 
    
    param_groups = [
        {'params': full_model.visual.parameters(), 'lr': LR_BACKBONE},
        {'params': full_model.classifier.parameters(), 'lr': LR_HEAD_FT}
    ]
    optimizer_finetune = optim.AdamW(param_groups)
    
    t_p2_start = time.time()
    run_training_phase(full_model, train_loader, criterion, optimizer_finetune, scaler, EPOCHS_FINETUNE, "PHASE 2 - DEEP FINETUNE")
    current_timing["phase2_seconds"] = round(time.time() - t_p2_start, 4)
    
    # 5. Eval
    results_json = evaluate_and_generate_json(full_model, test_loader, class_names)
    
    output_json_path = os.path.join(OUTPUT_DIR, "predictions_finetuned_testset.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
        
    print(f"Done! Results saved to {output_json_path}")
    
    # Timings
    current_timing["total_seconds"] = round(time.time() - t_global_start, 4)
    all_timings.append(current_timing)
    with open(TIMING_FILE, "w", encoding="utf-8") as f:
        json.dump(all_timings, f, indent=2)