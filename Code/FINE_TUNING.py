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
from tqdm import tqdm

# --- CONFIGURATION ---
# Dossier contenant les images DÉJÀ redimensionnées (via le script précédent)
DATA_DIR = os.path.join('Data', 'data_resized_224') 

# Fichier de métadonnées GBIF
METADATA_PATH = os.path.join('Data', 'metadata_images.json')

OUTPUT_DIR = "Data"
TIMING_FILE = os.path.join(OUTPUT_DIR, "training_summary_finetuning.json")
DETAILED_TIMING_FILE = os.path.join(OUTPUT_DIR, "detailed_timings_log.json")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_latest.pth")

# Hyperparamètres
SEED = 123
EPOCHS_WARMUP = 4       # Phase 1 : Chauffe du classifieur
EPOCHS_FINETUNE = 4     # Phase 2 : Ajustement fin du modèle complet

# Learning rates
LR_WARMUP = 1e-3        # Rapide pour la tête
LR_BACKBONE = 1e-6      # Très lent pour ne pas détruire BioClip
LR_HEAD_FT = 1e-5       # Lent pour la tête en phase 2

# Optimisation Hardware
PHYSICAL_BATCH_SIZE = 16 # Batch size augmenté car images légères (mettre 8 si erreur mémoire)
GRADIENT_ACCUMULATION = 2 # Simulation d'un batch de 32 (16 * 2)
HIDDEN_DIM = 512        
NUM_WORKERS = min(8, os.cpu_count()) 
PREFETCH_FACTOR = 2     # Prépare 2 batchs d'avance pour le GPU

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware: {device} | Workers: {NUM_WORKERS} | Batch Size: {PHYSICAL_BATCH_SIZE}")

# --- DATASET & MODEL ---

class PreSizedBugDataset(Dataset):
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
            # L'image est déjà redimensionnée. Preprocess très rapide.
            image = Image.open(path).convert("RGB")
            tensor = self.preprocess(image)
            return tensor, label, path 
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Retourne une image noire en cas d'erreur pour ne pas planter
            return torch.zeros((3, 224, 224)), label, path

class BioClipMLP(nn.Module):
    def __init__(self, bioclip_model, num_classes, hidden_dim=512):
        super(BioClipMLP, self).__init__()
        self.visual = bioclip_model.visual 
        
        # Tête de classification personnalisée
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        features = self.visual(x) 
        features = features / features.norm(dim=-1, keepdim=True)
        return self.classifier(features)
    
    def freeze_backbone(self):
        print("-> Backbone FROZEN (Training classifier only)")
        for param in self.visual.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        print("-> Backbone UNFROZEN (Training full model)")
        for param in self.visual.parameters():
            param.requires_grad = True

def load_taxonomy_from_gbif(metadata_path):
    """
    Charge et structure les données du fichier JSON GBIF fourni.
    Crée un dictionnaire: "Genus_species" -> {Info complètes}
    """
    print(f"-> Lecture des métadonnées : {metadata_path}...")
    mapping = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_list = json.load(f)
            
            for entry in meta_list:
                # Structure spécifique de ton fichier
                gbif = entry.get('gbif')
                if not gbif: continue
                
                species_name = gbif.get('species') # Ex: "Amara lunicollis"
                
                if species_name:
                    # Conversion "Amara lunicollis" -> "Amara_lunicollis" pour matcher les dossiers
                    clean_name = species_name.replace(" ", "_").replace(".", "")
                    
                    if clean_name not in mapping:
                        mapping[clean_name] = {
                            "order": gbif.get("order"),       
                            "family": gbif.get("family"),     
                            "genus": gbif.get("genus"),
                            "species": species_name,
                            "scientific_name": gbif.get("scientificName"),
                            "id": gbif.get("key")
                        }
    except FileNotFoundError:
        print("ERREUR : Fichier metadata_images.json introuvable dans Data/ !")
        return {}
    except json.JSONDecodeError:
        print("ERREUR : Le fichier metadata n'est pas un JSON valide.")
        return {}
        
    print(f"-> Taxonomie chargée pour {len(mapping)} espèces.")
    return mapping

def prepare_data_split(root_dir, preprocess_fn):
    random.seed(SEED)
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Le dossier {root_dir} n'existe pas ! Avez-vous lancé le script de redimensionnement ?")

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_paths, all_labels = [], []
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        cls_idx = class_to_idx[cls_name]
        for f in files:
            all_paths.append(os.path.join(cls_path, f))
            all_labels.append(cls_idx)
            
    # Stratification pour l'équilibre des classes
    X_train, X_test, y_train, y_test = train_test_split(
        all_paths, all_labels, test_size=0.20, random_state=SEED, stratify=all_labels
    )
    print(f"Dataset prêt : {len(X_train)} Train / {len(X_test)} Test ({len(classes)} classes)")
    
    train_ds = PreSizedBugDataset(X_train, y_train, preprocess_fn)
    test_ds = PreSizedBugDataset(X_test, y_test, preprocess_fn)
    return train_ds, test_ds, classes

# --- CHECKPOINT & TRAINING UTILS ---

def save_checkpoint(state, filename=CHECKPOINT_PATH):
    # Sauvegarde temporaire puis renommage (sécurité anti-corruption)
    torch.save(state, filename + ".tmp")
    os.replace(filename + ".tmp", filename)

def run_training_phase(model, loader, criterion, optimizer, scaler, total_epochs, phase_name, 
                       start_epoch=0, stats_history=[]):
    
    print(f"\n=== {phase_name} (Epochs {start_epoch+1} -> {total_epochs}) ===")
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad(set_to_none=True) # Optimisation RAM
        
        t_cycle_start = time.time()
        
        # Barre de chargement TQDM
        pbar = tqdm(enumerate(loader), total=len(loader), 
                    desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)
        
        for i, (batch_imgs, batch_lbls, _) in pbar:
            # Stats IO
            t_data_loaded = time.time()
            loading_seconds = t_data_loaded - t_cycle_start
            
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_lbls = batch_lbls.to(device, non_blocking=True)
            
            # Mixed Precision (AMP)
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_lbls)
                loss = loss / GRADIENT_ACCUMULATION 
            
            scaler.scale(loss).backward()
            
            # Gradient Accumulation Step
            if (i + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if device == "cuda": torch.cuda.synchronize()
            
            # Stats Inférence
            t_inference_done = time.time()
            inference_seconds = t_inference_done - t_data_loaded
            
            # Calcul Accuracy live
            loss_val = loss.item() * GRADIENT_ACCUMULATION
            total_loss += loss_val
            _, predicted = outputs.max(1)
            total += batch_lbls.size(0)
            correct += predicted.eq(batch_lbls).sum().item()
            
            # Logging
            stats_history.append({
                "phase": phase_name,
                "epoch": epoch + 1,
                "batch": i,
                "loading_seconds": loading_seconds,
                "inference_seconds": inference_seconds,
                "total_cycle_seconds": time.time() - t_cycle_start
            })
            
            t_cycle_start = time.time()
            
            # Mise à jour affichage
            current_acc = 100. * correct / total
            pbar.set_postfix({"Loss": f"{loss_val:.3f}", "Acc": f"{current_acc:.1f}%"})
        
        # --- FIN DE L'EPOCH ---
        avg_loss = total_loss / len(loader)
        final_acc = 100. * correct / total
        print(f"-> End Epoch {epoch+1}: Avg Loss: {avg_loss:.4f} | Acc: {final_acc:.2f}%")
        
        # Sauvegarde JSON Stats
        with open(DETAILED_TIMING_FILE, "w", encoding="utf-8") as f:
            json.dump(stats_history, f, indent=2)
            
        # Création Checkpoint
        checkpoint_state = {
            'phase': phase_name,
            'epoch': epoch + 1, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'stats_history': stats_history
        }
        save_checkpoint(checkpoint_state)

    return stats_history

def evaluate_model(model, loader, class_names, taxonomy_map, stats_history):
    model.eval()
    session_results = []
    print("\n=== EVALUATION FINALE ===")
    
    t_start_eval = time.time()
    pbar = tqdm(loader, desc="Testing")
    
    with torch.no_grad():
        for batch_imgs, batch_lbls, batch_paths in pbar:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                outputs = model(batch_imgs)
                probs = F.softmax(outputs, dim=1)
            
            if device == "cuda": torch.cuda.synchronize()
            
            top5_probs, top5_indices = probs.topk(5, dim=1)
            
            # Transfert CPU pour le traitement JSON
            top5_probs = top5_probs.cpu()
            top5_indices = top5_indices.cpu()
            batch_lbls = batch_lbls.cpu()
            
            for i in range(len(batch_imgs)):
                path = batch_paths[i]
                
                # Vérité terrain
                real_label_idx = batch_lbls[i].item()
                real_name = class_names[real_label_idx]
                real_info = taxonomy_map.get(real_name, {})
                
                # Prédictions
                top5_list = []
                for k in range(5):
                    pred_name = class_names[top5_indices[i][k].item()]
                    # On récupère les infos riches (Genre, Famille...)
                    info_pred = taxonomy_map.get(pred_name, {})
                    
                    top5_list.append({
                        "label": pred_name,
                        "prob": round(top5_probs[i][k].item(), 4),
                        "genus": info_pred.get("genus", "Unknown"),
                        "family": info_pred.get("family", "Unknown")
                    })
                
                session_results.append({
                    "path": path,
                    "real_label": real_name,
                    "real_taxonomy": {
                        "genus": real_info.get("genus"),
                        "family": real_info.get("family")
                    },
                    "predictions": top5_list
                })
    
    stats_history.append({
        "phase": "EVALUATION",
        "epoch": 0, "batch": 0,
        "total_cycle_seconds": time.time() - t_start_eval
    })
    
    return session_results

# --- MAIN ---
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Chargement & Préparation
    TAXONOMY_MAP = load_taxonomy_from_gbif(METADATA_PATH)
    
    print("Loading BioClip Backbone...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    
    train_ds, test_ds, class_names = prepare_data_split(DATA_DIR, preprocess)
    
    # Dataloaders optimisés
    train_loader = DataLoader(
        train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR
    )
    test_loader = DataLoader(
        test_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Init Modèle
    full_model = BioClipMLP(clip_model, len(class_names), hidden_dim=HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))
    
    ALL_DETAILED_STATS = []
    
    # 2. Gestion Reprise (Checkpoint)
    start_phase = "WARMUP"
    start_epoch = 0
    
    if os.path.exists(CHECKPOINT_PATH):
        try:
            chk = torch.load(CHECKPOINT_PATH, map_location=device)
            print(f"\n>>> CHECKPOINT RESTORED: Phase {chk['phase']}, Last Epoch {chk['epoch']-1} <<<")
            
            full_model.load_state_dict(chk['model_state_dict'])
            scaler.load_state_dict(chk['scaler_state_dict'])
            ALL_DETAILED_STATS = chk.get('stats_history', [])
            
            # Logique pour déterminer où reprendre
            if chk['phase'] == "WARMUP":
                if chk['epoch'] >= EPOCHS_WARMUP:
                    start_phase = "FINETUNE"
                    start_epoch = 0
                else:
                    start_phase = "WARMUP"
                    start_epoch = chk['epoch']
                    
            elif chk['phase'] == "FINETUNE":
                start_phase = "FINETUNE"
                start_epoch = chk['epoch']
                
        except Exception as e:
            print(f"Erreur chargement checkpoint (ignoré): {e}")

    # 3. Exécution Phase 1 (Warmup)
    if start_phase == "WARMUP":
        full_model.freeze_backbone()
        optimizer = optim.AdamW(full_model.classifier.parameters(), lr=LR_WARMUP)
        
        # Reprise optimizer si checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            chk = torch.load(CHECKPOINT_PATH, map_location=device)
            if chk['phase'] == "WARMUP" and 'optimizer_state_dict' in chk:
                optimizer.load_state_dict(chk['optimizer_state_dict'])

        ALL_DETAILED_STATS = run_training_phase(
            full_model, train_loader, criterion, optimizer, scaler, 
            EPOCHS_WARMUP, "WARMUP", start_epoch=start_epoch, stats_history=ALL_DETAILED_STATS
        )
        # Transition
        start_phase = "FINETUNE"
        start_epoch = 0 

    # 4. Exécution Phase 2 (Fine-tuning)
    if start_phase == "FINETUNE":
        if start_epoch < EPOCHS_FINETUNE:
            full_model.unfreeze_backbone()
            
            # Param groups différents pour backbone et tete
            param_groups = [
                {'params': full_model.visual.parameters(), 'lr': LR_BACKBONE},
                {'params': full_model.classifier.parameters(), 'lr': LR_HEAD_FT}
            ]
            optimizer = optim.AdamW(param_groups)
            
            # Reprise optimizer si checkpoint
            if os.path.exists(CHECKPOINT_PATH):
                chk = torch.load(CHECKPOINT_PATH, map_location=device)
                if chk['phase'] == "FINETUNE" and 'optimizer_state_dict' in chk:
                    optimizer.load_state_dict(chk['optimizer_state_dict'])

            ALL_DETAILED_STATS = run_training_phase(
                full_model, train_loader, criterion, optimizer, scaler, 
                EPOCHS_FINETUNE, "FINETUNE", start_epoch=start_epoch, stats_history=ALL_DETAILED_STATS
            )
    
    # 5. Evaluation Finale & Sauvegarde
    results_json = evaluate_model(full_model, test_loader, class_names, TAXONOMY_MAP, ALL_DETAILED_STATS)
    
    final_json_path = os.path.join(OUTPUT_DIR, "predictions_final.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
        
    # Nettoyage
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint intermédiaire supprimé (Entrainement terminé).")

    print(f"\nTerminé ! Résultats complets dans : {final_json_path}")