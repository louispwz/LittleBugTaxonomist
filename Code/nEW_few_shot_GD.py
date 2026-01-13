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
import gc # Garbage Collector pour libérer la RAM

# --- CONFIGURATION STRICTE POUR ÉVITER LE CRASH ---
DATA_DIR = os.path.join('Data', 'data_new_few_shot') 
N_SHOT = [1, 5, 10, 25, 50]
N_QUERY = 5     
SEED = 123
OUTPUT_DIR = "Data"

# Paramètres d'entraînement
EPOCHS = 5              
LEARNING_RATE = 1e-5    

# --- OPTIMISATION CRITIQUE ---
# On traite les images 4 par 4 pour ne pas saturer le GPU.
# Mais on met à jour les poids toutes les 4 étapes (4*4 = 16 images "virtuelles")
BATCH_SIZE = 4          
GRADIENT_ACCUMULATION_STEPS = 4 

TIMING_FILE = os.path.join(OUTPUT_DIR, "new_timings_summary_finetune_lazy.json") 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use of {device}")

# --- 1. MODÈLE & RESET ---
print("Loading Model once...")
base_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
base_model.to(device)

# Sauvegarde de l'état initial en RAM (c'est léger, ce ne sont que des poids)
INITIAL_STATE_DICT = copy.deepcopy(base_model.state_dict())
print("Model loaded.")

def reset_model_to_initial_state(model):
    model.load_state_dict(INITIAL_STATE_DICT)
    return model

# --- 2. DATASET "PARESSEUX" (LAZY LOADING) ---
# C'est cette classe qui empêche la RAM d'exploser.
# Elle ne stocke que les chemins (strings), pas les pixels.
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
            # On charge l'image seulement maintenant !
            image = Image.open(path).convert("RGB")
            tensor = self.preprocess(image)
            return tensor, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # En cas d'erreur, on renvoie un tenseur noir pour ne pas planter
            return torch.zeros((3, 224, 224)), label

# --- 3. PRÉPARATION DES DONNÉES (SANS CHARGEMENT) ---
def prepare_dataset_paths(root_dir, n_shot):
    random.seed(SEED)
    
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # On stocke juste les chemins, pas les images !
    support_paths = []
    support_labels = []
    query_paths = []
    query_labels = []
    
    print(f"Scanning files for {len(classes)} species...")
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()
        random.shuffle(files)
            
        split_idx = min(n_shot, len(files))
        files_support = files[:split_idx]
        files_query = files[split_idx : split_idx + N_QUERY]
        
        cls_idx = class_to_idx[cls_name]
        
        for f in files_support:
            support_paths.append(os.path.join(cls_path, f))
            support_labels.append(cls_idx)
            
        for f in files_query:
            query_paths.append(os.path.join(cls_path, f))
            query_labels.append(cls_idx)

    return (support_paths, support_labels, query_paths, query_labels, classes)

# --- 4. FINE-TUNING (GRADIENT ACCUMULATION) ---
def fine_tune_model(model, support_dataset, num_classes, epochs=5, lr=1e-5):
    model.train() 
    
    classifier_head = nn.Linear(768, num_classes).to(device)
    classifier_head.train()
    
    params_to_optimize = list(model.visual.parameters()) + list(classifier_head.parameters())
    optimizer = optim.AdamW(params_to_optimize, lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))
    
    # num_workers=0 est crucial sur Windows pour éviter que ça "tourne dans le vide"
    # pin_memory=True accélère le transfert vers le GPU
    loader = DataLoader(support_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    # Mapping label global -> local
    # On doit trouver les labels uniques présents dans le dataset pour mapper 0..N
    unique_labels = sorted(list(set(support_dataset.labels)))
    map_label = {real_cls: i for i, real_cls in enumerate(unique_labels)}
    
    print(f"Fine-tuning ({epochs} epochs)...")
    
    optimizer.zero_grad()
    
    for epoch in range(epochs): 
        for i, (batch_imgs, batch_lbls) in enumerate(loader):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            local_lbls = torch.tensor([map_label[l.item()] for l in batch_lbls]).to(device)
            
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                features = model.encode_image(batch_imgs)
                logits = classifier_head(features)
                loss = criterion(logits, local_lbls)
                # On divise la loss par le nombre d'accumulation pour normaliser
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            # On met à jour les poids seulement tous les X steps
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
    model.eval()
    # On supprime la tête de classification et on vide le cache
    del classifier_head
    del optimizer
    torch.cuda.empty_cache()
    return model

# --- 5. ENCODAGE & INFERENCE ---
def encode_dataset_features(model, dataset):
    """Encode tout un dataset Lazy sans tout charger en même temps"""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    all_features = []
    
    model.eval()
    with torch.no_grad():
        for batch_imgs, _ in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                features = model.encode_image(batch_imgs)
                features = F.normalize(features, dim=-1)
            all_features.append(features.cpu())
            
    return torch.cat(all_features)

def few_shot_classification_final(model, support_dataset, query_dataset, top_k=5):
    
    # 1. Encodage
    support_features = encode_dataset_features(model, support_dataset).to(device)
    query_features = encode_dataset_features(model, query_dataset).to(device)
    
    # Les labels sont stockés dans le dataset (liste python), on les convertit en tenseur
    support_labels = torch.tensor(support_dataset.labels).to(device)

    # 2. Prototypes
    unique_classes = torch.unique(support_labels)
    unique_classes = sorted(unique_classes.tolist())
    prototypes = []
    
    for c in unique_classes:
        class_mask = (support_labels == c)
        class_features = support_features[class_mask]
        mean_feature = class_features.mean(dim=0)
        mean_feature = F.normalize(mean_feature, dim=-1)
        prototypes.append(mean_feature)
        
    prototypes = torch.stack(prototypes) 

    # 3. Similarité
    logits = 100.0 * torch.matmul(query_features, prototypes.T)
    probs = logits.softmax(dim=-1)
    
    real_top_k = min(top_k, len(unique_classes))
    top_probs, top_indices = probs.topk(real_top_k, dim=1)
    
    return top_probs, top_indices

# --- MAIN LOOP ---
if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_timings = []
    model = base_model

    for n in N_SHOT:
        print(f"\n==========================================")
        print(f"STARTING {n}-SHOT FINE-TUNING EXPERIMENT")
        print(f"==========================================")
        
        current_timing = {"n_shot": n}
        
        # 0. NETTOYAGE MÉMOIRE AVANT DE COMMENCER
        gc.collect()
        torch.cuda.empty_cache()
        model = reset_model_to_initial_state(model)

        try:
            # 1. PRÉPARATION (SCAN SEULEMENT, TRÈS RAPIDE)
            t_start_load = time.time()
            s_paths, s_labels, q_paths, q_labels, class_names = prepare_dataset_paths(DATA_DIR, n)
            
            # Création des Datasets Paresseux
            support_ds = LazyBugDataset(s_paths, s_labels, preprocess)
            query_ds = LazyBugDataset(q_paths, q_labels, preprocess)
            
            t_end_load = time.time()
            current_timing["loading_seconds"] = round(t_end_load - t_start_load, 4)
            print(f"Paths ready in {current_timing['loading_seconds']:.4f}s. (Lazy Loading Active)")
            
            # 2. TRAINING (Fine-tuning avec petits batchs)
            t_start_train = time.time()
            num_classes_support = len(set(s_labels))
            
            model = fine_tune_model(
                model, 
                support_ds, 
                num_classes_support,
                epochs=EPOCHS, 
                lr=LEARNING_RATE
            )
            t_end_train = time.time()
            current_timing["training_seconds"] = round(t_end_train - t_start_train, 4)
            print(f"Training done in {current_timing['training_seconds']:.4f}s")
            
            # 3. INFERENCE
            t_start_infer = time.time()
            if device == "cuda": torch.cuda.synchronize()
            
            top_probs, top_indices = few_shot_classification_final(model, support_ds, query_ds, top_k=5)
            
            if device == "cuda": torch.cuda.synchronize()
            t_end_infer = time.time()
            current_timing["inference_seconds"] = round(t_end_infer - t_start_infer, 4)
            print(f"Inference done in {current_timing['inference_seconds']:.4f}s")

            # 4. JSON
            t_start_format = time.time()
            session_results = []
            
            top_indices = top_indices.cpu()
            top_probs = top_probs.cpu()
            
            # On récupère les vrais labels depuis la liste python (pas besoin de .cpu())
            qry_lbl_list = q_labels 

            for i in range(len(qry_lbl_list)):
                real_label_idx = qry_lbl_list[i]
                real_name = class_names[real_label_idx]
                real_genus = real_name.split(' ')[0]
                
                top5 = []
                for k in range(top_indices.shape[1]):
                    pred_idx = top_indices[i, k].item()
                    prob = top_probs[i, k].item()
                    pred_label = class_names[pred_idx]
                    top5.append({
                        "label": pred_label,
                        "prob": prob,              
                        "genus": pred_label.split(' ')[0],
                        "family": None               
                    })
                
                session_results.append({
                    "archive_path": q_paths[i], # On a déjà le path dans la liste
                    "folder_number": real_label_idx, 
                    "real_name": real_name,
                    "real_genus": real_genus,
                    "real_family": None,
                    "top5": top5
                })
            t_end_format = time.time()
            current_timing["formatting_seconds"] = round(t_end_format - t_start_format, 4)

            # 5. SAVE
            t_start_save = time.time()
            output_json_path = os.path.join(OUTPUT_DIR, f"new_few_shot_predictions_{n}shot_finetuned.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(session_results, f, ensure_ascii=False, indent=2)
            t_end_save = time.time()
            current_timing["saving_seconds"] = round(t_end_save - t_start_save, 4)
            
            current_timing["total_cycle_seconds"] = round(current_timing["loading_seconds"] + 
                                                          current_timing["training_seconds"] + 
                                                          current_timing["inference_seconds"] + 
                                                          current_timing["formatting_seconds"] + 
                                                          current_timing["saving_seconds"], 4)
            all_timings.append(current_timing)
            print(f"Results saved in {output_json_path}")
            
        except Exception as e:
            print(f"Error {n}-shot : {e}")
            import traceback
            traceback.print_exc()
            current_timing["error"] = str(e)
            all_timings.append(current_timing)

    with open(TIMING_FILE, "w", encoding="utf-8") as f:
        json.dump(all_timings, f, indent=2)
    
    print(f"\nGlobal timing summary saved in {TIMING_FILE}")