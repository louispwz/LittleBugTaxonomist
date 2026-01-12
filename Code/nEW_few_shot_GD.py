import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import open_clip
import json
import time 
import copy

DATA_DIR = os.path.join('Data', 'data_new_few_shot') 
N_SHOT = [1, 5, 10, 25, 50]
N_QUERY = 5     
SEED = 123
OUTPUT_DIR = "Data"

# config
EPOCHS = 5              
LEARNING_RATE = 1e-5    # Vitesse d'apprentissage faible pour ne pas casser le modèle pré-entrainé
BATCH_SIZE = 16         

# Noms de fichiers pour le NON-FREEZE
TIMING_FILE = os.path.join(OUTPUT_DIR, "new_timings_summary_finetune.json") 


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use of {device}")

# Fonction pour charger le modèle à neuf (Reset)
def load_fresh_model():
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    model.to(device)
    return model, preprocess

# Encodage par batch
def encode_in_batches(model, images_tensor, batch_size=32):
    all_features = []
    total = len(images_tensor)
    model.eval()
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = images_tensor[i : i + batch_size].to(device)
            features = model.encode_image(batch)
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu())
            
    return torch.cat(all_features)

# fct de fine tuning
def fine_tune_model(model, support_images, support_labels, epochs=5, lr=1e-5, batch_size=16):

    model.train() 
    
    # On gèle la partie texte du modèle car on n'a pas de texte ici, on veut juste adapter la vision
    # Ou alors on entraîne tout, mais attention à la mémoire. 
    # Pour Bioclip, souvent on entraîne tout le vision encoder.
    
    optimizer = optim.AdamW(model.visual.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Création d'un DataLoader
    dataset = TensorDataset(support_images, support_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting Fine-tuning for {epochs} epochs")
    
    for epoch in range(epochs): 
        total_loss = 0
        for batch_imgs, batch_lbls in loader:
            batch_imgs, batch_lbls = batch_imgs.to(device), batch_lbls.to(device)
            
            optimizer.zero_grad()
            
            # on récupère les features images
            image_features = model.encode_image(batch_imgs)
            image_features = F.normalize(image_features, dim=-1)
            
    #  tête de classification temporaire
    # Input=768 , Output=nombre de classes dans le support
    num_classes = len(torch.unique(support_labels))
    classifier_head = nn.Linear(768, num_classes).to(device) # 
    
    # On optimise les deux le modèle et la tête
    params_to_optimize = list(model.visual.parameters()) + list(classifier_head.parameters())
    optimizer = optim.AdamW(params_to_optimize, lr=lr)
    
    # Mapping global label -> local label (0 à N-1) pour CrossEntropy
    # support_labels contient peut-être [10, 55, 120] CrossEntropy veut [0, 1, 2]
    unique_classes = sorted(torch.unique(support_labels).tolist())
    map_label = {real_cls: i for i, real_cls in enumerate(unique_classes)}
    
    model.train()
    classifier_head.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_imgs, batch_lbls in loader:
            batch_imgs = batch_imgs.to(device)
            
            # Conversion des labels globaux en labels locaux (0..N)
            local_lbls = torch.tensor([map_label[l.item()] for l in batch_lbls]).to(device)
            
            optimizer.zero_grad()
            
            # Extraction des features (grad)
            features = model.encode_image(batch_imgs)
            # features = F.normalize(features, dim=-1)
            
            # classifier
            logits = classifier_head(features)
            
            # Calcul de la loss
            loss = criterion(logits, local_lbls)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    model.eval()
    print("Fine-tuning done")
    return model


# Préparation des données (Identique au code précédent)
def charger_dataset_few_shot(root_dir, n_shot, preprocess_fn):
    random.seed(SEED)
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Le dossier {root_dir} n'existe pas.")

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    support_imgs = []
    support_labels = []
    query_imgs = []
    query_labels = []
    query_paths = []  
    
    print(f"Dataset contains {len(classes)} species.")
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()
        random.shuffle(files)
            
        # Sélection Support / Query
        split_idx = min(n_shot, len(files))
        files_support = files[:split_idx]
        files_query = files[split_idx : split_idx + N_QUERY]
        
        cls_idx = class_to_idx[cls_name]
        
        # Chargement Support
        for f in files_support:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess_fn(image)
            support_imgs.append(transformed_image)
            support_labels.append(cls_idx)
            
        # Chargement Query
        for f in files_query:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess_fn(image)
            query_imgs.append(transformed_image)
            query_labels.append(cls_idx)
            query_paths.append(os.path.join(cls_name, f))
            
    if not support_imgs:
        raise ValueError("No support images found.")

    print(f"Total loaded: {len(support_imgs)} Support images, {len(query_imgs)} Query images.")

    return (
        torch.stack(support_imgs),        
        torch.tensor(support_labels),     
        torch.stack(query_imgs),          
        torch.tensor(query_labels),       
        classes,
        query_paths
    )

# classifier
def few_shot_classification_after_finetune(model, support_images, support_labels, query_images, top_k=5, batch_size=32):
    
    # Maintenant que le modèle a changé, on recalcule les embeddings pour le support et le query avec les nouveaux poids.
    
    support_features_cpu = encode_in_batches(model, support_images, batch_size)
    query_features_cpu = encode_in_batches(model, query_images, batch_size)

    # Transfert vers GPU
    support_features = support_features_cpu.to(device)
    query_features = query_features_cpu.to(device)
    support_labels = support_labels.to(device)

    # Prototypes
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

    # Similarité cos
    logits = 100.0 * torch.matmul(query_features, prototypes.T)
    probs = logits.softmax(dim=-1)
    
    # Top K
    real_top_k = min(top_k, len(unique_classes))
    top_probs, top_indices = probs.topk(real_top_k, dim=1)
    
    return top_probs, top_indices


if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_timings = []

    for n in N_SHOT:
        print(f"\nCLASSIFICATION {n}-SHOT")
        
        current_timing = {"n_shot": n}
        
        # reset du modèle a chaque few-shot
        if device == "cuda": torch.cuda.empty_cache()
        model, preprocess_fn = load_fresh_model()

        try:
            # loading
            t_start_load = time.time()
            sup_img, sup_lbl, qry_img, qry_lbl, class_names, qry_paths = charger_dataset_few_shot(DATA_DIR, n, preprocess_fn)
            t_end_load = time.time()
            
            current_timing["loading_seconds"] = round(t_end_load - t_start_load, 4)
            print(f"Data Loaded in {current_timing['loading_seconds']:.4f}s.")
            
            # Fine tuning
            t_start_train = time.time()
            
            model = fine_tune_model(
                model, 
                sup_img, 
                sup_lbl, 
                epochs=EPOCHS, 
                lr=LEARNING_RATE, 
                batch_size=BATCH_SIZE
            )
            t_end_train = time.time()
            
            current_timing["training_seconds"] = round(t_end_train - t_start_train, 4)
            print(f"Training done in {current_timing['training_seconds']:.4f}s")
            
            # inference
            t_start_infer = time.time()
            if device == "cuda": torch.cuda.synchronize()
            
            top_probs, top_indices = few_shot_classification_after_finetune(
                model, sup_img, sup_lbl, qry_img, top_k=5, batch_size=BATCH_SIZE
            )
            
            if device == "cuda": torch.cuda.synchronize()
            t_end_infer = time.time()
            
            current_timing["inference_seconds"] = round(t_end_infer - t_start_infer, 4)
            print(f"Inference done in {current_timing['inference_seconds']:.4f}s")

            # json
            t_start_format = time.time()
            session_results = []
            
            top_indices = top_indices.cpu()
            top_probs = top_probs.cpu()
            qry_lbl = qry_lbl.cpu()

            for i in range(len(qry_lbl)):
                real_label_idx = qry_lbl[i].item()
                real_name = class_names[real_label_idx]
                real_genus = real_name.split(' ')[0]
                
                top5 = []
                for k in range(top_indices.shape[1]):
                    pred_idx = top_indices[i, k].item()
                    prob = top_probs[i, k].item()
                    
                    pred_label = class_names[pred_idx]
                    pred_genus = pred_label.split(' ')[0]
                    
                    top5.append({
                        "label": pred_label,
                        "prob": prob,              
                        "genus": pred_genus,
                        "family": None               
                    })
                
                session_results.append({
                    "archive_path": qry_paths[i],
                    "folder_number": real_label_idx, 
                    "real_name": real_name,
                    "real_genus": real_genus,
                    "real_family": None,
                    "top5": top5
                })
            t_end_format = time.time()
            current_timing["formatting_seconds"] = round(t_end_format - t_start_format, 4)

            # 5. SAVING
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
            current_timing["error"] = str(e)
            all_timings.append(current_timing)

    # Save timings
    with open(TIMING_FILE, "w", encoding="utf-8") as f:
        json.dump(all_timings, f, indent=2)
    
    print(f"\nGlobal timing summary saved in {TIMING_FILE}")