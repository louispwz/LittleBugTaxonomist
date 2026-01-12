import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import json
import time 

DATA_DIR = os.path.join('Data', 'data_new_few_shot') 
N_SHOT = [1, 5, 10, 25, 50]
SEED = 123
OUTPUT_DIR = "Data"

TIMING_FILE = os.path.join(OUTPUT_DIR, "new_timings_summary_freeze.json") 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use of {device}")

# charge modele + preprocess
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
model.to(device)
model.eval()

# Préparation des données
def charger_dataset_few_shot(root_dir, n_shot):
    random.seed(SEED)
    
    # Vérification que le dossier existe
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Le dossier {root_dir} n'existe pas. Lancez l'extraction d'abord.")

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
        # récupère les images 
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()
        random.shuffle(files)
            
        # Séparation support / query
        split_idx = min(n_shot, len(files))
        files_support = files[:split_idx]
        files_query = files[split_idx:]
        
        cls_idx = class_to_idx[cls_name]
        
        # Chargement + preprocessing support
        for f in files_support:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image)
            support_imgs.append(transformed_image)
            support_labels.append(cls_idx)
            
        # Chargement + preprocessing Query
        for f in files_query:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image)
            query_imgs.append(transformed_image)
            query_labels.append(cls_idx)

            # Stocke le chemin pour JSON output
            query_paths.append(os.path.join(cls_name, f))
            
    if not support_imgs:
        raise ValueError("No support images found in the dataset.")

    print(f"Total loaded: {len(support_imgs)} Support images, {len(query_imgs)} Query images.")

    return (
        torch.stack(support_imgs).to(device),
        torch.tensor(support_labels).to(device),
        torch.stack(query_imgs).to(device),
        torch.tensor(query_labels).to(device),
        classes,
        query_paths
    )

# Classifier
def few_shot_classification_topk(model, support_images, support_labels, query_images, top_k=5):
    
    with torch.no_grad():
        # Encodage
        support_features = model.encode_image(support_images)
        support_features = F.normalize(support_features, dim=-1)
        
        query_features = model.encode_image(query_images)
        query_features = F.normalize(query_features, dim=-1)

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
        
    prototypes = torch.stack(prototypes) # [Nb_Classes, 768]

    # Calcul de similarité cosinus [Nb_Query, Nb_Classes]
    logits = 100.0 * torch.matmul(query_features, prototypes.T)
    
    probs = logits.softmax(dim=-1)
    
    # Récupération des top k
    real_top_k = min(top_k, len(unique_classes))
    top_probs, top_indices = probs.topk(real_top_k, dim=1)
    
    return top_probs, top_indices


if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Liste pour accumuler les stats de temps
    all_timings = []

    for n in N_SHOT:
        print(f"\n----- CLASSIFICATION {n}-SHOT (FULL DATASET) -----")
        
        # Dictionnaire pour stocker les temps de ce cycle
        current_timing = {"n_shot": n}
        
        try:
            # tps chargement 
            t_start_load = time.time()
            sup_img, sup_lbl, qry_img, qry_lbl, class_names, qry_paths = charger_dataset_few_shot(DATA_DIR, n)
            t_end_load = time.time()
            
            duration_load = t_end_load - t_start_load
            current_timing["loading_seconds"] = round(duration_load, 4)
            print(f"Data Loaded in {duration_load:.4f}s.")
            
            # tps classification (=inference ?)
            t_start_infer = time.time()
            if device == "cuda": torch.cuda.synchronize()
            top_probs, top_indices = few_shot_classification_topk(model, sup_img, sup_lbl, qry_img, top_k=5)
            if device == "cuda": torch.cuda.synchronize()
            t_end_infer = time.time()
            
            duration_infer = t_end_infer - t_start_infer
            current_timing["inference_seconds"] = round(duration_infer, 4)
            print(f"Inference done in {duration_infer:.4f}s")

            # tps json
            t_start_format = time.time()
            session_results = []
            
            # itère sur chaque Query
            for i in range(len(qry_lbl)):
                
                real_label_idx = qry_lbl[i].item()
                real_name = class_names[real_label_idx]
                real_genus = real_name.split(' ')[0]    # On suppose format "Genus_species"
                real_family = None                      # Info non disponible
                
                # Predictions
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
                
                # On ajoute l'entrée
                session_results.append({
                    "archive_path": qry_paths[i],
                    "folder_number": real_label_idx, 
                    "real_name": real_name,
                    "real_genus": real_genus,
                    "real_family": real_family,
                    "top5": top5
                })
            t_end_format = time.time()
            current_timing["formatting_seconds"] = round(t_end_format - t_start_format, 4)

            # tps sauvegarde
            t_start_save = time.time()
            
            # MODIFICATION ICI : Ajout de "_freeze"
            output_json_path = os.path.join(OUTPUT_DIR, f"new_few_shot_predictions_{n}shot_freeze.json")
            
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(session_results, f, ensure_ascii=False, indent=2)
            t_end_save = time.time()
            current_timing["saving_seconds"] = round(t_end_save - t_start_save, 4)
            
            # Calcul du total pour ce shot
            current_timing["total_cycle_seconds"] = round(current_timing["loading_seconds"] + 
                                                          current_timing["inference_seconds"] + 
                                                          current_timing["formatting_seconds"] + 
                                                          current_timing["saving_seconds"], 4)
            
            # Ajout à la liste globale
            all_timings.append(current_timing)
            print(f"Results saved in {output_json_path}")
            
        except Exception as e:
            print(f"Error {n}-shot : {e}")
            current_timing["error"] = str(e)
            all_timings.append(current_timing)

    # json timing global
    with open(TIMING_FILE, "w", encoding="utf-8") as f:
        json.dump(all_timings, f, indent=2)
    
    print(f"\nGlobal timing summary saved in {TIMING_FILE}")