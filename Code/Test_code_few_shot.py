import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import json
import time 


DATA_DIR = os.path.join('Data', 'data_few_shot')
METADATA_PATH = os.path.join('Data', 'metadata_images.json') # Chemin des métadonnées

N_SHOT = [1, 5, 10, 25, 50]
SEED = 123
OUTPUT_DIR = "Data"
TIMING_FILE = os.path.join(OUTPUT_DIR, "timings_summary.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use of {device}")

# dictionnaire des metadata

TAXONOMY_MAP = {}

try:
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        meta_list = json.load(f)
        
    for entry in meta_list:
        gbif = entry.get('gbif')
        
        # Si gbif est null, on passe à la suite
        if not gbif:
            continue
            
        species = gbif.get('species')
        if species:
            # On normalise le nom pour qu'il corresponde aux noms de dossiers (Espaces -> Underscores)
            # Ex: "Agonum nigrum" -> "Agonum_nigrum"
            clean_name = species.replace(" ", "_").replace(".", "")
            
            TAXONOMY_MAP[clean_name] = {
                "family": gbif.get("family"),
                "folder_number": entry.get("folder_number")
            }
            
    print(f"Taxonomie chargee pour {len(TAXONOMY_MAP)} especes")
    
except FileNotFoundError:
    print("ATTENTION : Fichier metadata_images.json introuvable")
except json.JSONDecodeError:
    print("error le fichier JSON est mal formé.")



model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
model.to(device)
model.eval()

def charger_dataset_few_shot(root_dir, n_shot):
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
    
    print(f"Data {len(classes)} species :")
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()
        random.shuffle(files)
            
        # Séparation support / query
        split_idx = min(n_shot, len(files))
        files_support = files[:split_idx]
        files_query = files[split_idx:]
        
        cls_idx = class_to_idx[cls_name]
        
        # Chargement Support
        for f in files_support:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image)
            support_imgs.append(transformed_image)
            support_labels.append(cls_idx)
            
        # Chargement Query
        for f in files_query:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image)
            query_imgs.append(transformed_image)
            query_labels.append(cls_idx)
            query_paths.append(os.path.join(cls_name, f))
            
    if not support_imgs:
        raise ValueError("No support images found")

    return (
        torch.stack(support_imgs).to(device),
        torch.tensor(support_labels).to(device),
        torch.stack(query_imgs).to(device),
        torch.tensor(query_labels).to(device),
        classes,
        query_paths
    )

def few_shot_classification_topk(model, support_images, support_labels, query_images, top_k=5):
    with torch.no_grad():
        support_features = model.encode_image(support_images)
        support_features = F.normalize(support_features, dim=-1)
        
        query_features = model.encode_image(query_images)
        query_features = F.normalize(query_features, dim=-1)

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

    logits = 100.0 * torch.matmul(query_features, prototypes.T)
    probs = logits.softmax(dim=-1)
    
    top_probs, top_indices = probs.topk(min(top_k, len(unique_classes)), dim=1)
    return top_probs, top_indices


if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_timings = []

    for n in N_SHOT:
        print(f"\nCLASSIFICATION {n}-SHOT")
        current_timing = {"n_shot": n}
        
        try:
            # Loading
            t_start_load = time.time()
            sup_img, sup_lbl, qry_img, qry_lbl, class_names, qry_paths = charger_dataset_few_shot(DATA_DIR, n)
            t_end_load = time.time()
            
            duration_load = t_end_load - t_start_load
            current_timing["loading_seconds"] = round(duration_load, 4)
            print(f"Data Loaded in {duration_load:.4f}s. Processing {len(qry_lbl)} test images")
            
            # Inference
            t_start_infer = time.time()
            if device == "cuda": torch.cuda.synchronize()
            top_probs, top_indices = few_shot_classification_topk(model, sup_img, sup_lbl, qry_img, top_k=5)
            if device == "cuda": torch.cuda.synchronize()
            t_end_infer = time.time()
            
            duration_infer = t_end_infer - t_start_infer
            current_timing["inference_seconds"] = round(duration_infer, 4)
            print(f"Inference done in {duration_infer:.4f}s")

            # Formatting & Adding Metadata
            t_start_format = time.time()
            session_results = []
            
            for i in range(len(qry_lbl)):
                real_label_idx = qry_lbl[i].item()
                real_name = class_names[real_label_idx]
                
                # --- RECUPERATION INFOS VIA DICTIONNAIRE ---
                info_espece = TAXONOMY_MAP.get(real_name, {})
                real_genus = real_name.split('_')[0]
                real_family = info_espece.get("family") # Maintenant disponible
                real_folder_num = info_espece.get("folder_number")
                
                top5 = []
                for k in range(top_indices.shape[1]):
                    pred_idx = top_indices[i, k].item()
                    prob = top_probs[i, k].item()
                    
                    pred_label = class_names[pred_idx]
                    pred_genus = pred_label.split('_')[0]
                    
                    # Infos de la prédiction aussi
                    pred_info = TAXONOMY_MAP.get(pred_label, {})
                    
                    top5.append({
                        "label": pred_label,
                        "prob": prob,              
                        "genus": pred_genus,
                        "family": pred_info.get("family")             
                    })
                
                session_results.append({
                    "archive_path": qry_paths[i],
                    "folder_number": real_folder_num, 
                    "real_name": real_name,
                    "real_genus": real_genus,
                    "real_family": real_family,
                    "top5": top5
                })
            t_end_format = time.time()
            current_timing["formatting_seconds"] = round(t_end_format - t_start_format, 4)

            # Saving
            t_start_save = time.time()
            output_json_path = os.path.join(OUTPUT_DIR, f"few_shot_predictions_{n}shot.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(session_results, f, ensure_ascii=False, indent=2)
            t_end_save = time.time()
            current_timing["saving_seconds"] = round(t_end_save - t_start_save, 4)
            
            current_timing["total_cycle_seconds"] = round(current_timing["loading_seconds"] + 
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