###########################
# Imports des librairies  #
###########################


import os
from PIL import Image
import tarfile
import open_clip
import torch
import torch.nn.functional
import json
from Dataset_shrinker import dataset_shrinker
from Metadata_dataset import extract_metadata_from_tar


############################
# Import du model Bioclip2 #
############################


# verifie si GPU disponible (mais moi je n'ai pas)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Charge le modele et le preprocess
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
model.to(device)
# Tokenizer pour transformer du texte en embedding (on utilise pas ici mais c'est dans le code exemple)
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')


#############################
# Predictions des candidats #
#############################


# Fonction pour pred en zero-shot
def predict_image_zero_shot(image_file, candidate_labels): # en entrée une image et des candidats potentiels
    
    # Prétraitement de l'image
    image = Image.open(image_file).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Embedding de l'image
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # prompt pour chaque photo
    textes = [f"an insect belonging to the species {label}" for label in candidate_labels]
    texte_tokens = tokenizer(textes).to(device)

    # Embedding du texte
    with torch.no_grad():
        text_features = model.encode_text(texte_tokens)

    # norm
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cos sim et softmax pour obtenir des probas pour chauqe candidats
    logits = 100.0 * image_features @ text_features.T
    probs = logits.softmax(dim=-1)[0]

    # Trie les labels par proba decroissante (meilleur en premier)
    probs = probs.cpu().numpy()
    tri_index = probs.argsort()[::-1]
    return [(candidate_labels[i], float(probs[i])) for i in tri_index]





if __name__ == "__main__":
    
    tar_path = "Data/database.tar"
    small_tar_path = "Data/small_database.tar"
    small_metadata_path = "Data/small_metadata_images.json"
    pred_results_path = "Data/zero_shot_predictions.json"
    
    # petit dataset
    dataset_small = dataset_shrinker(input_tar=tar_path,n_folders=50,n_files=10, output_tar=small_tar_path)
    # json du dataset
    metadata = extract_metadata_from_tar(tar_path=small_tar_path, out_json_path=small_metadata_path)
    
    # load le json metadata
    with open(small_metadata_path, "r", encoding="utf-8") as f:
        small_metadata_json = json.load(f)


    bug_metadata = {}
    unique_candidate = set()
    for bug in small_metadata_json:
        fold_num = bug.get("folder_number")
        gbif_info = bug.get("gbif")
        if gbif_info is not None:
            canonical_name = gbif_info.get("canonicalName")
            bug_metadata[fold_num] = canonical_name
            if canonical_name is not None:
                unique_candidate.add(canonical_name)

    unique_candidate = sorted(list(unique_candidate))

    candidate_info = {}
    for bug in small_metadata_json:
        gbif_info = bug.get("gbif")
        if not gbif_info:
            continue
        canonical = gbif_info.get("canonicalName")
        if not canonical or canonical in candidate_info:
            continue
        candidate_info[canonical] = {
            "genus": gbif_info.get("genus"),
            "family": gbif_info.get("family")
        }

    # Pred zeroshot
    session_results = []
    with tarfile.open(small_tar_path, "r:*") as star:
        for member in star.getmembers():
            if not member.isfile():
                continue

            parts = member.name.split("/")
            if len(parts) < 2:
                continue

            # folder number  
            folder_num = int(parts[-2])

            with star.extractfile(member) as bug_image:
                preds = predict_image_zero_shot(bug_image, unique_candidate)[:5]

            real_name = bug_metadata.get(folder_num)

            # le max de vrais enfos de l'insecte
            real_info = candidate_info.get(real_name, {}) if real_name else {}
            real_genus = real_info.get("genus")
            real_family = real_info.get("family")

            # enrich top5 avec genus et family si dans candidate info
            top5 = []
            for label, prob in preds:
                info = candidate_info.get(label, {})
                top5.append({
                    "label": label,
                    "prob": prob,
                    "genus": info.get("genus"),
                    "family": info.get("family")
                })

            session_results.append({
                "archive_path": member.name,
                "folder_number": folder_num,
                "real_name": real_name,
                "real_genus": real_genus,
                "real_family": real_family,
                "top5": top5
            })



    os.makedirs(os.path.dirname(pred_results_path), exist_ok=True)
    with open(pred_results_path, "w", encoding="utf-8") as f:
        json.dump(session_results, f, ensure_ascii=False, indent=2)
    print(f"{len(session_results)} dans {pred_results_path}")