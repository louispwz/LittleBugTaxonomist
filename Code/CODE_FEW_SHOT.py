import os
import random
import torch
import torch.nn.functional as F
import numpy as np 
from PIL import Image
import open_clip
from sklearn.linear_model import LogisticRegression 

DATA_DIR = os.path.join('Data', 'data_few_shot')
N_SHOT = 10
SEED = 42   

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use of {device}")

# On charge le modele et le preprocess
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
model.to(device)
model.eval()

# Préparation des données
def charger_dataset_few_shot(root_dir, n_shot):
    random.seed(SEED)
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    support_imgs = []
    support_labels = []
    query_imgs = []
    query_labels = []
    
    print(f"Data {len(classes)} species :")
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg'))]
        files.sort()
        random.shuffle(files)
            
        files_support = files[:n_shot]
        files_query = files[n_shot:]
        
        cls_idx = class_to_idx[cls_name]
        
        for f in files_support:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image)
            support_imgs.append(transformed_image)
            support_labels.append(cls_idx)
            
        for f in files_query:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image)
            query_imgs.append(transformed_image)
            query_labels.append(cls_idx)
            
        print(f"  - {cls_name} : {len(files_support)} Support / {len(files_query)} Query")

    if not support_imgs:
        raise ValueError("No support images found")

    return (
        torch.stack(support_imgs).to(device),
        torch.tensor(support_labels).to(device),
        torch.stack(query_imgs).to(device),
        torch.tensor(query_labels).to(device),
        classes
    )

# Extraire les features + covertir en numpy
def get_features_as_numpy(model, images):
    with torch.no_grad():
        # Encodage via BioCLIP
        features = model.encode_image(images)
        # Normalisation
        features = F.normalize(features, dim=-1)
        
    # Transfert vers CPU + covertir e numpy
    return features.cpu().numpy()

if __name__ == "__main__":
    # On prépare les données
    sup_img, sup_lbl, qry_img, qry_lbl, class_names = charger_dataset_few_shot(DATA_DIR, N_SHOT)
    
    # Extraction des features (freeze)
    # On transforme les images en vecteurs de nombres
    train_features = get_features_as_numpy(model, sup_img)
    test_features = get_features_as_numpy(model, qry_img)
    
    # On récupère les labels en numpy aussi
    train_labels = sup_lbl.cpu().numpy()
    test_labels = qry_lbl.cpu().numpy()

    # Entraînement de Logistic Regression
    # C=0.316 est une valeur issue du papier de CLIP pour le few-shot
    classifier = LogisticRegression(random_state=SEED, C=0.316, max_iter=1000, verbose=0)
    classifier.fit(train_features, train_labels)

    # Prédiction
    predictions_numpy = classifier.predict(test_features)
    
    # On reconvertit en Tensor
    preds = torch.from_numpy(predictions_numpy).to(device)

    # Calcul de l'Accuracy
    correct = (preds == qry_lbl).sum().item()
    total = len(qry_lbl)
    accuracy = correct / total * 100
    
    print(f"Global Accuracy : {accuracy:.2f}%")
    print("-" * 30)
    
    # par espèce
    for i, class_name in enumerate(class_names):
        indices_espece = (qry_lbl == i).nonzero(as_tuple=True)[0]
        
        if len(indices_espece) > 0:
            preds_espece = preds[indices_espece]
            targets_espece = qry_lbl[indices_espece]
            
            correct_espece = (preds_espece == targets_espece).sum().item()
            total_espece = len(indices_espece)
            acc_espece = correct_espece / total_espece * 100
            
            detail_erreurs = ""
            if correct_espece < total_espece:
                mask_erreur = preds_espece != targets_espece
                mauvaises_preds = preds_espece[mask_erreur]
                
                comptage_confusions = {}
                for p in mauvaises_preds:
                    nom_confus = class_names[p.item()]
                    comptage_confusions[nom_confus] = comptage_confusions.get(nom_confus, 0) + 1
                
                detail_erreurs = " missclassed with " + ", ".join([f"{k} ({v})" for k, v in comptage_confusions.items()])
            
            print(f"  - {class_name:<25} : {acc_espece:6.2f}% ({correct_espece}/{total_espece}){detail_erreurs}")