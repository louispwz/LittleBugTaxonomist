import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

DATA_DIR = os.path.join('Data', 'data_few_shot')
N_SHOT = [1, 5, 10,25,50]
SEED = 123

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use of {device}")

# On charge le modele et le preprocess
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
model.to(device)
model.eval()                                       # on met le modèle en mode évaluation (pas d'entraînement)

# Préparation des données
def charger_dataset_few_shot(root_dir, n_shot):
    
    # Parcourt les dossiers, sépare les images en Support (n_shot) et Query (le reste)
    # Renvoie des Tensors prêts pour le modèle
    
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
        # On récupère les images 
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg'))]
        files.sort()
        random.shuffle(files)
            
        # Séparation support / query
        files_support = files[:n_shot]                      # les n premières
        files_query = files[n_shot:]                        # le reste
        
        cls_idx = class_to_idx[cls_name]
        
        # Chargement et Preprocessing Support
        for f in files_support:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image) # Transforme en Tensor [3, 224, 224]
            support_imgs.append(transformed_image)
            support_labels.append(cls_idx)
            
        # Chargement et Preprocessing Query
        for f in files_query:
            img_path = os.path.join(cls_path, f)
            image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(image)
            query_imgs.append(transformed_image)
            query_labels.append(cls_idx)
            
        print(f"  - {cls_name} : {len(files_support)} Support / {len(files_query)} Query")

    # Conversion en gros Tensors PyTorch
    if not support_imgs:
        raise ValueError("No support images found")

    return (
        torch.stack(support_imgs).to(device),
        torch.tensor(support_labels).to(device),
        torch.stack(query_imgs).to(device),
        torch.tensor(query_labels).to(device),
        classes
    )

# Fonction de classification
def few_shot_classification(model, support_images, support_labels, query_images):
    
    # Classifie les query_images en comparant leur distance avec les prototypes des support_images
    
    with torch.no_grad():                       # Pas de calcul de gradient
        
        # Encodage (extraction des caractéristiques)
        # vecteur à 768 dimensions
        support_features = model.encode_image(support_images)
        support_features = F.normalize(support_features, dim=-1)
        
        query_features = model.encode_image(query_images)
        query_features = F.normalize(query_features, dim=-1)

    # Prototypes
    unique_classes = torch.unique(support_labels)
    unique_classes = sorted(unique_classes.tolist()) # S'assurer que l'ordre est 0, 1, 2...
    prototypes = []
    
    for c in unique_classes:
        # On sélectionne les vecteurs de la classe 'c'
        class_mask = (support_labels == c)
        class_features = support_features[class_mask]
        
        # Moyenne (Centroïde)
        mean_feature = class_features.mean(dim=0)
        mean_feature = F.normalize(mean_feature, dim=-1) # O ormalise
        prototypes.append(mean_feature)
        
    prototypes = torch.stack(prototypes) # Taille : [Nb_Classes, 768]

    # Calcul de similarité cosinus
    # Matrice de scores : [Nb_Query, Nb_Classes]
    similarities = torch.matmul(query_features, prototypes.T)
    
    # Prédiction
    predictions = torch.argmax(similarities, dim=1)
    
    return predictions

# On exécute le tout

if __name__ == "__main__":
    for n in N_SHOT:
        print(f"CLASSIFICATION{n}-SHOT")
    # On prépare les données
        sup_img, sup_lbl, qry_img, qry_lbl, class_names = charger_dataset_few_shot(DATA_DIR, n)
    
        print(f"{n}-shot classification with {len(qry_lbl)} test")
    
    # Classification
        preds = few_shot_classification(model, sup_img, sup_lbl, qry_img)
    
    # Acuracy
        correct = (preds == qry_lbl).sum().item()
        total = len(qry_lbl)
        accuracy = correct / total * 100
    
        print(" iveau espece ")
        print(f"Acuracy : {accuracy:.2f}%")
    
    
    # par espèce 
    
    # On boucle sur chaque espèce 
        for i, class_name in enumerate(class_names):
        # On récupère les indices des images de test appartenant à cette espèce
            indices_espece = (qry_lbl == i).nonzero(as_tuple=True)[0]
        
            if len(indices_espece) > 0:
                preds_espece = preds[indices_espece]
                targets_espece = qry_lbl[indices_espece]
            
                correct_espece = (preds_espece == targets_espece).sum().item()
                total_espece = len(indices_espece)
                acc_espece = correct_espece / total_espece * 100
            
            # On regarde les erreurs
                detail_erreurs = ""
                if correct_espece < total_espece:
                # Predictions fausses
                    mask_erreur = preds_espece != targets_espece
                    mauvaises_preds = preds_espece[mask_erreur]
                
                    comptage_confusions = {}
                    for p in mauvaises_preds:
                        nom_confus = class_names[p.item()]
                        comptage_confusions[nom_confus] = comptage_confusions.get(nom_confus, 0) + 1
                
                # texte
                    detail_erreurs = " missclassed with " + ", ".join([f"{k} ({v})" for k, v in comptage_confusions.items()])
            
                print(f"  - {class_name:<25} : {acc_espece:6.2f}% ({correct_espece}/{total_espece}){detail_erreurs}")

        # par image
        #print("Predictions")
        #for i in range(total):
        #    vrai_nom = class_names[qry_lbl[i]]
        #    pred_nom = class_names[preds[i]]
        #    statut = "Correct" if preds[i] == qry_lbl[i] else "False"
        #    print(f"Image {i+1} ({vrai_nom}) PREDICTED AS {pred_nom} {statut}")
        
        print(" iveau genre ")
        
        # Extraction des genres + mapping
        genres_names_list = [c.split(' ')[0] for c in class_names]
        unique_genera = sorted(list(set(genres_names_list)))
        
        genus_to_idx = {g: i for i, g in enumerate(unique_genera)}
        
        # Conversion index espèce / index genre
        species_id_to_genus_id = torch.tensor([genus_to_idx[g] for g in genres_names_list]).to(device)
        
        # Conversion de tous les labels 
        qry_lbl_genus = species_id_to_genus_id[qry_lbl]
        preds_genus = species_id_to_genus_id[preds]

        correct_genus_global = (preds_genus == qry_lbl_genus).sum().item()
        total_imgs = len(qry_lbl)
        acc_genus_global = correct_genus_global / total_imgs * 100
        
        print(f" Accuracy : {acc_genus_global:.2f}%")
        print("-" * 30)

        # Détails par Genre 
        for i, genus_name in enumerate(unique_genera):
            indices_genus = (qry_lbl_genus == i).nonzero(as_tuple=True)[0]
            
            if len(indices_genus) > 0:
                preds_g = preds_genus[indices_genus]
                targets_g = qry_lbl_genus[indices_genus]
                
                correct_g = (preds_g == targets_g).sum().item()
                total_g = len(indices_genus)
                acc_g = correct_g / total_g * 100
                
                detail_erreurs = ""
                if correct_g < total_g:
                    mask_erreur = preds_g != targets_g
                    mauvaises_preds = preds_g[mask_erreur]
                    
                    comptage_confusions = {}
                    for p in mauvaises_preds:
                        nom_confus = unique_genera[p.item()] 
                        comptage_confusions[nom_confus] = comptage_confusions.get(nom_confus, 0) + 1
                    
                    detail_erreurs = " missclassed with " + ", ".join([f"{k} ({v})" for k, v in comptage_confusions.items()])
                
                print(f"  - {genus_name:<25} : {acc_g:6.2f}% ({correct_g}/{total_g}){detail_erreurs}")