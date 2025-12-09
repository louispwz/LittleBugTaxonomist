import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

DATA_DIR = os.path.join('Data', 'data_few_shot')
N_SHOT = 5 
SEED = 42   

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de : {device}")

# On charge le modele et le preprocess
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
model.to(device)
model.eval()                                       # on met le modèle en mode évaluation (pas d'entraînement)

# Préparation des données
def charger_dataset_few_shot(root_dir, n_shot):
    """
    Parcourt les dossiers, sépare les images en Support (n_shot) et Query (le reste).
    Renvoie des Tensors prêts pour le modèle.
    """
    random.seed(SEED)
    
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    support_imgs = []
    support_labels = []
    query_imgs = []
    query_labels = []
    
    print(f"\nPréparation des données ({len(classes)} espèces trouvées) :")
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        # On récupère les images 
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg'))]
        files.sort()
        random.shuffle(files)
            
        # Séparation support / query
        files_support = files[:n_shot]                      # les 5 premières
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
        raise ValueError("Aucune image n'a été chargée. Vérifiez vos dossiers.")

    return (
        torch.stack(support_imgs).to(device),
        torch.tensor(support_labels).to(device),
        torch.stack(query_imgs).to(device),
        torch.tensor(query_labels).to(device),
        classes
    )

# Fonction de classification
def few_shot_classification(model, support_images, support_labels, query_images):
    """
    Classifie les query_images en comparant leur distance avec les prototypes des support_images.
    """
    with torch.no_grad():                       # Pas de calcul de gradient
        
        # Encodage (extraction des caractéristiques)
        # BioCLIP transforme l'image en vecteur à 768 dimensions
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
        mean_feature = F.normalize(mean_feature, dim=-1) # Renormalisation
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
    # On prépare les données
    sup_img, sup_lbl, qry_img, qry_lbl, class_names = charger_dataset_few_shot(DATA_DIR, N_SHOT)
    
    print(f"\nLancement de la classification 5-shot sur {len(qry_lbl)} images de test...")
    
    # On lance la classification
    preds = few_shot_classification(model, sup_img, sup_lbl, qry_img)
    
    # On calculer la précision
    correct = (preds == qry_lbl).sum().item()
    total = len(qry_lbl)
    accuracy = correct / total * 100
    
    print("-" * 30)
    print(f"RÉSULTAT FINAL : {accuracy:.2f}% de précision")
    print("-" * 30)
    
    # Détail par image
    print("\nDétails des prédictions :")
    for i in range(total):
        vrai_nom = class_names[qry_lbl[i]]
        pred_nom = class_names[preds[i]]
        statut = "Correct" if preds[i] == qry_lbl[i] else "False"
        print(f"Image {i+1} ({vrai_nom}) -> Prédit : {pred_nom} {statut}")