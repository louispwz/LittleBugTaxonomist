import os
from PIL import Image
import open_clip
import torch
import torch.nn.functional as F

# verifie si GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charge le modele et le preprocess
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:imageomics/bioclip-2')
model.to(device)

# Tokenizer pour transformer du texte en embedding (on utilise pas ici mais c'est dans le code exemple)
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

# ----- préparation de la base de données few-shot -----




def few_shot_classification(model, support_images, support_labels, query_images):
    
    # 1. Extraction des Embeddings (Le modèle est gelé avec torch.no_grad())
    with torch.no_grad():
        # On encode les images de support (exemples)
        support_features = model.encode_image(support_images)
        support_features = F.normalize(support_features, dim=-1) # Normalisation importante !
        
        # On encode les images de query (à tester)
        query_features = model.encode_image(query_images)
        query_features = F.normalize(query_features, dim=-1)

    # 2. Création des Prototypes (Moyenne des features par classe)
    # On cherche à obtenir un vecteur unique par classe
    unique_classes = torch.unique(support_labels)
    prototypes = []
    
    for c in unique_classes:
        # On prend toutes les features appartenant à la classe 'c'
        class_features = support_features[support_labels == c]
        # On calcule la moyenne (le centroïde)
        mean_feature = class_features.mean(dim=0)
        # On re-normalise le prototype (bon pour la similarité cosinus)
        mean_feature = F.normalize(mean_feature, dim=-1)
        prototypes.append(mean_feature)
        
    prototypes = torch.stack(prototypes) # Tensor de taille [3, 768] (si dim=768)

    # 3. Classification (Plus proche voisin via Cosine Similarity)
    # Produit scalaire entre Query et Prototypes = Cosine Similarity (car normalisés)
    # [N_query, 768] x [768, N_classes] -> [N_query, N_classes]
    similarities = torch.matmul(query_features, prototypes.T)
    
    # La classe prédite est celle avec le score le plus élevé
    predictions = torch.argmax(similarities, dim=1)
    
    return predictions









if __name__ == "__main__":
    # image
    # image_path = "Code/Test_images/images_raw/00014500.jpg"
    # image_path = "Code/Test_images/images_raw/00013600.jpg"
    image_path = "Code/Test_images/images_raw/00000053.jpg"

    # Liste des labels candidats un peu mis au hasard mais on peut le changer si on sait ce qu'on fait
    candidate_labels = ["grass-hopper", "mantis religiosa", "bee", "dog", "butterfly", "Danaus plexippus", "beetle", "Animalia Arthropoda Insecta Lepidoptera Nymphalidae Danaus Plexippus"]

    # Pred zeroshot
    result = predict_image_few_shot(image_path, candidate_labels)
    print(f"Image: {os.path.basename(image_path)}  Predictions:")
    for label, p in result:
        print(f"   {label}: {p:.4f}")
    best_label, best_prob = result[0]
    print(f"Cette image est predite comme '{best_label}'")
    





