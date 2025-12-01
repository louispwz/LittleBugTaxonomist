import os
from PIL import Image
import open_clip
import torch
import torch.nn.functional as F

# verifie si GPU disponible (mais moi je n'ai pas)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charge le modele et le preprocess
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:imageomics/bioclip-2')
model.to(device)

# Tokenizer pour transformer du texte en embedding (on utilise pas ici mais c'est dans le code exemple)
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')


# Fonction pour pred en zero-shot
def predict_image_zero_shot(image_path, candidate_labels): # en entrée une image et des candidats potentiels
    
    # Prétraitement de l'image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Embedding de l'image
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # prompt pour chaque photo
    texts = [f"a photo of a {label}" for label in candidate_labels]
    text_tokens = tokenizer(texts).to(device)

    # Embedding du texte
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # norm
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cos sim et softmax pour obtenir des probas pour chauqe candidats
    logits = 100.0 * image_features @ text_features.T
    probs = logits.softmax(dim=-1)[0]

    # Trie les labels par proba decroissante (meilleur en premier)
    probs = probs.cpu().numpy()
    sorted_idx = probs.argsort()[::-1]
    return [(candidate_labels[i], float(probs[i])) for i in sorted_idx]





if __name__ == "__main__":
    # image
    # image_path = "Code/Test_images/images_raw/00014904.jpg"
    image_path = "Code/Test_images/images_raw/00013600.jpg"
    # image_path = "Code/Test_images/images_raw/00000053.jpg"

    # Liste des labels candidats un peu mis au hasard mais on peut le changer si on sait ce qu'on fait
    candidate_labels = ["grass-hopper", "mantis religiosa", "bee", "dog", "butterfly", "Danaus plexippus", "beetle"]

    # Pred zeroshot
    result = predict_image_zero_shot(image_path, candidate_labels)
    print(f"Image: {os.path.basename(image_path)}  Predictions:")
    for label, p in result:
        print(f"   {label}: {p:.4f}")
    best_label, best_prob = result[0]
    print(f"Cette image est predite comme '{best_label}'")
    