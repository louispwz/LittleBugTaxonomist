###########################
# Imports des librairies  #
###########################


import os
from PIL import Image
import open_clip
import torch
import torch.nn.functional
import json
from Dataset_shrinker import dataset_shrinker
from Data.Metadata_dataset import extract_metadata_from_tar


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
#Recuperation des candidats #
#############################











# Fonction pour pred en zero-shot
def predict_image_zero_shot(image_path, candidate_labels): # en entrée une image et des candidats potentiels
    
    # Prétraitement de l'image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Embedding de l'image
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # prompt pour chaque photo
    textes = [f"a photo of a {label}" for label in candidate_labels]
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
    
    # petit dataset
    dataset_small = dataset_shrinker(input_tar="Data/database.tar",n_folders=50,n_files=50, output_tar="Data/small_database.tar", seed = 123)
    # json du dataset
    metadata = extract_metadata_from_tar(tar_path="Data/small_database.tar", out_json_path="Data/small_metadata_images.json")

    
#     # Load the JSON file
# with open("Data/metadata_images.json", "r", encoding="utf-8") as f:
#     metadata = json.load(f)

# # Extract unique names
# unique_candidate = set()
# for bugs in metadata : 
#     bug_info = bugs.get()
# print(unique_names)


    # # Pred zeroshot
    # result = predict_image_zero_shot(image_path, candidate_labels)
    # print(f"Image: {os.path.basename(image_path)}  Predictions:")
    # for label, p in result:
    #     print(f"   {label}: {p:.4f}")
    # best_label, best_prob = result[0]
    # print(f"Cette image est predite comme '{best_label}'")
    


