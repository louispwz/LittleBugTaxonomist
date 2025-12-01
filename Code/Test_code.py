import os
from PIL import Image
import open_clip
import torch
import torch.nn.functional as F

# 1. load model & preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2', pretrained=True)
model.to(device)
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

def predict_image(image_path, candidate_labels):
    # candidate_labels: list of strings, e.g. ['dog', 'cat', 'sparrow', ...]
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # tokenize text labels
    texts = [f"a photo of a {label}" for label in candidate_labels]
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # normalize (optional but often helps)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # compute similarity scores
    logits = 100.0 * image_features @ text_features.T  # scale as CLIP does
    probs = logits.softmax(dim=-1)[0]

    # return labels sorted by probability
    probs = probs.cpu().numpy()
    sorted_idx = probs.argsort()[::-1]
    return [(candidate_labels[i], float(probs[i])) for i in sorted_idx]

if __name__ == "__main__":
    # Example usage:
    image_folder = "/path/to/your/images"
    # Define your custom classes
    candidate_labels = ["Homo sapiens", "Felis catus", "Canis lupus", "Corvus corax", "Quercus robur"]  # etc.

    for fname in os.listdir(image_folder):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(image_folder, fname)
            result = predict_image(path, candidate_labels)
            print(f"Image: {fname} -> Predictions:")
            for label, p in result[:5]:
                print(f"   {label}: {p:.4f}")
            print("---")
