import torch
import clip
import json
import tarfile
import io
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
METADATA_PATH = 'Data/metadata_images.json'
DATASET_TAR = 'Data/small_database.tar'
BATCH_SIZE = 8 
ACCUMULATION_STEPS = 4
EPOCHS = 10
LR = 1e-5  # Augmenté pour compenser la lenteur du CPU

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)

class CarabidDataset(Dataset):
    def __init__(self, metadata_list, tar_path, preprocess):
        self.metadata = metadata_list
        self.tar_path = tar_path
        self.preprocess = preprocess
        self.tar = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.tar is None:
            self.tar = tarfile.open(self.tar_path, "r")
        item = self.metadata[idx]
        try:
            member = self.tar.getmember(item['archive_path'])
            f = self.tar.extractfile(member)
            image = self.preprocess(Image.open(io.BytesIO(f.read())))
            # Prompt hiérarchique pour tester l'impact taxonomique
            prompt = f"Kingdom: Animalia, Phylum: Arthropoda, Class: Insecta, Order: Coleoptera, Family: Carabidae, Genus: {item['gbif']['genus']}, Species: {item['gbif']['species']}"
            tokens = clip.tokenize([prompt], truncate=True)[0]
            return image, tokens, item['gbif']['species'], item['gbif']['genus']
        except Exception:
            return None

def validate(model, loader, species_list, species_to_genus):
    model.eval()
    all_prompts = [f"Kingdom: Animalia, Phylum: Arthropoda, Class: Insecta, Order: Coleoptera, Family: Carabidae, Genus: {species_to_genus[s]}, Species: {s}" for s in species_list]
    all_tokens = clip.tokenize(all_prompts).to(device)
    
    y_true_sp, y_pred_sp, y_true_ge, y_pred_ge = [], [], [], []
    
    with torch.no_grad():
        text_features = model.encode_text(all_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for batch in tqdm(loader, desc="Validation"):
            if batch is None: continue
            images, _, sp_names, ge_names = batch
            img_feat = model.encode_image(images.to(device))
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
            # Calcul de similarité cosinus
            probs = (100.0 * img_feat @ text_features.T).softmax(dim=-1).cpu().numpy()
            
            # Debug : Visualisation du Top 5 pour vérifier si le modèle "hésite"
            top5_idx = np.argsort(probs[0])[-5:][::-1]
            print(f"\nDEBUG Image 0: True={sp_names[0]} | Preds={[species_list[i] for i in top5_idx]}")
            
            for i, prob in enumerate(probs):
                pred_idx = np.argmax(prob)
                y_true_sp.append(sp_names[i])
                y_pred_sp.append(species_list[pred_idx])
                y_true_ge.append(ge_names[i])
                
                # [cite_start]Agrégation par genre (impact de la hiérarchie) [cite: 139]
                gen_probs = {}
                for idx, p in enumerate(prob):
                    g = species_to_genus[species_list[idx]]
                    gen_probs[g] = gen_probs.get(g, 0) + p
                y_pred_ge.append(max(gen_probs, key=gen_probs.get))

    return accuracy_score(y_true_sp, y_pred_sp), accuracy_score(y_true_ge, y_pred_ge)

def main():
    print(f"Device: {device.upper()}")
    # Utilisation de ViT-B/32 pour permettre l'exécution sur CPU
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
    
    # [cite_start]Identification à l'espèce requise pour l'étude [cite: 112]
    valid_data = [m for m in meta if m.get('gbif') and m['gbif'].get('species')]
    species_list = sorted(list(set([m['gbif']['species'] for m in valid_data])))
    species_to_genus = {m['gbif']['species']: m['gbif']['genus'] for m in valid_data}
    
    random.seed(123)
    random.shuffle(valid_data)
    # Split 50/20 conforme à Hansen et al. (2019) [cite_start][cite: 122]
    train_data = valid_data[:int(len(valid_data)*0.5)] 
    val_data = valid_data[int(len(valid_data)*0.5):int(len(valid_data)*0.7)] 

    train_loader = DataLoader(CarabidDataset(train_data, DATASET_TAR, preprocess), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(CarabidDataset(val_data, DATASET_TAR, preprocess), 
                            batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=device, enabled=(device == 'cuda'))

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Train E{epoch}")
        
        for i, batch in enumerate(pbar):
            if batch is None: continue
            images, tokens = batch[0].to(device), batch[1].to(device)
            
            with torch.amp.autocast(device_type=device):
                # Calcul de la perte contrastive (image <-> texte)
                logits_img, logits_txt = model(images, tokens)
                ground_truth = torch.arange(len(images), device=device)
                
                # Perte symétrique
                loss = (loss_fn(logits_img, ground_truth) + loss_fn(logits_txt, ground_truth)) / 2
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
        
        # [cite_start]Validation : Comparaison avec les scores de Hansen (51.9% Sp, 74.9% Gen) [cite: 32]
        acc_sp, acc_ge = validate(model, val_loader, species_list, species_to_genus)
        print(f"\nRESULTATS E{epoch} - Accuracy Espèce: {acc_sp:.2%}, Accuracy Genre: {acc_ge:.2%}")
        torch.save(model.state_dict(), f"carabid_clip_e{epoch}.pt")

if __name__ == "__main__":
    main()