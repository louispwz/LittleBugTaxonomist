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
LR = 1e-5  
# Few-shot config
FEW_SHOT_SHOTS = 50
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
            prompt = f"Kingdom: Animalia, Phylum: Arthropoda, Class: Insecta, Order: Coleoptera, Family: Carabidae, Genus: {item['gbif']['genus']}, Species: {item['gbif']['species']}"
            tokens = clip.tokenize([prompt], truncate=True)[0]
            return image, tokens, item['gbif']['species'], item['gbif']['genus']
        except Exception:
            return None


def run_few_shot(model, meta, train_data, val_data, tar_path, preprocess, n_shot=20):
    """Build prototypes from `train_data` using up to `n_shot` examples per species,
    then evaluate on `val_data` (both lists are metadata entries)."""
    model.eval()

    # group train examples by species
    species_to_items = {}
    for item in train_data:
        sp = item['gbif']['species']
        species_to_items.setdefault(sp, []).append(item)

    # only keep species that have at least one support example
    support_species = []
    support_images = []
    support_labels = []

    # open tar once
    tar = tarfile.open(tar_path, 'r')

    for sp_idx, (sp, items) in enumerate(sorted(species_to_items.items())):
        # take up to n_shot
        chosen = items[:n_shot]
        if len(chosen) == 0:
            continue
        # collect images
        imgs = []
        for it in chosen:
            try:
                member = tar.getmember(it['archive_path'])
                f = tar.extractfile(member)
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
                imgs.append(preprocess(img))
            except Exception:
                continue
        if len(imgs) == 0:
            continue
        # register species
        support_species.append(sp)
        support_images.extend(imgs)
        support_labels.extend([len(support_species)-1] * len(imgs))

    if len(support_images) == 0:
        print("No support images found for few-shot. Abort.")
        tar.close()
        return None, None

    support_tensor = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels, device=device)

    with torch.no_grad():
        supp_feats = model.encode_image(support_tensor)
        supp_feats = supp_feats / supp_feats.norm(dim=-1, keepdim=True)

    # compute prototypes
    prototypes = []
    for c in range(len(support_species)):
        mask = (support_labels == c)
        proto = supp_feats[mask].mean(dim=0)
        proto = proto / proto.norm()
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)

    # evaluate on val_data
    y_true_sp, y_pred_sp, y_true_ge, y_pred_ge = [], [], [], []

    # species -> genus mapping (only for supported species)
    species_to_genus = {s: next((m['gbif']['genus'] for m in meta if m.get('gbif') and m['gbif'].get('species') == s), None) for s in support_species}

    with torch.no_grad():
        for item in tqdm(val_data, desc=f"Few-shot {n_shot}-shot evaluation"):
            try:
                member = tar.getmember(item['archive_path'])
                f = tar.extractfile(member)
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
                inp = preprocess(img).unsqueeze(0).to(device)
            except Exception:
                continue

            img_feat = model.encode_image(inp)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            logits = (100.0 * img_feat @ prototypes.T).softmax(dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(logits))
            pred_sp = support_species[pred_idx]

            y_true_sp.append(item['gbif']['species'])
            y_pred_sp.append(pred_sp)
            y_true_ge.append(item['gbif']['genus'])

            # aggregate by genus
            gen_probs = {}
            for idx, p in enumerate(logits):
                g = species_to_genus.get(support_species[idx])
                if g is None: continue
                gen_probs[g] = gen_probs.get(g, 0) + p
            if len(gen_probs) > 0:
                y_pred_ge.append(max(gen_probs, key=gen_probs.get))
            else:
                y_pred_ge.append(None)

    tar.close()

    sp_acc = accuracy_score(y_true_sp, y_pred_sp) if len(y_true_sp) else 0.0
    ge_acc = accuracy_score(y_true_ge, y_pred_ge) if len(y_true_ge) else 0.0

    print(f"Few-shot results ({n_shot}-shot): Species acc={sp_acc:.2%}, Genus acc={ge_acc:.2%}")
    return sp_acc, ge_acc

def main():

    # Charge modèle viT clib B/32 mais il faudrait utiliser L/14 a terme
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # Charge les metadata de l'archive
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
    valid_data = [m for m in meta if m.get('gbif') and m['gbif'].get('species')]

    # espèces triées et mapping espèce et genre
    species_list = sorted({m['gbif']['species'] for m in valid_data})
    species_to_genus = {m['gbif']['species']: m['gbif']['genus'] for m in valid_data}

    # Build few-shot support/query sets per species (no full re-training)
    random.seed(123)
    species_items = {}
    for m in valid_data:
        sp = m['gbif']['species']
        species_items.setdefault(sp, []).append(m)

    train_support = []
    val_query = []
    for sp, items in species_items.items():
        random.shuffle(items)
        support = items[:FEW_SHOT_SHOTS]
        query = items[FEW_SHOT_SHOTS:]
        train_support.extend(support)
        val_query.extend(query)

    print(f"Support set: {len(train_support)} images across {len(species_items)} species; query set: {len(val_query)} images")

    # Run prototype-based few-shot evaluation and exit
    run_few_shot(model, meta, train_support, val_query, DATASET_TAR, preprocess, n_shot=FEW_SHOT_SHOTS)
    return

if __name__ == "__main__":
    main()
    
    # Fine tuning classique : 1.14% espece 10.6% genre
    # Few-shot results (20-shot): Species acc=16.82%, Genus acc=41.18%
    # Few-shot results (20-shot): Species acc=19.29%, Genus acc=49.18%
    # Few-shot results (30-shot): Species acc=23.60%, Genus acc=50.89%
    # Few-shot results (50-shot): Species acc=26.26%, Genus acc=52.67%