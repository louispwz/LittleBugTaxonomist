import os
import tarfile
import json
from PIL import Image
import clip
import torch
import random
import string
from Dataset_shrinker import dataset_shrinker
from Metadata_dataset import extract_metadata_from_tar


device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_image_zero_shot(image_file, candidate_labels, model, preprocess, device):
    image = Image.open(image_file).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    textes = [f"an insect belonging to the species {label}" for label in candidate_labels]
    texte_tokens = clip.tokenize(textes).to(device)
    with torch.no_grad():
        text_features = model.encode_text(texte_tokens)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_features @ text_features.T
    probs = logits.softmax(dim=-1)[0].cpu().numpy()
    idx = probs.argsort()[::-1]
    return [(candidate_labels[i], float(probs[i])) for i in idx]


def main(tar_path="Data/small_database.tar", metadata_path="Data/metadata_images.json"):
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Pas de fichier .tar {tar_path}")

    # si pas de metadata genere
    if not os.path.exists(metadata_path):
        extract_metadata_from_tar(tar_path=tar_path, out_json_path=metadata_path)

    # load des metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    bug_metadata = {}
    unique_candidate = set()
    for bug in meta:
        fold_num = bug.get("folder_number")
        gbif_info = bug.get("gbif")
        if gbif_info is not None:
            canonical_name = gbif_info.get("canonicalName")
            bug_metadata[fold_num] = canonical_name
            if canonical_name is not None:
                unique_candidate.add(canonical_name)

    unique_candidate = sorted(list(unique_candidate))
    if len(unique_candidate) == 0:
        raise RuntimeError("Pas de candidats dans les metadata")

    # modele CLIP ViT-L/14 
    model, preprocess = clip.load('ViT-L/14', device=device)
    model.to(device)
    # label hierarchique
    # prompts = [f"image of an insect belonging to the species {label}" for label in unique_candidate]
    # label plat
    # prompts = [f"image of an insect {label}" for label in unique_candidate]
    # texte constant
    # prompts = [f"image of an insect" for label in unique_candidate]
    # texte random
    prompts = ["".join(random.choices(string.ascii_uppercase + string.digits, k=10)) for label in unique_candidate]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # build de candidate_info et mapping metadata
    candidate_info = {}
    for bug in meta:
        gb = bug.get('gbif') or {}
        canonical = gb.get('canonicalName') if isinstance(gb, dict) else None
        if canonical and canonical not in candidate_info:
            candidate_info[canonical] = {
                'genus': gb.get('genus'),
                'family': gb.get('family')
            }

    # Iterattion sur tout les images, cherche top5
    
    results = []
    total = 0
    skipped = 0
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            total += 1
            try:
                with tar.extractfile(member) as f:
                    try:
                        image = Image.open(f).convert('RGB')
                    except Exception as e_img:
                        skipped += 1
                        print(f"ne peut pas ouvrir {member.name}: {e_img}")
                        continue
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        logits = 100.0 * image_features @ text_features.T
                        probs = logits.softmax(dim=-1)[0].cpu().numpy()
                        idxs = probs.argsort()[::-1][:5]
                        top5 = []
                        for i in idxs:
                            lab = unique_candidate[i]
                            prob = float(probs[i])
                            info = candidate_info.get(lab, {})
                            top5.append({
                                'label': lab,
                                'prob': prob,
                                'genus': info.get('genus'),
                                'family': info.get('family')
                            })

                parts = member.name.split('/')
                folder_number = None
                if len(parts) >= 2:
                    try:
                        folder_number = int(parts[-2])
                    except Exception:
                        folder_number = None

                real_name = bug_metadata.get(folder_number)
                real_info = candidate_info.get(real_name, {}) if real_name else {}
                results.append({
                    'archive_path': member.name,
                    'folder_number': folder_number,
                    'real_name': real_name,
                    'real_genus': real_info.get('genus'),
                    'real_family': real_info.get('family'),
                    'top5': top5
                })

                if total % 100 == 0:
                    print(f"{total} images, resultats: {len(results)}")

            except Exception as e:
                skipped += 1
                print(f"Warning: unexpected error on {member.name}: {e}")

    out_path = os.path.join(os.path.dirname(metadata_path) or '.', 'viT_predictions_random.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Finished. Processed={total}, skipped={skipped}, saved={len(results)} -> {out_path}")


if __name__ == '__main__':
    main()