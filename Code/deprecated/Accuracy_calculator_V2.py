import json
import os
from typing import Tuple, Optional, List, Dict



def compute_top1_accuracy(json_path: str, level: str = "species") -> Tuple[float, int, int]:
    if level not in ("species", "genus"):
        raise ValueError("level doit être 'species' ou 'genus'")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct = 0
    total = 0
    key_real = "real_name" if level == "species" else "real_genus"
    key_pred = "label" if level == "species" else "genus"

    for entry in data:
        real = entry.get(key_real)
        top5 = entry.get("top5")
        if not real or not top5 or not isinstance(top5, list) or len(top5) == 0:
            continue
        top = top5[0] or {}
        pred = top.get(key_pred)
        if not pred:
            continue

        try:
            real_norm = str(real).strip().lower() if real is not None else ""
        except Exception:
            real_norm = ""
        try:
            pred_norm = str(pred).strip().lower() if pred is not None else ""
        except Exception:
            pred_norm = ""

        if real_norm == pred_norm:
            correct += 1
        total += 1

    accuracy = (correct / total) if total > 0 else 0.0
    return accuracy, correct, total

def accuracy_name(json_path: str) -> Tuple[float, int, int]:
    return compute_top1_accuracy(json_path, level="species")

def accuracy_genus(json_path: str) -> Tuple[float, int, int]:
    return compute_top1_accuracy(json_path, level="genus")



# C4EST ICI LA V2

def process_batch_accuracies(experiments_map: Dict[str, List[str]], output_json_path: str) -> None:
    """
    experiments_map: Dictionnaire { "TITRE DU GROUPE": [liste_fichiers]}
    """
    all_results = []


    # boucle sur chaque groupe
    for group_title, file_list in experiments_map.items():
        
        # boucle sur les fichiers de ce groupe
        for path in file_list:
            if not os.path.exists(path):
                print(f"no such file{path}")
                continue
            
            # Calcul
            acc_species, corr_species, tot_species = accuracy_name(path)
            acc_genus, corr_genus, tot_genus = accuracy_genus(path)

            #Formatage JSON

            file_result = {
                "technique": group_title, 
                "filename": path,
                "metrics": {
                    "top1_accuracy": {
                        "species": {
                            "accuracy": f"{acc_species*100:.2f}%",
                            "correct": f"{corr_species}/{tot_species}"
                        },
                        "genus": {
                            "accuracy": f"{acc_genus*100:.2f}%",
                            "correct": f"{corr_genus}/{tot_genus}",
                        }
                    }
                }
            }
            all_results.append(file_result)

    # Sauvegarde
    try:
        with open(output_json_path, "w", encoding="utf-8") as f_out:
            json.dump(all_results, f_out, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"\nError {e}")


if __name__ == "__main__":
    
    output_file = "Results/global_accuracy_results.json"

    
    experiments_config = {
        "PREDICTIONS ZERO SHOT": [
            "Data/zero_shot_predictions.json"
        ],
        "PREDICTIONS FEW SHOT sur 4 especes": [
            "Data/few_shot_predictions_1shot.json",
            "Data/few_shot_predictions_5shot.json",
            "Data/few_shot_predictions_10shot.json",
            "Data/few_shot_predictions_25shot.json",
            "Data/few_shot_predictions_50shot.json"
        ],
        
        "PREDICTIONS FEW SHOT all dataset": [
            "Data/new_few_shot_predictions_1shot_freeze.json",
            "Data/new_few_shot_predictions_5shot_freeze.json",
            "Data/new_few_shot_predictions_10shot_freeze.json",
            "Data/new_few_shot_predictions_25shot_freeze.json",
            "Data/new_few_shot_predictions_50shot_freeze.json"
        ]
    }

    process_batch_accuracies(experiments_config, output_file)