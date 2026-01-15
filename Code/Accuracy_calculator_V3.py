import json
import os
from typing import Tuple, Optional, List, Dict

def normalize_text(text) -> str:
    try:
        return str(text).strip().lower() if text is not None else ""
    except Exception:
        return ""

def compute_accuracies(json_path: str, level: str = "species") -> Tuple[int, int, int]:
    """
    Retourne (correct_top1, correct_top5, total)
    """
    if level not in ("species", "genus"):
        raise ValueError("level doit être 'species' ou 'genus'")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    key_real = "real_name" if level == "species" else "real_genus"
    key_pred = "label" if level == "species" else "genus"

    for entry in data:
        real_raw = entry.get(key_real)
        top5_list = entry.get("top5")
        
        if not real_raw or not top5_list or not isinstance(top5_list, list) or len(top5_list) == 0:
            continue

        real_norm = normalize_text(real_raw)
        if not real_norm:
            continue


        # 
        top1_entry = top5_list[0] or {}
        pred_top1_norm = normalize_text(top1_entry.get(key_pred))
        
        if real_norm == pred_top1_norm:
            correct_top1 += 1

        # On vérifie si la vraie valeur est présente n'importe où dans la liste top5
        found_in_top5 = False
        for candidate in top5_list:
            pred_candidate_norm = normalize_text(candidate.get(key_pred))
            if real_norm == pred_candidate_norm:
                found_in_top5 = True
                break 
        
        if found_in_top5:
            correct_top5 += 1

        total += 1

    return correct_top1, correct_top5, total


def process_batch_accuracies(experiments_map: Dict[str, List[str]], output_json_path: str) -> None:
    """
    experiments map: Dictionnaire { "TITRE DU GROUPE": [liste_fichiers]}
    """
    all_results = []

    # boucle sur chaque groupe
    for group_title, file_list in experiments_map.items():
        
        # boucle sur les fichiers de ce groupe
        for path in file_list:
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            
            # Calcul Species (Top1 et Top5)
            c1_sp, c5_sp, tot_sp = compute_accuracies(path, level="species")
            acc1_sp = (c1_sp / tot_sp) if tot_sp > 0 else 0.0
            acc5_sp = (c5_sp / tot_sp) if tot_sp > 0 else 0.0

            # Calcul Genus (Top1 et Top5)
            c1_ge, c5_ge, tot_ge = compute_accuracies(path, level="genus")
            acc1_ge = (c1_ge / tot_ge) if tot_ge > 0 else 0.0
            acc5_ge = (c5_ge / tot_ge) if tot_ge > 0 else 0.0

            # Formatage JSON enrichi
            file_result = {
                "technique": group_title, 
                "filename": path,
                "metrics": {
                    "top1_accuracy": {
                        "species": {
                            "accuracy": f"{acc1_sp*100:.2f}%",
                            "correct": f"{c1_sp}/{tot_sp}"
                        },
                        "genus": {
                            "accuracy": f"{acc1_ge*100:.2f}%",
                            "correct": f"{c1_ge}/{tot_ge}",
                        }
                    },
                    "top5_accuracy": {
                        "species": {
                            "accuracy": f"{acc5_sp*100:.2f}%",
                            "correct": f"{c5_sp}/{tot_sp}"
                        },
                        "genus": {
                            "accuracy": f"{acc5_ge*100:.2f}%",
                            "correct": f"{c5_ge}/{tot_ge}",
                        }
                    }
                }
            }
            all_results.append(file_result)





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