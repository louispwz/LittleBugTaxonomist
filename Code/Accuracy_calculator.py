import json
from typing import Tuple, Optional
DEFAULT_JSON = "zero_shot_predictions.json"

def compute_top1_accuracy(json_path: str, level: str = "species") -> Tuple[float, int, int]:
	"""
	Calcule la précision top-1.
	- level="species": compare real_name vs top5[0]['label']
	- level="genus":   compare real_genus vs top5[0]['genus']
	Renvoie (accuracy_fraction, correct_count, total_valid).
	Les entrées invalides sont ignorées.
	"""
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
			# données réelles ou prédiction manquantes -> on ignore l'entrée
			continue
		top = top5[0] or {}
		pred = top.get(key_pred)
		if not pred:
			# pas de prédiction top-1 pour ce niveau -> ignorer
			continue

		# Normalisation inline (None-safe) au lieu d'appeler une fonction séparée
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
	"""Précision top-1 pour le nom d'espèce (real_name vs top5[0]['label'])."""
	return compute_top1_accuracy(json_path, level="species")

def accuracy_genus(json_path: str) -> Tuple[float, int, int]:
	"""Précision top-1 pour le genre (real_genus vs top5[0]['genus'])."""
	return compute_top1_accuracy(json_path, level="genus")

def _print_result(name: str, result: Tuple[float, int, int]) -> None:
	accuracy, correct, total = result
	print(f"{name}: {correct}/{total} corrects — précision = {accuracy:.4f} ({accuracy*100:.2f}%)")

def print_accuracies(json_path: str, only: Optional[str] = None) -> None:
	"""
	Calculer et afficher les précisions top-1 directement depuis Python.
	- json_path: chemin vers zero_shot_predictions.json
	- only: None (affiche espèce+genre), "species" (seulement espèce) ou "genus" (seulement genre)

	Exemple d'utilisation depuis un autre script / REPL:
		from Accuracy_calculator import print_accuracies
		print_accuracies("/chemin/vers/zero_shot_predictions.json")
	"""
	if only in (None, "species"):
		res_species = accuracy_name(json_path)
		_print_result("Précision top-1 (espèce)", res_species)
	if only in (None, "genus"):
		res_genus = accuracy_genus(json_path)
		_print_result("Précision top-1 (genre)", res_genus)

if __name__ == "__main__":
	# Exécution simple : demande le chemin (avec valeur par défaut) et affiche les résultats.
	path = "Data/zero_shot_predictions.json"
	print_accuracies(path)
