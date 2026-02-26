# LittleBugTaxonomist

**Hierarchical semantic information–guided insect recognition** Ce projet explore l'intégration de connaissances taxonomiques et sémantiques dans des modèles de vision (modèles de fondation) pour l'identification automatisée des insectes, avec un focus particulier sur les Carabidés.

---

## Présentation du Projet
L'objectif est de transformer des modèles de vision généralistes en outils capables de respecter la hiérarchie biologique (Espèce > Genre > Famille) dans sa vision d'une image. Le projet s'articule autour de trois axes :
1. **Défrichement méthodologique** : Tester la robustesse des modèles de fondation (BioClip2, CLIP) sur des tâches spécifiques.
2. **Évaluation hiérarchique** : Mesurer comment les erreurs se répartissent dans l'arbre taxonomique.
3. **Apprentissage efficace** : Comparer le Fine-tuning complet avec des approches Few-shot et Zero-shot.


---

## Structure du Dépôt
* **`/Code`** : Scripts principaux d'entraînement et d'inférence, l'intégralité des tests sont dans ce dossier
* **`/Data`** : Stockage des jeux de données (redimensionnées ou en `.tar`) et des métadonnées GBIF extraites
* **`/Results`** : Logs d'entraînement, prédictions JSON et statistiques de performance
* **`/Research`** : Comptes-rendus de réunions et notes théoriques
* **`/Bibliographie`** : Articles de référence principalement sur BioClip, ViT, apprentissage contrastif

---

##  Techniques (`/Code`)

### 1. Fine-Tuning de BioClip2 (`FINE_TUNING.py`)
Implémentation d'un entraînement en deux phases pour adapter le modèle sans perdre sa capacité de généralisation :
* **Phase de Warmup** : Entraînement de la tête de classification seule ($LR = 1e-3$).
* **Phase de Fine-tune** : Ajustement du backbone visuel avec un taux d'apprentissage très faible ($LR = 1e-6$). lourd [non finit]

### 2. Évaluation Few-Shot (`Test_code_few_shot.py`)
Utilisation de **prototypes** pour classer les insectes à partir de très peu d'exemples (1, 5, 10, 25, 50-shot) :
* Extraction des caractéristiques via BioClip2.
* Calcul de la similarité cosinus entre les images de test et la moyenne des images de support.

### 3. Approches Zero-Shot et ViT (`Test_code_zero_shot.py`, `Test_code_ViT.py`)
* Comparaison entre des prompts "plats" (nom d'espèce seul) et des **prompts hiérarchiques** intégrant la lignée complète (Famille, Genre).
* Tests comparatifs sur l'architecture Vision Transformer (ViT-L/14).

---

### Problématiques de Recherche
Le projet s'articule autour de trois questions :
1. **Organisation de l'espace latent** : Comment l'espace latent de BioClip2 est-il structuré et est-il performant pour distinguer les traits morphologiques des insectes spécifiquement ?
2. **Adaptabilité (Fine-tuning)** : Est-il possible d'adapter facilement BioClip2 à une tâche spécifique via du fine-tuning ou du few-shot ?
3. **Impact de la hiérarchie** : Quelle est l'influence réelle de l'intégration de la taxonomie sur les performances du modèle ? Est-ce qu'une structure hiérarchique améliore la précision par rapport à une classification avec un label plat ?

---

## Installation & Utilisation
1. **Dépendances** : "json", "open_clip", "os", "PIL", "pillow", "random", "scikit-learn", "tarfile", "time", "torch", "tqdm"
2. **Données** : Placer `metadata_images.json` et les images dans le dossier `Data/`.
