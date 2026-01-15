import os
import multiprocessing
from PIL import Image, ImageOps
from tqdm import tqdm
import time

# --- CONFIGURATION ---
# Ton dossier source actuel
INPUT_DIR = os.path.join('Data', 'data_new_few_shot') 

# Le nouveau dossier optimisé qui sera créé
OUTPUT_DIR = os.path.join('Data', 'data_resized_224')

# Taille attendue par BioClip
TARGET_SIZE = (224, 224) 

# Qualité JPEG (90 est un très bon compromis)
JPEG_QUALITY = 90

def process_single_image(args):
    """
    Fonction exécutée par chaque coeur du CPU.
    Traite une image : resize + save.
    """
    src_path, dest_path = args
    
    # Si l'image existe déjà, on passe (utile si on relance le script)
    if os.path.exists(dest_path):
        return
        
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB") # Assure que c'est bien du 3 canaux (pas de transparence)
            
            # --- REDIMENSIONNEMENT INTELLIGENT ---
            # ImageOps.pad ajoute des bordures noires au lieu d'écraser l'image.
            # Cela préserve la morphologie exacte du carabe.
            img_processed = ImageOps.pad(img, TARGET_SIZE, color="black", centering=(0.5, 0.5))
            
            # Sauvegarde
            img_processed.save(dest_path, "JPEG", quality=JPEG_QUALITY)
            
    except Exception as e:
        print(f"Erreur sur {src_path}: {e}")

def main():
    print(f"--- DÉBUT DU TRAITEMENT ---")
    print(f"Source : {INPUT_DIR}")
    print(f"Destination : {OUTPUT_DIR}")
    print(f"Cibles : {TARGET_SIZE} pixels")
    
    # 1. Lister tous les fichiers et préparer la structure de dossiers
    tasks = []
    
    # On parcourt récursivement
    print("Analyse des fichiers...")
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')):
                # Chemin source complet
                src_path = os.path.join(root, file)
                
                # Calcul du chemin relatif (ex: Carabus_auratus/img1.jpg)
                relative_path = os.path.relpath(src_path, INPUT_DIR)
                
                # Chemin destination complet
                dest_path = os.path.join(OUTPUT_DIR, relative_path)
                
                # On s'assure que l'extension de destination est .jpg (pour standardiser)
                base, _ = os.path.splitext(dest_path)
                dest_path = base + ".jpg"
                
                # Créer le dossier parent s'il n'existe pas
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                tasks.append((src_path, dest_path))
    
    print(f"Nombre total d'images à traiter : {len(tasks)}")
    
    # 2. Lancer le traitement parallèle
    # On utilise le nombre de coeurs CPU - 1 pour laisser le système respirer un peu
    num_processes = max(1, os.cpu_count() - 1)
    print(f"Utilisation de {num_processes} coeurs CPU...")
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Tqdm affiche la barre de progression
        list(tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks), unit="img"))
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- TERMINÉ ---")
    print(f"Temps total : {duration:.2f} secondes")
    print(f"Vitesse moyenne : {len(tasks)/duration:.1f} images/seconde")
    print(f"Les données prêtes sont dans : {OUTPUT_DIR}")

if __name__ == '__main__':
    # Protection nécessaire pour multiprocessing sous Windows
    multiprocessing.freeze_support()
    main()