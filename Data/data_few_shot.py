import json
import os
import random
import tarfile
import shutil

json_file_path = os.path.join('Data', 'metadata_images.json')
tar_file_path = 'database.tar'
output_base_dir = os.path.join('Data', 'data_few_shot')

NB_ESPECES = 5
NB_IMAGES_PAR_ESPECE = 5

def cration_dataset_few_shot():
    # On charge le JSON
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : le fichier {json_file_path} est introuvable.")
        return

    # On groupe les images par espèce (en mémoire uniquement)
    images_par_espece = {}
    print("Organisation des données en mémoire...")
    
    for entry in data:
        # Le chemin dans le JSON (ex: "database/1035167/d162s0016.jpg")
        # Important : Les fichiers TAR utilisent toujours des '/' même sous Windows
        archive_path = entry.get('archive_path') 
        
        try:
            espece_raw = entry['gbif']['species']
        except KeyError:
            continue 

        if archive_path and espece_raw:
            # Nettoyage du nom pour le dossier Windows/Linux
            nom_espece = espece_raw.replace(" ", "_").replace(".", "")
            
            if nom_espece not in images_par_espece:
                images_par_espece[nom_espece] = []
            
            images_par_espece[nom_espece].append(archive_path)

    # 3. Filtrer et Sélectionner les espèces
    especes_eligibles = [
        esp for esp, images in images_par_espece.items() 
        if len(images) >= NB_IMAGES_PAR_ESPECE
    ]

    print(f"Espèces éligibles : {len(especes_eligibles)}")

    if len(especes_eligibles) < NB_ESPECES:
        print("Pas assez d'espèces éligibles.")
        return

    especes_choisies = random.sample(especes_eligibles, NB_ESPECES)
    print(f"Espèces retenues : {especes_choisies}")

    # 4. Ouvrir l'archive TAR en lecture seule
    # 'r' pour un tar simple, 'r:gz' si c'est un tar.gz
    mode_ouverture = 'r' 
    if tar_file_path.endswith('.gz'):
        mode_ouverture = 'r:gz'

    print(f"Ouverture de l'archive {tar_file_path}...")
    try:
        with tarfile.open(tar_file_path, mode_ouverture) as tar:
            
            compteur_total = 0
            
            for espece in especes_choisies:
                print(f"--- Traitement : {espece} ---")
                
                # Créer le dossier destination physique
                target_dir = os.path.join(output_base_dir, espece)
                os.makedirs(target_dir, exist_ok=True)

                # Sélectionner 5 images au hasard
                images_cibles = random.sample(images_par_espece[espece], NB_IMAGES_PAR_ESPECE)

                for img_path_tar in images_cibles:
                    # img_path_tar est le chemin interne (ex: "database/123/img.jpg")
                    
                    try:
                        # On récupère l'info du fichier dans le TAR
                        member = tar.getmember(img_path_tar)
                        
                        # Si c'est bien un fichier (et pas un dossier)
                        if member.isfile():
                            # Extraction sous forme de flux (stream)
                            f_source = tar.extractfile(member)
                            
                            if f_source:
                                # Nom du fichier final
                                nom_fichier = os.path.basename(img_path_tar)
                                destination = os.path.join(target_dir, nom_fichier)
                                
                                # Écriture sur le disque
                                with open(destination, 'wb') as f_dest:
                                    shutil.copyfileobj(f_source, f_dest)
                                
                                compteur_total += 1
                                # On ferme le flux interne du fichier (bonne pratique)
                                f_source.close()
                            
                    except KeyError:
                        print(f"  [Erreur] Fichier introuvable dans le TAR : {img_path_tar}")
                    except Exception as e:
                        print(f"  [Erreur] Problème d'extraction : {e}")

            print(f"\nTerminé ! {compteur_total} images extraites dans '{output_base_dir}'.")

    except FileNotFoundError:
        print(f"Le fichier TAR '{tar_file_path}' n'existe pas.")

if __name__ == "__main__":
    creer_dataset_depuis_tar()