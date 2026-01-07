import json
import os
import tarfile
import shutil

json_file_path = os.path.join('Data', 'metadata_images.json')
tar_file_path = 'Data/database.tar'
output_base_dir = os.path.join('Data', 'data_new_few_shot')


def extract_all_dataset():
    
    # Création du dossier de destination
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    
    os.makedirs(output_base_dir, exist_ok=True)

    # Chargement des métadonnées
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error file {json_file_path} not found")
        return

    # On crée un dictionnaire de correspondance "chemin/dans/tar" vers "Nom_Espece"
    path_to_species = {}
    
    ignored_count = 0
    
    for entry in data:
        path = entry.get('archive_path')
        gbif_info = entry.get('gbif')

        if path and gbif_info:
            espece_raw = gbif_info.get('species')
            if espece_raw:
                # formatage propre du nom de dossier "Genus_species"
                nom_folder = espece_raw.replace(" ", "_").replace(".", "")
                path_to_species[path] = nom_folder
            else:
                ignored_count += 1
        else:
            ignored_count += 1

    print(f"{len(path_to_species)} file to extract")

    # Extraction linéaire
    mode = 'r' if tar_file_path.endswith('.tar') else 'r:gz'

    extracted_count = 0

    try:
        with tarfile.open(tar_file_path, mode) as tar:
            for member in tar:
                # On vérifie si c'est un fichier et s'il est dans notre liste cible
                if member.isfile() and member.name in path_to_species:
                    
                    # On récupère le nom de l'espèce associé
                    species_folder = path_to_species[member.name]
                    
                    # On crée le dossier de l'espèce s'il n'existe pas
                    target_dir = os.path.join(output_base_dir, species_folder)
                    os.makedirs(target_dir, exist_ok=True)

                    # On extrait le fichier
                    source = tar.extractfile(member)
                    if source:
                        filename = os.path.basename(member.name)
                        destination_path = os.path.join(target_dir, filename)
                        
                        with open(destination_path, "wb") as dest:
                            shutil.copyfileobj(source, dest) # copie le flux de fichier vers sa destination finale
                        
                        extracted_count += 1

        print(f"\nCompleted")
        print(f"{extracted_count} images extracted")

    except FileNotFoundError:
        print(f"L'archive {tar_file_path} est introuvable.")
    except Exception as e:
        print(f"Error {e}")

if __name__ == "__main__":
    extract_all_dataset()