import json
import os
import random
import tarfile
import shutil

json_file_path = os.path.join('Data', 'metadata_images.json')
tar_file_path = 'Data/database.tar'
output_base_dir = os.path.join('Data', 'data_few_shot')

ESPECES_CIBLES = [
    "Agonum nigrum",
    "Elaphropus convexus",
    "Bembidion clarkei",
    "Elaphropus walkerianus"
]

NB_IMAGES_PAR_ESPECE = 25

SEED_VALUE = 42 

def create_5shot_dataset():
    
    # Netoyage 
    if os.path.exists(output_base_dir):
        # On liste tout ce qu'il y a dans le dossier
        for filename in os.listdir(output_base_dir):
            file_path = os.path.join(output_base_dir, filename)
            try:
                # Si c'est un fichier on supprime
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Si c'est un dossier on supprime tout l'arbre
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error with {file_path} : {e}")
                return
    
    
    random.seed(SEED_VALUE)

    # Préparation des noms cibles
    cibles_nettoyees = set(nom.replace(" ", "_").replace(".", "") for nom in ESPECES_CIBLES)
    images_par_espece = {nom: [] for nom in cibles_nettoyees}

    # On charge le JSON
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error : file {json_file_path} not found")
        return

    # Remplissage du dictionnaire
    ignored_count = 0
    for entry in data:
        path = entry.get('archive_path')
        
        gbif_info = entry.get('gbif')

        if not gbif_info:
            continue

        espece_raw = gbif_info.get('species')

        if path and espece_raw:
            nom_propre = espece_raw.replace(" ", "_").replace(".", "")
            if nom_propre in cibles_nettoyees:
                images_par_espece[nom_propre].append(path)
        else:
            ignored_count += 1

    print(f"Finish : {ignored_count} entries ignored (missing path or species)")
    # Extraction depuis le TAR
    mode = 'r' if tar_file_path.endswith('.tar') else 'r:gz'

    try:
        with tarfile.open(tar_file_path, mode) as tar:
            compteur_global = 0

            for espece_nom in cibles_nettoyees:
                images_dispos = images_par_espece[espece_nom]
                nb_dispo = len(images_dispos)
                
                print(f"Species : {espece_nom} ({nb_dispo} available images)")

                if nb_dispo < NB_IMAGES_PAR_ESPECE:
                    selection = images_dispos 
                else:
                    # On trie la liste avant de piocher au hasard car si l'ordre de lecture du JSON change, l'aléatoire change aussi
                    # Le tri garantit que la liste est TOUJOURS dans le même ordre avant le tirage.
                    images_dispos.sort() 
                    
                    selection = random.sample(images_dispos, NB_IMAGES_PAR_ESPECE)

                # Extraction physique
                target_dir = os.path.join(output_base_dir, espece_nom)
                os.makedirs(target_dir, exist_ok=True)

                for img_path_tar in selection:
                    try:
                        member = tar.getmember(img_path_tar)
                        if member.isfile():
                            f_source = tar.extractfile(member)
                            if f_source:
                                nom_fichier = os.path.basename(img_path_tar)
                                dest_path = os.path.join(target_dir, nom_fichier)
                                
                                with open(dest_path, 'wb') as f_dest:
                                    shutil.copyfileobj(f_source, f_dest)
                                
                                compteur_global += 1
                                f_source.close()
                    except Exception as e:
                        print(f"  Error extraction : {e}")

            print(f"\nFinish : {compteur_global} extracted images")

    except FileNotFoundError:
        print(f"Archive {tar_file_path} not found.")

if __name__ == "__main__":
    create_5shot_dataset()