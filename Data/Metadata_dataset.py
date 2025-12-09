import tarfile    # Pour lire les fichiers compressés .tar sans les extraire
import os         # Pour gérer les paths
import json       # Pour sauvegarder les résultats dans un fichier structuré
import requests   # Pour faire des appels web (API)


# ----- Fonction pour interroger l'API GBIF -----
def get_species_info(taxon_id, session):
    try:
        # Envoie une requête au site GBIF avec l'ID de l'espèce
        # timeout=1 signifie qu'on abandonne si le serveur met plus de 1sec à répondre
        r = session.get(f"https://api.gbif.org/v1/species/{taxon_id}", timeout=1)

        # Si le code retour est 200 (OK), on renvoie les données, sinon Rien (None)
        return r.json() if r.status_code == 200 else None
    except:
        # Si ça plante on renvoie None pour ne pas bloquer le script
        return None

# ----- Fonction principale (extraire et compiler données .tar + API) -----
def main():
    # Définition des chemins d'entrée (tar) et de sortie (json)
    tar_path = os.path.join("Data/database.tar")
    out_json_path = "Data/metadata_images.json"

    # Vérification de sécurité : si l'archive n'existe pas, on arrête tout
    if not os.path.exists(tar_path):
        print("archive non trouvée")
        return

    # Création d'une "session" (optimise la connexion pour faire plusieurs appels de suite)
    session = requests.Session()
    results = []       # Liste vide qui va stocker ce qu'on cherche

    # Ouverture de l'archive en mode lecture ("r:*")
    with tarfile.open(tar_path, "r:*") as tar:
        
        # On parcourt chaque élément (fichier ou dossier) dans l'archive
        for m in tar.getmembers():

            # Filtre : On ignore si ce n'est pas un fichier OU s'il est à la racine (pas dans un dossier)
            if not m.isfile() or m.name.count("/") < 1:
                continue

            print(f"ficher {m}")  # Affiche l'objet fichier en cours de traitement

            # _____ Extraction de l'ID ______
            # m.name = qqch comme "data/2435098/img01.jpg"
            # split("/") coupe le chemin en morceaux et [-2] prend l'avant-dernier morceau (le dossier parent)
            folder = m.name.split("/")[-2]

            try:
                # On essaie de convertir le nom du dossier en nombre entier (ID)
                taxon_id = int(folder)
            except ValueError:
                # Si le dossier s'appelle "photos" et pas "12345", ce n'est pas un ID valide
                taxon_id = None
                info = None
            else:
                # Si c'est bien un nombre, on appelle la fonction définie plus haut
                info = get_species_info(taxon_id, session)

            # On ajoute les infos trouvées dans notre liste de résultats
            results.append({
                "folder_number": taxon_id, # L'ID (ex 2435098)
                "archive_path": m.name,    # Le chemin interne (ex data/2435098/img01.jpg)
                "gbif": info               # Les autres ifo reçues de l'API
            })

# _______ Sauvegarde _______
    # On ouvre le fichier de sortie et on y écrit la liste results
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Fini : {len(results)} dans le fichier {out_json_path}")


if __name__ == "__main__":
    main()
