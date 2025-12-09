import tarfile
import os
import json
import requests



def get_species_info(taxon_id, session):
    try:
        r = session.get(f"https://api.gbif.org/v1/species/{taxon_id}", timeout=0.5)
        return r.json() if r.status_code == 200 else None
    except:
        return None



def extract_metadata_from_tar(tar_path, out_json_path=None):

    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    session = requests.Session()
    results = []

    with tarfile.open(tar_path, "r:*") as tar:
        
        file_counter = 0
        
        for m in tar.getmembers():
            
            if not m.isfile() or m.name.count("/") < 1:
                continue
            
            file_counter += 1
            if file_counter % 500 == 0:
                print(f"{file_counter} fichiers parsés")

            folder = m.name.split("/")[-2]
            try:
                taxon_id = int(folder)
            except ValueError:
                taxon_id = None
                info = None
            else:
                info = get_species_info(taxon_id, session)

            results.append({
                "folder_number": taxon_id,
                "archive_path": m.name,
                "gbif": info
            })

    if out_json_path is not None:
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Metadata extraite pour {len(results)} fichiers")
    return results



if __name__ == "__main__":
    extract_metadata_from_tar(tar_path="Data/database.tar",out_json_path="Data/metadata_images.json")
