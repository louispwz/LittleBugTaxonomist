import tarfile
import os
import json
import requests

def get_species_name(folder_id, session):
    # recup le json des especes par gbif
    try:
        r = session.get(f"https://api.gbif.org/v1/species/{folder_id}/name", timeout=0)
        return r.json() 
    except Exception:
        return None

def main():
    tar_path = os.path.join(os.path.dirname(__file__), "Test_images", "database.tar")
    out_path = "Code/metadata_images.json"

    if not os.path.exists(tar_path):
        print("Archive introuvable :", tar_path)
        return

    session = requests.Session()
    results = []

    with tarfile.open(tar_path, "r:*") as tar:
        for m in tar.getmembers():
            if not m.isfile() or m.name.count("/") < 1:
                continue
            
            folder = m.name.split("/")[-2]
            try:
                folder_id = int(folder)
                gbif_data = get_species_name(folder_id, session)
            except ValueError:
                folder_id = None
                gbif_data = None

            results.append({
                "name": gbif_data.get("scientificName") if gbif_data else None,
                "folder_number": folder_id,
                "archive_path": m.name,
                "gbif": gbif_data
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Fini : {len(results)} dans {out_path}")

if __name__ == "__main__":
    main()