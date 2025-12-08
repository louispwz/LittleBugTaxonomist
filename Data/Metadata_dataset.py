import tarfile
import os
import json
import requests

def get_species_info(taxon_id, session):
    try:
        r = session.get(f"https://api.gbif.org/v1/species/{taxon_id}", timeout=1)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def main():
    tar_path = os.path.join("Data/database.tar")
    out_json_path = "Data/metadata_images.json"

    if not os.path.exists(tar_path):
        print("archive non trouvée")
        return

    session = requests.Session()
    results = []

    with tarfile.open(tar_path, "r:*") as tar:
        for m in tar.getmembers():
            if not m.isfile() or m.name.count("/") < 1:
                continue

            print(f"ficher {m}")
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

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Fini : {len(results)} dans le fichier {out_json_path}")


if __name__ == "__main__":
    main()
