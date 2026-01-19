import tarfile
import os
import json
import requests
import pdfplumber
import re




def build_duff_taxonomy_from_pdf(pdf_path):
    taxonomy = {}

    family = subfamily = tribe = subgenus = None

    family_re = re.compile(r"^Family\s+([A-Z]+IDAE)")
    subfamily_re = re.compile(r"^Subfamily\s+([A-Z]+INAE)")
    tribe_re = re.compile(r"^Tribe\s+([A-Z]+INI)")
    subgenus_re = re.compile(r"^Subgenus\s+([A-Z][a-zA-Z]+)")
    genus_re = re.compile(r"^([A-Z][a-z]+)\s")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()

                if m := family_re.match(line):
                    family = m.group(1)
                    subfamily = tribe = subgenus = None

                elif m := subfamily_re.match(line):
                    subfamily = m.group(1)
                    tribe = subgenus = None

                elif m := tribe_re.match(line):
                    tribe = m.group(1)
                    subgenus = None

                elif m := subgenus_re.match(line):
                    subgenus = m.group(1)

                elif m := genus_re.match(line):
                    genus = m.group(1)
                    taxonomy.setdefault(genus, {
                        "family": family,
                        "subfamily": subfamily,
                        "tribe": tribe,
                        "subgenus": subgenus
                    })

    return taxonomy

def get_species_info(taxon_id, session):
    try:
        r = session.get(f"https://api.gbif.org/v1/species/{taxon_id}", timeout=0.5)
        return r.json() if r.status_code == 200 else None
    except:
        return None
    
    
def enrich_with_duff_taxonomy(gbif_info, duff_taxonomy):
    if not gbif_info:
        return None

    genus = gbif_info.get("genus")
    if not genus:
        return None

    return duff_taxonomy.get(genus)



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
                
            duff_info = enrich_with_duff_taxonomy(info, duff_taxonomy)

            results.append({
        "folder_number": taxon_id,
        "archive_path": m.name,
        "gbif": info,
        "duff_2012": duff_info
    })

    if out_json_path is not None:
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Metadata extraite pour {len(results)} fichiers")
    return results



if __name__ == "__main__":
    
    extract_metadata_from_tar(tar_path="Data/database.tar",out_json_path="Data/metadata_images.json")
    duff_taxonomy = build_duff_taxonomy_from_pdf("../Data/Duffetal2012.pdf")


    