import json
import re
import os

# config
FILE_MAIN = 'Results/global_accuracy_results.json'
FILE_TIME_STD = 'Data/few_shot_timings_summary.json' 
FILE_TIME_NEW = 'Data/new_few_shot_timings_summary_freeze.json' 
OUTPUT_FILE = 'Results/COMPILED_results.json'

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"no such file '{filepath}'")
        return None

def main():
    # Chargement des données
    main_data = load_json(FILE_MAIN)
    timings_std_list = load_json(FILE_TIME_STD)
    timings_new_list = load_json(FILE_TIME_NEW)

    if not main_data or not timings_std_list or not timings_new_list:
        return

    # On transforme les listes en dictionnaires avec 'n_shot' comme clé
    timings_std_map = {item['n_shot']: item for item in timings_std_list}
    timings_new_map = {item['n_shot']: item for item in timings_new_list}

    #Boucle de fusion
    for entry in main_data:
        filename = entry.get('filename', '')
        
        # On utilise une ??Regex?? pour trouver le nombre de shots
        match = re.search(r'_(\d+)shot', filename)
        
        timing_data = None
        
        if match:
            n_shot = int(match.group(1))
            
            # pour savoir dans quel fichier de timing chercher
            if "new_few_shot" in filename:
                timing_data = timings_new_map.get(n_shot)
            elif "few_shot" in filename:
                timing_data = timings_std_map.get(n_shot)
        
        if timing_data:
            entry['timings'] = timing_data.copy()

    # save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(main_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()