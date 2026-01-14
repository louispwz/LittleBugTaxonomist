import json
import pandas as pd

def json_to_csv(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extraction 
        rows = []
        for entry in data:
            metrics = entry.get('metrics', {}).get('top1_accuracy', {})
            species_metrics = metrics.get('species', {})
            genus_metrics = metrics.get('genus', {})
            timings = entry.get('timings', {})

            row = {
                'Technique': entry.get('technique', 'N/A'),
                'N Shot': timings.get('n_shot', 0),
                
                # Métriques
                'Précision Espèce': species_metrics.get('accuracy', ''),
                'Correct Espèce': species_metrics.get('correct', ''),
                'Précision Genre': genus_metrics.get('accuracy', ''),
                'Correct Genre': genus_metrics.get('correct', ''),
                
                # Temps
                'Chargement (s)': timings.get('loading_seconds', None),
                'Inférence (s)': timings.get('inference_seconds', None),
                'Formatage (s)': timings.get('formatting_seconds', None),
                'Sauvegarde (s)': timings.get('saving_seconds', None),
                'Temps Total (s)': timings.get('total_cycle_seconds', None)
            }
            rows.append(row)

        # DataFrame
        df = pd.DataFrame(rows)

        col_order = [
            'Technique', 
            'N Shot', 
            'Précision Espèce', 
            'Correct Espèce', 
            'Précision Genre', 
            'Correct Genre', 
            'Chargement (s)', 
            'Inférence (s)', 
            'Formatage (s)', 
            'Sauvegarde (s)', 
            'Temps Total (s)'
        ]
        
        df = df.reindex(columns=col_order)

        # Export vers CSV

        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')
        
        print(f"Succès ! Le fichier '{output_file}' a été créé (format Excel FR).")
        print(df.head())

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{input_file}' est introuvable.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    INPUT_FILENAME = "Results/COMPILED_results.json"
    OUTPUT_FILENAME = "Results/COMPILED_results.csv"
    
    json_to_csv(INPUT_FILENAME, OUTPUT_FILENAME)