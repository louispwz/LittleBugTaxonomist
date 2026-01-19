import json
import pandas as pd
import os

def json_to_csv(input_file, output_file):
    try:
        # Vérification si le fichier existe
        if not os.path.exists(input_file):
            print(f"error no such file'{input_file}' ")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extraction 
        rows = []
        for entry in data:
            metrics = entry.get('metrics', {})

            top1 = metrics.get('top1_accuracy', {})
            top5 = metrics.get('top5_accuracy', {})
            
            timings = entry.get('timings', {})

            row = {
                'Technique': entry.get('technique', 'N/A'),
                'N Shot': timings.get('n_shot', 0),
                
                # top1
                'Top1 Acc Espèce': top1.get('species', {}).get('accuracy', ''),
                'Top1 Corr Espèce': top1.get('species', {}).get('correct', ''),
                'Top1 Acc Genre': top1.get('genus', {}).get('accuracy', ''),
                'Top1 Corr Genre': top1.get('genus', {}).get('correct', ''),

                # top5
                'Top5 Acc Espèce': top5.get('species', {}).get('accuracy', ''),
                'Top5 Corr Espèce': top5.get('species', {}).get('correct', ''),
                'Top5 Acc Genre': top5.get('genus', {}).get('accuracy', ''),
                'Top5 Corr Genre': top5.get('genus', {}).get('correct', ''),
                
                # TEMPS
                'Chargement (s)': timings.get('loading_seconds', ''),
                'Inférence (s)': timings.get('inference_seconds', ''),
                'Formatage (s)': timings.get('formatting_seconds', ''),
                'Sauvegarde (s)': timings.get('saving_seconds', ''),
                'Temps Total (s)': timings.get('total_cycle_seconds', '')
            }
            rows.append(row)

        # Création du DataFrame
        df = pd.DataFrame(rows)

        col_order = [
            'Technique', 
            'N Shot', 
            'Top1 Acc Espèce', 'Top1 Corr Espèce', 
            'Top1 Acc Genre', 'Top1 Corr Genre',
            'Top5 Acc Espèce', 'Top5 Corr Espèce', 
            'Top5 Acc Genre', 'Top5 Corr Genre',
            'Chargement (s)', 'Inférence (s)', 'Formatage (s)', 'Sauvegarde (s)', 'Temps Total (s)'
        ]
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')
        print(df.head())

    except Exception as e:
        print(f"error : {e}")

if __name__ == "__main__":
    INPUT_FILENAME = "Results/COMPILED_results.json" 
    OUTPUT_FILENAME = "Results/COMPILED_results.csv"
    
    json_to_csv(INPUT_FILENAME, OUTPUT_FILENAME)