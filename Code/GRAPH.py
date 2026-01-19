import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generer_graphiques(input_file):
    output_dir = os.path.dirname(input_file) 

    try:
        df = pd.read_csv(input_file, sep=';')
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{input_file}' est introuvable.")
        return

    # Nettoyage des données
    cols_percent = ['Top1 Acc Espèce', 'Top1 Acc Genre', 'Top5 Acc Espèce', 'Top5 Acc Genre']
    
    for col in cols_percent:
        if col in df.columns and df[col].dtype == 'object': 
            df[col] = df[col].str.replace('%', '').astype(float)

    # config
    sns.set_theme(style="whitegrid")
    

    # Graphique 1 evolution précision espece top1
    plt.figure(figsize=(13, 6))
    sns.scatterplot(
        data=df, 
        x="N Shot", 
        y="Top1 Acc Espèce", 
        hue="Technique",    
        style="Technique",  
        s=120,              
        palette="viridis"
    )
    plt.title("Évolution de la précision au rang ESPECE selon le nombre de shots - top1", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Nombre de Shots (N-Shot)")
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    filename1 = os.path.join(output_dir, "Graphique_1_Acc_espece_top1.png")
    plt.savefig(filename1, dpi=300)
    plt.close()

    # Graphique 1.5 evolution précision epece top5
    plt.figure(figsize=(13, 6))
    sns.scatterplot(
        data=df, 
        x="N Shot", 
        y="Top5 Acc Espèce", 
        hue="Technique",    
        style="Technique",  
        s=120,
        palette="viridis"
    )
    plt.title("Évolution de la précision au rang ESPECE selon le nombre de shots - top5", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Nombre de Shots (N-Shot)")
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    filename1_5 = os.path.join(output_dir, "Graphique_1.5_Acc_espece_top5.png")
    plt.savefig(filename1_5, dpi=300)
    plt.close()
    
    
    # Graphique 2 evolution précision genre top1
    plt.figure(figsize=(13, 6))
    sns.scatterplot(
        data=df, 
        x="N Shot", 
        y="Top1 Acc Genre", 
        hue="Technique",
        style="Technique",  
        s=120,
        palette="viridis"
    )
    plt.title("Évolution de la précision au rang Genre selon le nombre de shots - top1", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Nombre de Shots (N-Shot)")
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    filename2 = os.path.join(output_dir, "Graphique_2_Acc_genre_top1.png")
    plt.savefig(filename2, dpi=300)
    plt.close()
    
    # Graphique 2,5 evolution précision genre top5
    plt.figure(figsize=(13, 6))
    sns.scatterplot(
        data=df, 
        x="N Shot", 
        y="Top5 Acc Genre", 
        hue="Technique",
        style="Technique",  
        s=120,
        palette="viridis"
    )
    plt.title("Évolution de la précision au rang Genre selon le nombre de shots - top5", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Nombre de Shots (N-Shot)")
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    filename2 = os.path.join(output_dir, "Graphique_2,5_Acc_genre_top5.png")
    plt.savefig(filename2, dpi=300)
    plt.close()
    
    # Graphique 3 Espèce vs Genre all dataset top1
    technique_name = "PREDICTIONS FEW SHOT all dataset"
    df_subset = df[df['Technique'] == technique_name]

    if not df_subset.empty:
        plt.figure(figsize=(13, 6))
        # Formatage pour Seaborn
        df_melted = df_subset.melt(
            id_vars=['N Shot'], 
            value_vars=['Top1 Acc Espèce', 'Top1 Acc Genre'], 
            var_name='Niveau Taxonomique', 
            value_name='Précision'
        )
        
        df_melted['Niveau Taxonomique'] = df_melted['Niveau Taxonomique'].replace({
            'Top1 Acc Espèce': 'Espèce',
            'Top1 Acc Genre': 'Genre'
        })
        
        sns.barplot(
            data=df_melted,
            x="N Shot",
            y="Précision",
            hue="Niveau Taxonomique",
            palette="viridis"
        )
        plt.title(f"Comparaison precision Espèce vs Genre - top1 - {technique_name}", fontsize=14)
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        filename3 = os.path.join(output_dir, "Graphique_3_EspeceGenre_alldataset_top1.png")
        plt.savefig(filename3, dpi=300)
        plt.close()

# Graphique 3,5 Espèce vs Genre all dataset top5
    technique_name = "PREDICTIONS FEW SHOT all dataset"
    df_subset = df[df['Technique'] == technique_name]

    if not df_subset.empty:
        plt.figure(figsize=(13, 6))
        # Formatage pour Seaborn
        df_melted = df_subset.melt(
            id_vars=['N Shot'], 
            value_vars=['Top5 Acc Espèce', 'Top5 Acc Genre'], 
            var_name='Niveau Taxonomique', 
            value_name='Précision'
        )
        
        df_melted['Niveau Taxonomique'] = df_melted['Niveau Taxonomique'].replace({
            'Top5 Acc Espèce': 'Espèce',
            'Top5 Acc Genre': 'Genre'
        })
        
        sns.barplot(
            data=df_melted,
            x="N Shot",
            y="Précision",
            hue="Niveau Taxonomique",
            palette="viridis"
        )
        plt.title(f"Comparaison precision Espèce vs Genre - top5 - {technique_name}", fontsize=14)
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        filename3 = os.path.join(output_dir, "Graphique_3,5_EspeceGenre_alldataset_top5.png")
        plt.savefig(filename3, dpi=300)
        plt.close()
        
    # Graphique 4 Espèce vs Genre 4especes top1
    technique_name = "PREDICTIONS FEW SHOT sur 4 especes"
    df_subset = df[df['Technique'] == technique_name]

    if not df_subset.empty:
        plt.figure(figsize=(13, 6))
        df_melted = df_subset.melt(
            id_vars=['N Shot'], 
            value_vars=['Top1 Acc Espèce', 'Top1 Acc Genre'], 
            var_name='Niveau Taxonomique', 
            value_name='Précision'
        )
        
        df_melted['Niveau Taxonomique'] = df_melted['Niveau Taxonomique'].replace({
            'Top1 Acc Espèce': 'Espèce',
            'Top1 Acc Genre': 'Genre'
        })
        
        sns.barplot(
            data=df_melted,
            x="N Shot",
            y="Précision",
            hue="Niveau Taxonomique",
            palette="viridis"
        )
        plt.title(f"Comparaison precision Espèce vs Genre - top1- {technique_name}", fontsize=14)
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        filename4 = os.path.join(output_dir, "Graphique_4_EspeceGenre_4especes_top1.png")
        plt.savefig(filename4, dpi=300)
        plt.close()
        
# Graphique 4,5 Espèce vs Genre 4especes top5
    technique_name = "PREDICTIONS FEW SHOT sur 4 especes"
    df_subset = df[df['Technique'] == technique_name]

    if not df_subset.empty:
        plt.figure(figsize=(13, 6))
        df_melted = df_subset.melt(
            id_vars=['N Shot'], 
            value_vars=['Top5 Acc Espèce', 'Top5 Acc Genre'], 
            var_name='Niveau Taxonomique', 
            value_name='Précision'
        )
        
        df_melted['Niveau Taxonomique'] = df_melted['Niveau Taxonomique'].replace({
            'Top5 Acc Espèce': 'Espèce',
            'Top5 Acc Genre': 'Genre'
        })
        
        sns.barplot(
            data=df_melted,
            x="N Shot",
            y="Précision",
            hue="Niveau Taxonomique",
            palette="viridis"
        )
        plt.title(f"Comparaison precision Espèce vs Genre - top5- {technique_name}", fontsize=14)
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        filename4 = os.path.join(output_dir, "Graphique_4,5_EspeceGenre_4especes_top5.png")
        plt.savefig(filename4, dpi=300)
        plt.close()
        
        
    # Graphique 5 temps d'inférence
    df_time = df[df['N Shot'] > 0].copy() 
    
    if not df_time.empty and 'Inférence (s)' in df_time.columns:
        plt.figure(figsize=(13, 6))
        sns.barplot(
            data=df_time,
            x="N Shot",
            y="Inférence (s)",
            hue="Technique",
            palette="viridis"
        )
        plt.title("Temps d'inférence selon le N-Shot", fontsize=14)
        plt.ylabel("Temps (s)")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        
        filename5 = os.path.join(output_dir, "Graphique_5_temps_inference.png")
        plt.savefig(filename5, dpi=300)
        plt.close()
    
    
        # Graphique 5bis temps d'inférence log
        plt.figure(figsize=(13, 6))
        sns.barplot(
            data=df_time,
            x="N Shot",
            y="Inférence (s)",
            hue="Technique",
            palette="viridis"
        )
        plt.title("Temps d'inférence selon le N-Shot (Log)", fontsize=14)
        plt.ylabel("Temps (s)")
        plt.yscale("log")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()

        filename5bis = os.path.join(output_dir, "Graphique_5bis_temps_inference_log.png")
        plt.savefig(filename5bis, dpi=300)
        plt.close()


# main

if __name__ == "__main__":
    
    INPUT_CSV = "Results/COMPILED_results.csv" 
    
    generer_graphiques(INPUT_CSV)