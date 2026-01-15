import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generer_graphiques_v2(input_file):

    try:
        df = pd.read_csv(input_file, sep=';')
    except FileNotFoundError:
        print(f"no such file'{input_file}'")
        return

    # Nettoyage 
    cols_percent = ['Précision Espèce', 'Précision Genre']
    for col in cols_percent:
        if df[col].dtype == 'object': 
            df[col] = df[col].str.replace('%', '').astype(float)

    sns.set_theme(style="whitegrid")
    
    # graph 1 : evolution precision espece top1
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="N Shot", 
        y="Précision Espèce", 
        hue="Technique",    
        style="Technique",  
        s=100,             
        palette="viridis"
    )
    plt.title("Évolution de la précision (au rang de l'espèce) selon le nombre de shots", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Nombre de Shots (N-Shot)")
    plt.ylim(0, 100)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    filename1 = f"{output_dir}/Graphique_1_evolution_precision_espece_top1.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close()

    # graph 1,5 evolution precision genre top1
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="N Shot", 
        y="Précision Genre", 
        hue="Technique",
        style="Technique",  
        s=100,              
        palette="viridis"
    )
    plt.title("Évolution de la précision (au rang du genre) selon le nombre de shots", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Nombre de Shots (N-Shot)")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    filename1_5 = f"{output_dir}/Graphique_1,5_evolution_precision_genre_top1.png"
    plt.savefig(filename1_5, dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # graph 2 : espece vs genre all dataset top1
    technique_name = "PREDICTIONS FEW SHOT all dataset"
    df_subset = df[df['Technique'] == technique_name]

    if not df_subset.empty:
        plt.figure(figsize=(10, 6))
        # Formatage pour Seaborn
        df_melted = df_subset.melt(id_vars=['N Shot'], value_vars=['Précision Espèce', 'Précision Genre'], 
                                   var_name='Niveau Taxonomique', value_name='Précision')
        
        sns.barplot(
            data=df_melted,
            x="N Shot",
            y="Précision",
            hue="Niveau Taxonomique",
            palette="viridis"
        )
        plt.title(f"Précision Espèce vs Genre ({technique_name})", fontsize=14)
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend(loc='lower right')
        
        filename2 = f"{output_dir}/Graphique_2_espece_vs_genre_alldataset_top1.png"
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close()

# graph 2,5 : espece vs genre sur 4 especes top1
    technique_name = "PREDICTIONS FEW SHOT sur 4 especes"
    df_subset = df[df['Technique'] == technique_name]

    if not df_subset.empty:
        plt.figure(figsize=(10, 6))
        # Formatage pour Seaborn
        df_melted = df_subset.melt(id_vars=['N Shot'], value_vars=['Précision Espèce', 'Précision Genre'], 
                                   var_name='Niveau Taxonomique', value_name='Précision')
        
        sns.barplot(
            data=df_melted,
            x="N Shot",
            y="Précision",
            hue="Niveau Taxonomique",
            palette="viridis"
        )
        plt.title(f"Précision Espèce vs Genre ({technique_name})", fontsize=14)
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend(loc='lower right')
        
        filename2 = f"{output_dir}/Graphique_2,5_espece_vs_genre_4especes_top1.png"
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close()
        
        
    # graph 3 : temps inference
    df_time = df[df['N Shot'] > 0].copy() # On ignore le 0 shot pour le temps
    
    if not df_time.empty and 'Inférence (s)' in df_time.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_time,
            x="N Shot",
            y="Inférence (s)",
            hue="Technique",
            palette="viridis"
        )
        plt.title("Temps d'inférence selon le N-Shot", fontsize=14)
        plt.ylabel("Temps (s)")
        #plt.yscale("log")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        filename3 = f"{output_dir}/Graphique_3_temps_inference.png"
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        plt.close()
    
        # graph 3,5 temps inference echelle log
    df_time = df[df['N Shot'] > 0].copy() # On ignore le 0 shot pour le temps
    
    if not df_time.empty and 'Inférence (s)' in df_time.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_time,
            x="N Shot",
            y="Inférence (s)",
            hue="Technique",
            palette="viridis"
        )
        plt.title("Temps d'inférence selon le N-Shot", fontsize=14)
        plt.ylabel("Temps (s)")
        plt.yscale("log")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        filename3 = f"{output_dir}/Graphique_3,5_temps_inference_log.png"
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        plt.close()

        # graph 4 evolution precision espece top5
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# main

if __name__ == "__main__":
    
    output_dir = "Results"
    
    INPUT_CSV = "Results/COMPILED_results.csv" 
    
    generer_graphiques_v2(INPUT_CSV)