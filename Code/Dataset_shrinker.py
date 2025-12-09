################
#    Imports   #
################

import tarfile
import os 
import random
from collections import defaultdict


################
#   Shrinker   #
################


def dataset_shrinker(input_tar, n_folders, n_files, output_tar, seed=None):

    # graine si besoin pour ravoir les memes dossier par exemple
    if seed is not None:
        random.seed(seed)

    # orga des fichiers 
    folders = defaultdict(list)
    with tarfile.open(input_tar, "r") as tar_in:
        for member in tar_in.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 2:
                continue
            folder = parts[-2]
            folders[folder].append(member)

        # choisi n dossier
        all_folders = sorted(folders.keys())
        if n_folders >= len(all_folders):
            selected_folders = all_folders
        else:
            selected_folders = random.sample(all_folders, n_folders)

        # selection n fichier et ecrit dans tar_out
        with tarfile.open(output_tar, "w:tar") as tar_out:
            for folder in selected_folders:
                members = folders[folder]
                if n_files >= len(members):
                    selected_files = members
                else:
                    selected_files = random.sample(members, n_files)

                for member in selected_files:
                    file = tar_in.extractfile(member)
                    if file is not None:
                        tar_out.addfile(member, file)

    print(f'Dataset reduit de taille {n_folders} dossiers et {n_files} fichiers créé à {output_tar} Youhou 🎉🎉🎉')
    
    
################
#   Call func  #
################

if __name__ == "__main__":
    dataset_shrinker(input_tar="Data/database.tar",n_folders=50,n_files=50, output_tar="Data/small_database.tar")