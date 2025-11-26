---
editor_options: 
  markdown: 
    wrap: 72
---

# Compte Rendu réu init 24 / 11

### Général

-   But : faire un modèle qui résout toute les taches, modèles hyper
    généraliste entrainée sur un énorme volume de données, très fort en
    généralisation

-   BioClip 2 -\> modele de fondation = modèles tres fort en
    generalisation

-   Idée : être capable de tester des choses qui existent pour les
    modeles hierarchiques

-   Regarder comment les erreurs sont calculé et justifié.

### Démarche projet :

Etape 1 : Tester ce modèle de fondation (BioClip2) :

-   Comprendre comment il marche et regarder limites/+données etc
-   Test effectif du code

Etape 2 : Question de l’évaluation : on a plus que l’acuracy, métriques
a trouver pour chaque niveaux, quand on se trompe, comment on se trompe
???

### Questions ?

-   Question des données : images centrées ? bien faites

-   Comprendre ce qu’est un modèle de fondation, et comprendre les
    rouages de celui-ci : quest ce qu’on fourni au modèle ? données et
    train/test ?

-   Est-ce que les choses qu’on retrouve/voit dans les images reflètent
    la taxonomie ? et donc le modèle fait les liens et la hiérarchie
    tout seul ? ou alors il y a des notions de hiérarchisation dans le
    modèle ?

### Tâches

-   Comprendre rouage du modèle

-   Tirer la ficelles des ref : aller voir les papiers cité dans sources
    init

- Premier tests ?

### Orga :

Visio mercredi 26/11 avec Laeticia : point sur la compréhension du
papier

Reu vendredi 28/11 avec Laeticia à Vannes :

Reu mardi 02/12 avec chercheurs et Laeticia :

-   Discussion sur les données de l'INRAE
-   A préparer : slide présentation du papier BIOCLIP : ce qu’on a
    compris/pas compris sur le modèle, la méthode, les données entrées,
    les sorties ??? etc




    
    
- zero shot = besoin d'info auxiliaire type texte (--> CLIP text-to-image), pas ici 

- regarder ce que c'est un ViT ? ce que c'est un ViT

- regarder comment la hiérarchie se comporte

- prendre un jeu de donnée random et tester le code --> voir output

- comment les classes s'organisent dans l'espace latent ? comment se structure ce nouvel espace ? remonte ? 

- 
