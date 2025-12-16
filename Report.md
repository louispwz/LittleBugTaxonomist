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

### Démarche projet

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

-   Premier tests ?

### Orga

Visio mercredi 26/11 avec Laeticia : point sur la compréhension du
papier

Reu mardi 02/12 avec chercheurs et Laeticia :

-   Discussion sur les données de l'INRAE
-   A préparer : slide présentation du papier BIOCLIP : ce qu’on a
    compris/pas compris sur le modèle, la méthode, les données entrées,
    les sorties ??? etc

# Compte Rendu Visio Laeticia 26/11

### Général

-   zero shot = besoin d'info auxiliaire type texte (--\> CLIP
    text-to-image), pas ici

-   regarder ce que c'est un ViT ? ce que c'est un ViT

-   regarder comment la hiérarchie se comporte

-   prendre un jeu de donnée random et tester le code --\> voir output

-   comment les classes s'organisent dans l'espace latent ? comment se
    structure ce nouvel espace ? remonte ?

# Compte Rendu Reu Team HIVE 02/12

### Général

Projet dans projet : on se place au début d'un projet de grande ampleur
et on "défriche" un peu le terrain en cherchant des trucs qui marchent
dans le but d'une automatisation et généralisation plus tard

-   Questions : quelle bestioles sont presente et quelle bestioles sont
    dangereuses ?

-   Plusieurs méthodes d'échantillonages/récolte d'info :
    « morphologique » (capture + observation) ou via adn (broyage
    environnement) : nous on va regarder la morpho

Nous on se place dans la situation pre broyage : on veut maximiser les
capa d’iddentification de la bestiole

-   Certaines bestioles sont des marqueurs de biodiversité : en trouver
    moins est pas bon → les carabes ??

-   Dans BioClip2 : taxons pas tous égaux en termes de nombres d’image :
    certains taxons très bien représenté, acuracy sur chaque dataset
    différents

### Questions ?

QUESTIONS IMPORTANTES A FOCUS :

-   IMPACT DE LA HIERARCHIE SUR LES PERF : comment l’intégrer ? Qu’est
    ce que ca apporte ?

-   comment se positionne bioclip2 dans la problématqiue de nos
    bestioles ?

-   fine tuning ? Est qu’un ViT sans hiérarchie ca marche ?
    (structuration des données, espaces des données??)

-   explicabilité ? Sur quoi s’appuie le modèle pour prendre des
    decisions ? ON VEUT : un modèles spécialisé interesse tt les
    domaines, et chaque domaine et spécialisation ont leur manière de
    prendre des décision (experts regardent les mandibules pour troiver
    espèces, etc)

Nouvelles données :

-   downstream tasks ? Question de la robustesse via diversité puis
    downstream sur ta tache de base pour l’affiner et rajouter
    raisonnement et connaissances → modèles fondateur utilisé comme base
    avant spécialisation à une tâche précise

-   comparer spécificité / robustesse / précision ?? → pour certaines
    taches

    → centralité de la question à porter sur la hiérarchie → comment
    l’évaluer correctement (métrique) + impact de la rajouter ou de
    l’enlever etc

-   Question de la structuration de l’espace pour la compréhension : IA
    explicable → on a qqch qui marche mais qui sait pas expliquer
    comment et pk il marche

### Tâches

Tâche 1 : Concrête, avancer le code pas à pas

DANS UN PREMIER TPS : EVALUER au niveau des espèces

-   Récupérer les données : faire un parceur --\> répertoire avec
    iddentifiant unique GBIF a récupérer (API????) =\> 291 espèce, 180
    genres, … tribut, 1 famille (biais car viens du british museum,
    empire colonial ??)

-   on lui file que les espèces et on inferera des classes après
    seulement, car on connaît la taxonomie nous de notre coté → évaluer
    ca d’abord

-   papier sur les données des danois sur les carabes : ils ont fais
    avec CNN

    → on doit tester avec ViT / contrastive learning / loss contrastive
    / etc

    → résultats du papier sur ces données battu par nimporte quel modele
    moderne : tester avec un ViT a plat + d'autres choses plus modernes
    pour monter la précision

    → pas aussi diversifié que les jeux de données qu’on voudrait dans
    le gros projet mais un bon jeu de données pour jouer et tester des
    choses

ENSUITE

-   méthodo : petit exemple, petit a petit, pour façonner un modèle et
    arriver à qqch qui tourne bien sur du petit

-   comparer perf au papier original, évaluer avec differente metriques
    --\> trouver quelles métriques ?

-   attention a la balance des classes : devrait etre ok mais à verifier

Tâche 2 : Théorique, compréhension du modèle a fond

-   Question de la gestion des classes unbalanced : est ce que BioClip /
    BioClip2 le fait ? comment ?

-   La loss fct : qu'est ce qu'elle fait concretement ? comment elle
    marche ? Question des espaces de départ et d'arrivée (latent ?)

-   Question du few shot : comprendre comment ca marche et comment
    l’appliquer : comment placer dans l’espace en rajoutant des images
    few ??

-   A FAIRE : GRAND SCHEMA RECAP DU MODELE AVEC LES "MODULES"
    (encodeurs, double projecteur, etc)

### Bonus

Problème des calcules et allocation de GPU :

-   réseau «matrice»: informatitien du CNRS, abonnement pour les calculs
    ?

# Compte Rendu réu 08 / 12

-   Lire bioclip pour plus d'explicabilité sur la construction du modele

-   voir si on trouve une certaine ponderation dans la validation
    (evaluation fine)

-   si je met une hierarchie qui combine des taxons, comment il le gere
    ? si c'est de la simalirité dans les images / dans le texte. Tester
    en apprenant sur une taxonomie qui ne marche pas

tester avec un dataset et permutter les fins de taxons pour voir
l'impact de la hierarchie

-   faire des premieres campagnes d'evaluation. faire un code sur les
    differentes accuracy n-1 et n-2
    
    
# Compte Rendu réu LABO 12 / 12

### General 

- Explication du projet aux chercheurs et inge du site qui bossent avec nous, ou on en est + enjeux pour future (these etc)

- But du projet : décrassage méthodologique sur les modèle de fodatios et leur mainmise + ineficacité en cas de tâches spécifiques ? 

  → Passer d'un modèle de fondation à un modèle + qualifié/spécifique

  → Libérer du temps aux stagiaires/chercheurs pour la reconnaissance de l'individus lors des collectes
  
  → Axer le "rapport" sur méthodo/decrassage/exploration et premieres réflexions sur le fond de ces modèles de fondation (question des ressources imense necessaires a l'etrainement, souveraineté en EU etc)

- Discussion autour d'améliorations de l'etrainement sur les carabes via des nouvelles méthodes de récupérations de data : shaking ? photo a 360 (theses de la doctorate) ? constructiosn possible pour etayer le modèles

  → Pricipe de multivue + data multi modale pour etraiemet, discussion sur ces modalités : multivue (échelles, angles, rotations, éléments parasites ex terre, 3D, gènomes ?)

### Questions 

- voir pour le "je sais pas" au niveau de la feuille ? peut simuler ca ? peut le faire s'arreter au dessus ou il a tres souvent juste ? serait bien pour biologistes/stagiaires

- matrice de confusion utile ici ? 

- forme de sortie attendu pour nos modèle ? JSOn ? 

### Tâche : 

- Tester trucs au pif sur few shot pour voir ce qu'il trouve, normalement il va donner ue réponse alors que ca a pas lieu d'etre





# compte Rendu réu 18 / 12
