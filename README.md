# Who's That Pokemon

Site Flask de prédiction du Pokémon sur une image donnée à un modèle. 

- Rémy THIBAUT
- Quentin DELNEUF
- Damien VAURETTE

## Présentation 

**Who's that Pokemon** est un projet python permettant d'utiliser un modèle de prédiction d'images que vous pouvez entrainer. Une interface Web permet de déposer une image et d'obtenir une prédiction. Cette prédiction mène par la suite à l'affichage d'information sur le pokémon déterminer (tel un pokédex).

Les informations que vous pouvez retrouvez sont : 
- Les **stats du Pokémon prédit** (type, description, taille, poids...)
- Sa **localisation dans Pokémon Go** sur une map interactive (localisation centrée sur la France métropolitaine)
- Des **graphiques** sur les stats du pokémon
- Un **calcul du taux de capture** estimé en fonction de la situation lors du combat (vie restante du pokémon, son niveau, le type de pokéball...)

Ce projet met à disposition l'ensemble des éléments pour pouvoir : 
- Lancer le site **Flask**
- **Entrainer le modèle**
- Utiliser le **modèle pour une prédiction**

Le modèle que nous avons entrainé dépasse le 90 % de précision (la plus grande précisions que nous avons obtenu est de 94.4 %).

## Installation 

Pour utiliser ce projet, vous devez d'abord le cloner depuis le repository Github : 

```bash
git clone https://github.com/Hitoyu22/WhosThatPokemon
cd WhosThatPokemon
```

Pour pouvoir lancer le projet, vous devez avoir au préalable : 
- **Python 3**
- Un **environnement python**

Pour créer un environnement Python, utilisez la commande suivante dans le dossier parent du projet.

- Pour **Linux** : 
```bash
python3 -m venv ./nomDeMonEnvironnement
source nomDeMonEnvironnement/bin/activate
```

- Pour **Windows** :
```bash
python3 -m venv ./
```

Désormais, il ne vous reste plus qu'à installer l'ensemble des librairies utilisées par le projet avec la commande suivante pour toutes les télécharger d'une traite. 

```bash
pip instal -r requirements.txt
```

## Lancement des différents scripts

WhoThatPokemon propose différent script pour diverses tâches : 

- Lancer le serveur **Flask** (veillez à ne rien à avoir sur le port **5000** avant de lancer) : 
```bash
python3 main.py
```
Vous pourrez accéder au site à l'adresse suivante : http://localhost:5000

- Lancer l'entrainement du modèle (nous vous recommandons un ordinateur puissant avec un bon processeur sinon l'entrainement peut prendre du temps) : 
```bash
python3 scripts/trainPictureModele.py
```

- Nettoyer le dossier d'upload des images uploader pour la prédiction : 
```bash
python3 scripts/clean.py
```

## Librairies utilisées 
Les librairies utilisées dans ce projet sont : 

- **torch** : Framework pour l’apprentissage automatique et la construction de modèles de réseaux de neurones.
- **torchvision** : Outils pour la vision par ordinateur, incluant des modèles pré-entraînés et des transformations d’images.
- **efficientnet-pytorch** : Implémentation de l’architecture EfficientNet pour la classification d’images.
- **pandas** : Bibliothèque pour la manipulation et l'analyse de données, notamment les fichiers CSV.
- **matplotlib** : Bibliothèque pour la visualisation de données et l'affichage de graphiques.
- **tqdm** : Outil pour afficher des barres de progression lors d'opérations longues.
- **Flask** : Framework web léger pour créer des applications web et interagir avec ton modèle de prédiction.
- **werkzeug** : Outils pour gérer les requêtes HTTP dans des applications web, utilisé avec Flask.
- **fpdf** : Bibliothèque pour générer des fichiers PDF.

## Objectifs du projet 

Les objectifs du projet sont divers et variés : 
- **Produire un projet qui nous intéressait** : Nous avons été très heureux de travailler sur ce projet qui nous intéressait tous. Créer un site permettant de déterminer le pokémon sur une image que vous avez importé est un challenge qui nous a tout de suite plu.
- **Utiliser les librairies et techniques de Datamining vu en cours** : Afin de valider cette matière et s'assurer que nous ayons bien compris les enjeux du datamining / machine learning, nous avons utilisé les librairies que nous avions étudié en cours (Pandas, numpy...).
- **Apprendre de nouvelles technologies de datamining (pyTorch)** : Pour créer notre modèle nous étions dans l'obligation d'utiliser une librairie dédiée aux images comme pytorch.Cela nous a permis d'apprendre une nouvelle technologie intéressante dans le cas de ce projet.
- **Développer un projet complet dans une interface web** : Afin de proposer une expérience innovante pour les utilisateurs de notre modèle, nous avons utiliser Flask afin de proposer une interfaçe web pour pouvoir importer l'image et regarder les informations du pokemon.

## Difficultés rencontrées

Durant ce projet, nous avons rencontrés plusieurs difficultés : 

- **Trouver des jeux de données utiles** : 
Il est bien important avant de trouver de faire du datamining / machine learning d'avoir des datasets de bonnes qualités (données intéressantes, peu de données manquantes, cohérences des données). Nous sommes finalement parvenus à trouver nos différents dataset sur Kaggle et nous avons pu développer notre application de reconnaissance de pokémon.
- **Développer un modèle de reconnaissance d'image performant** : 
Il s'agit selon nous de la tâche la plus difficile de notre projet. N'ayant pas vu en cours le machine learning sur image, nous avons du chercher comment faire pour développer un modèle suffisamment performant pour pouvoir le présenter comme projet final. 
Nous avions essayé à plusieurs reprises de créer un modèle avec la librairie tensorflow mais le modèle ne dépassait jamais les 30% malgré le fait que nous modifions les paramètres de nombreuses fois. 
Nous avons alors décidé de chercher sur kaggle et github afin de trouver une code pour nous aider à comprendre comment créer un modèle performant. Nous sommes alors tomber sur un code sur kaggle avec un dataset de 30 000 images sur lequel nous arrivions à avoir plus de 90% de précision lorsque nous l'entrainions. 
Nous avons alors décidé de nous en inspirer pour développer notre modèle. Nous n'avons donc pas créer de 0 notre modèle mais nous nous sommes inspirés afin de comprendre le fonctionnement et les bonnes manières de machine learning. 
Le code qui nous a inspiré est le suivant : https://www.kaggle.com/code/bipinkrishnan/pokemon-image-classification-with-efficientnet
