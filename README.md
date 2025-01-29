# Who's That Pokemon

Site Flask de prédiction du Pokémon sur une image donnée à un modèle. 

- Rémy THIBAUT
- Quentin DELNEUF
- Damien VAURETTE

## Présentation 

Who's that Pokemon est un projet python permettant d'utiliser un modèle de prédiction d'images que vous pouvez entrainer. Une interface Web permet de déposer une image et d'obtenir une prédiction. Cette prédiction mène par la suite à l'affichage d'information sur le pokémon déterminer (tel un pokédex).

Les informations que vous pouvez retrouvez sont : 
- Les stats du Pokémon prédit (type, description, taille, poids...)
- Sa localisation dans Pokémon Go sur une map interactive (localisation centrée sur la France métropolitaine)
- Des graphiques sur les stats du pokémon
- Un calcul du taux de capture estimé en fonction de la situation lors du combat (vie restante du pokémon, son niveau, le type de pokéball...)

Ce projet met à disposition l'ensemble des éléments pour pouvoir : 
- Lancer le site Flask
- Entrainer le modèle
- Utiliser le modèle pour une prédiction

## Installation 

Pour utiliser ce projet, vous devez d'abord le cloner depuis le repository Github : 

```bash
git clone https://github.com/Hitoyu22/WhosThatPokemon
cd WhosThatPokemon
```

Pour pouvoir lancer le projet, vous devez avoir au préalable : 
- Python 3
- Un environnement python

Pour créer un environnement Python, utilisez la commande suivante dans le dossier parent du projet.

- Pour Linux : 
```bash
python3 -m venv ./nomDeMonEnvironnement
source nomDeMonEnvironnement/bin/activate
```

- Pour Windows :
```bash
python 
```

Désormais, il ne vous reste plus qu'à installer l'ensemble des librairies utilisées par le projet avec la commande suivante pour toutes les télécharger d'une traite. 

```bash
pip instal -r requirements.txt
```

## Lancement des différents scripts

WhoThatPokemon propose différent script pour diverses tâches : 

- Lancer le serveur Flask (veuillez à ne rien à avoir sur le port 5000 avant de lancer) : 
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

## Librairies utilisées 
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

## Objectifs du projet 

## Difficultés rencontrées
