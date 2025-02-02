from PIL import Image

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
from saveClassNames import ClassNameGestion
from datetime import datetime
from rapportTrainPicture import RapportPDF

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

"""
Ce script permet d'entraîner un modèle de classification d'images en utilisant PyTorch et EfficientNet. Il comprend plusieurs étapes, de la préparation des données à l'entraînement, la validation du modèle, et la sauvegarde des résultats dans des fichiers CSV et des rapports PDF.

Le processus de ce script est le suivant :

1. Chargement et préparation des données :
   - Chemins des données : Les chemins vers les images d'entraînement et de test sont spécifiés dans les variables `chemin_entrainement` et `chemin_test`. Ces chemins sont utilisés pour récupérer les images et les préparer pour l'entraînement et la validation.
   - Gestion des classes : Le fichier `class_names.txt` contient les noms des classes. S'il existe, il est utilisé pour charger les classes. Sinon, les classes sont extraites des noms de répertoires des images, en assumant que chaque dossier correspond à une classe différente.
   - Transformation des images : Des transformations sont appliquées aux images avant de les envoyer dans le modèle. Les images sont redimensionnées, converties en tenseurs et normalisées.

2. Définition du modèle :
   - Le modèle utilisé est EfficientNet, un réseau de neurones convolutifs pré-entraîné sur une grande base de données. Ce modèle est chargé à l'aide de la méthode `from_pretrained()` d'EfficientNet.
   - Pour s'adapter au nombre de classes du jeu de données, la dernière couche (la couche entièrement connectée) est modifiée en fonction du nombre de classes.
   - Avant de commencer l'entraînement, les paramètres du modèle sont gelés (`param.requires_grad = False`) afin que seules les couches modifiées soient entraînées.
   
3. Entraînement et validation :
   - L'optimiseur utilisé pour mettre à jour les poids du modèle est l'optimiseur Adam, qui est couramment utilisé pour l'entraînement des réseaux neuronaux.
   - La fonction de perte utilisée est CrossEntropyLoss, qui est appropriée pour la classification multi-classes.
   - Le taux d'apprentissage est régulé à l'aide de la classe `CyclicLR` de PyTorch, qui ajuste dynamiquement le taux d'apprentissage pendant l'entraînement pour améliorer la convergence.
   - Le processus d'entraînement est effectué sur plusieurs époques. Après chaque époque, une validation est effectuée sur le jeu de données de test pour mesurer la performance du modèle. Les statistiques de perte et de précision sont collectées à chaque époque.

4. Sauvegarde des résultats :
   - Les statistiques de l'entraînement (perte d'entraînement, perte de validation, et précision de validation) sont sauvegardées dans des fichiers CSV pour une analyse ultérieure.
   - Un rapport PDF est généré à la fin de l'entraînement, contenant les statistiques d'entraînement, la répartition des classes et d'autres informations pertinentes.

5. Collecte et gestion des données :
   - La fonction `obtenir_chemin_images()` parcourt le répertoire d'entraînement et collecte les chemins de toutes les images.
   - La fonction `encoder_cible()` permet d'encoder les classes des images, soit en les lisant depuis un fichier existant, soit en les extrayant des répertoires des images.

6. Classes et objets utilisés :
   - La classe `ChargerDonnees` est définie pour charger les images et les classes dans un format compatible avec PyTorch. Elle utilise la méthode `__getitem__()` pour charger une image spécifique à partir de son indice, en appliquant éventuellement une transformation.
   - Des objets `DataLoader` sont créés pour gérer l'entraînement et la validation des données en batchs. Les images sont traitées par lot afin d'accélérer le processus d'entraînement.

7. Répartition des classes et collecte des statistiques :
   - La fonction `collecter_repartition_classes()` permet de collecter et sauvegarder la répartition des classes dans un fichier CSV. Cela permet de vérifier si les données sont équilibrées.
   - La fonction `collecter_stats_entrainement()` enregistre les statistiques d'entraînement, y compris la perte et la précision de chaque époque. Ces statistiques sont également sauvegardées dans un fichier CSV.

8. Enregistrement du modèle :
   - À intervalles réguliers (chaque 4 époques dans ce script), le modèle est sauvegardé avec un nom de fichier contenant la précision du modèle et la date/heure actuelles. Cela permet de garder une trace des modèles au fur et à mesure de leur évolution.
   - À la fin de l'entraînement, les paramètres du modèle sont rendus entraînables (pour un ajustement fin si nécessaire).

9. Génération du rapport :
   - Un rapport PDF est généré avec toutes les statistiques d'entraînement collectées, la répartition des classes et d'autres informations pertinentes pour une analyse plus approfondie de la performance du modèle.

Ce script permet de réaliser un entraînement efficace et bien documenté d'un modèle de classification d'images avec EfficientNet en utilisant PyTorch, et inclut la gestion des données, la sauvegarde des résultats et la génération de rapports pour une analyse complète de l'entraînement.
"""


# Chemins des données
chemin_entrainement = './dataset/images/train/'
chemin_test = './dataset/images/test/'

chemin_fichier_classes = './dataset/class_names.txt'

className = ClassNameGestion(train_path="./dataset/images/train", class_file_path="./dataset/class_names.txt")

# Structures pour collecter les données
stats_entrainement = []
repartition_classes = defaultdict(int)

def obtenir_chemin_images(path):
    """
    Fonction pour obtenir les chemins des images
    """
    # On utilise glob pour obtenir les chemins de toutes les images
    chemins_images = []
    for p in Path(chemin_entrainement).glob('*/*'):
        chemins_images.append(p)
        
    return chemins_images

def encoder_cible(path):
    """
    Fonction pour encoder les classes cibles
    """
    if chemin_fichier_classes and Path(chemin_fichier_classes).exists():
        cible = className.load_class_names()
    else:
        cible = []
        # On parcourt les dossiers pour obtenir les classes
        for p in Path(path).glob('*'):
            nom_classe = p.stem[:-1] if p.stem == 'NidoranF' else p.stem
            cible.append(nom_classe)
        # On trie les classes
        if chemin_fichier_classes:
            with open(chemin_fichier_classes, 'w') as f:
                for nom in cible:
                    f.write(nom + '\n')
    
    return cible

class ChargerDonnees(Dataset):
    """
    Classe pour charger les données
    """
    def __init__(self, chemin_images, cible, transformation=None):
        """
        Initialisation
        """
        self.chemin_images = chemin_images
        self.cible = cible
        self.transformation = transformation
        
    def __len__(self): return len(self.chemin_images)
    """
    Fonction pour obtenir la longueur du dataset
    """
    
    def __getitem__(self, idx):
        """
        Fonction pour obtenir un élément du dataset
        """
        image = Image.open(self.chemin_images[idx]).convert('RGB')
        cible = self.cible.index(Path(self.chemin_images[idx]).stem.split('.')[0])
        
        if self.transformation:
            image = self.transformation(image)
            
            return image, cible
        else:
            return image, cible
        
# On définit les transformations à appliquer aux images : redimensionnement, normalisation et conversion en tenseur
transformation = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))
])

# On charge les données
chemin_images = obtenir_chemin_images(chemin_entrainement)
# On encode les classes cibles
cible = encoder_cible(chemin_entrainement)
# On crée un objet de la classe ChargerDonnees
chemin_images_val = obtenir_chemin_images(chemin_test)
# On encode les classes cibles
cible_val = encoder_cible(chemin_test) 

# On crée un objet de la classe ChargerDonnees
ds = ChargerDonnees(chemin_images, cible, transformation=transformation)
# On crée un objet de la classe ChargerDonnees
ds_val = ChargerDonnees(chemin_images_val, cible_val, transformation=transformation)

# On crée des chargeurs de données
for i, (x, y) in enumerate(ds):
    print(x.shape, y, '------ ensemble d\'entrainement')
    if i ==6:
      break
    
for j, (q, w) in enumerate(ds_val):
    print(q.shape, w, '------ ensemble de validation')
    if j ==6:
      break

# On crée des chargeurs de données
bs = 16
# On crée des chargeurs de données
chargeur_entrainement = DataLoader(ds, batch_size=bs, shuffle=True)
chargeur_test = DataLoader(ds_val, batch_size=bs, shuffle=True)

for i, (x, y) in enumerate(chargeur_entrainement):
    print(x.shape, y.shape, '------ ensemble d\'entrainement')
    if i ==8:
      break
    
for j, (q, w) in enumerate(chargeur_test):
    print(q.shape, w.shape, '------ ensemble de validation')
    if j ==8:
      break

# On définit le dispositif : 'cuda' s'il est disponible, sinon 'cpu'
dispositif = 'cuda' if torch.cuda.is_available() else 'cpu'
modele = EfficientNet.from_pretrained("efficientnet-b2")

# On gèle les paramètres du modèle
for param in modele.parameters():
    param.requires_grad = False
    
# On modifie la dernière couche du modèle pour qu'elle ait le nombre de classes de notre jeu de données
modele._fc = nn.Linear(1408, len(cible))

# On affiche les paramètres qui nécessitent un gradient
for param in modele.parameters():
    if param.requires_grad == True:
        print(param.shape)

# On envoie le modèle sur le dispositif
modele = modele.to(dispositif)

# On définit l'optimiseur, le critère et le planificateur de taux d'apprentissage
optimiseur = optim.Adam(modele.parameters(), lr=1e-3)
critere = nn.CrossEntropyLoss()
planificateur_lr = optim.lr_scheduler.CyclicLR(optimiseur, base_lr=1e-3, max_lr=0.01, cycle_momentum=False)

# Fonction pour valider le modèle
def valider(dataloader):
    """
    Fonction pour valider le modèle
    """
    # On met le modèle en mode évaluation
    modele.eval()
    # On initialise les variables pour le calcul de la perte et de la précision
    total, correct = 0, 0
        # On parcourt les données avec tqdm pour afficher une barre de progression
    for donnees in tqdm(dataloader, total=len(dataloader), leave=False):

        # On obtient les entrées et les étiquettes
        entrees, etiquettes = donnees
        entrees, etiquettes = entrees.to(dispositif), etiquettes.to(dispositif)
        # On obtient les sorties du modèle
        sorties = modele(entrees)
        _, pred = torch.max(sorties, 1)
        # On met à jour les variables pour le calcul de la perte et de la précision
        total += etiquettes.size(0)
        correct += (pred == etiquettes).sum().item()
        # On retourne la perte et la précision
    return critere(sorties, etiquettes), (correct/total * 100)

# Collecte de la répartition des classes
def collecter_repartition_classes(chemin_images):
    """
    Fonction pour collecter la répartition des classes
    """
    for path in chemin_images:
        nom_classe = Path(path).parent.stem
        repartition_classes[nom_classe] += 1

    # Sauvegarde dans un DataFrame
    repartition_classes_df = pd.DataFrame(list(repartition_classes.items()), columns=["Classe", "Nombre d'images"])
    repartition_classes_df.to_csv("./pdf/modeleImage/datasets/class_distribution.csv", index=False)

# Collecte des statistiques d'entraînement
def collecter_stats_entrainement(epoch, perte_entrainement, perte_val, accuracy_val):
    """
    Fonction pour collecter les statistiques d'entraînement
    """
    # On ajoute les statistiques à la liste : epoch, perte_entrainement, perte_val, accuracy_val
    stats_entrainement.append({
        "epoch": epoch,
        "perte_entrainement": perte_entrainement,
        "perte_val": perte_val,
        "accuracy_val": accuracy_val
    })

# Sauvegarde des stats à la fin
def sauvegarder_stats_entrainement():
    """
    Fonction pour sauvegarder les statistiques d'entraînement
    """
    # On sauvegarde les statistiques dans un fichier CSV
    stats_entrainement_df = pd.DataFrame(stats_entrainement)
    stats_entrainement_df.to_csv("./pdf/modeleImage/datasets/training_stats.csv", index=False)

# Collecte des données pour la répartition des classes
collecter_repartition_classes(chemin_images)

# Exemple d'utilisation dans la boucle d'entraînement : entraînement pendant 3 époques pour un modèle pré-entraîné
epochs = 3
for epoch in range(epochs):
    # On met le modèle en mode entraînement
    modele.train()
    # On parcourt les données avec tqdm pour afficher une barre de progression
    for donnees, etiquettes in tqdm(chargeur_entrainement, total=len(chargeur_entrainement), leave=False):      
        # On remet les gradients à zéro
        optimiseur.zero_grad()
        # Passage avant
        sortie = modele(donnees.to(dispositif))
        perte = critere(sortie, etiquettes.to(dispositif))
        perte.backward()
        # Mise à jour des paramètres         
        optimiseur.step()
        planificateur_lr.step()
    # Validation après l'entraînement de l'époque
    validation = valider(chargeur_test)
    print(f"Époque: {epoch+1}/{epochs}\tperte_entrainement: {perte.item()}\tperte_val: {validation[0].item()}\taccuracy_val: {validation[1]}")

# On passe les paramètres du modèle en mode entraînable
for params in modele.parameters():
    params.requires_grad = True

# On crée un nouvel optimiseur : Adam avec un taux d'apprentissage de 1e-3, Adam est un optimiseur très utilisé en pratique
optimiseur1 = optim.Adam(modele.parameters(), lr=1e-3)

# On crée une liste pour collecter les statistiques par batch
stats_par_batch = []

# Exemple d'utilisation dans la boucle d'entraînement (seulement pour les époques multiples de 4)
epochs = 20
for epoch in range(epochs):
    # On met le modèle en mode entraînement
    modele.train()
    # On parcourt les données avec tqdm pour afficher une barre de progression
    for i, (donnees, etiquettes) in enumerate(tqdm(chargeur_entrainement, total=len(chargeur_entrainement), leave=False)):      
        optimiseur1.zero_grad()
        
        # Passage avant
        sortie = modele(donnees.to(dispositif))
        perte = critere(sortie, etiquettes.to(dispositif))
        
        # Calcul des gradients
        perte.backward()
        optimiseur1.step()
        
        # Enregistrement des statistiques par batch si c'est l'époque voulue
            # Calcul de la précision pour ce batch
        _, pred = torch.max(sortie, 1)
        correct = (pred == etiquettes.to(dispositif)).sum().item()
        total = etiquettes.size(0)
        accuracy = correct / total * 100
        
        # Sauvegarde des statistiques par batch
        if (epoch + 1) % 4 == 0:
        # Ajout des stats de batch
            stats_par_batch.append({
                "epoch": epoch + 1,
                "batch": i + 1,
                "perte_entrainement": perte.item(),
                "accuracy_entrainement": accuracy
            })
    
    # Validation après l'entraînement de l'époque
    validation = valider(chargeur_test) 
    
    # Sauvegarder tous les 4 epochs
    if (epoch + 1) % 4 == 0:
        # Obtenir la date et l'heure actuelles
        maintenant = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        acc = validation[1]
        nom_modele = f'./model/model-{round(acc, 4)}-{maintenant}.pt'
        torch.save(modele.state_dict(), nom_modele)
        print(f"Modèle sauvegardé: {nom_modele}")
    
    # Collecte des statistiques d'entraînement
    collecter_stats_entrainement(epoch + 1, perte.item(), validation[0].item(), validation[1])
    
    print(f"Époque: {epoch+1}/{epochs}\tperte_entrainement: {perte.item()}\tperte_val: {validation[0].item()}\taccuracy_val: {validation[1]}")

# Sauvegarde des statistiques des batches dans un fichier CSV après l'entraînement
if stats_par_batch:
    stats_par_batch_df = pd.DataFrame(stats_par_batch)
    stats_par_batch_df.to_csv("./pdf/modeleImage/datasets/batch_stats.csv", index=False)

# Sauvegarde des données après l'entraînement
sauvegarder_stats_entrainement()


# Exemple d'utilisation
chemin_repartition = "./pdf/modeleImage/datasets/class_distribution.csv"
chemin_statistiques = "./pdf/modeleImage/datasets/training_stats.csv"
chemin_batch_stats = "./pdf/modeleImage/datasets/batch_stats.csv"
chemin_rapport = f"./pdf/modeleImage/rapport_apprentissage_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"

rapport = RapportPDF(chemin_repartition, chemin_statistiques, chemin_batch_stats, chemin_rapport)
rapport.generer_pdf()