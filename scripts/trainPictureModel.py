from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
from saveClassNames import load_class_names
from datetime import datetime
from rapportTrainPicture import RapportPDF

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Chemins des données
chemin_entrainement = './dataset/images/train/'
chemin_test = './dataset/images/test/'

chemin_fichier_classes = './dataset/class_names.txt'

# Structures pour collecter les données
stats_entrainement = []
repartition_classes = defaultdict(int)

def obtenir_chemin_images(path):
    chemins_images = []
    for p in Path(chemin_entrainement).glob('*/*'):
        chemins_images.append(p)
        
    return chemins_images

def encoder_cible(path):
    if chemin_fichier_classes and Path(chemin_fichier_classes).exists():
        # Charger les classes depuis un fichier si disponible
        cible = load_class_names(chemin_fichier_classes)
        print("ici")
    else:
        # Générer dynamiquement les classes à partir des sous-dossiers
        cible = []
        for p in Path(path).glob('*'):
            nom_classe = p.stem[:-1] if p.stem == 'NidoranF' else p.stem
            cible.append(nom_classe)
        
        # Sauvegarder les classes dans un fichier si un chemin est fourni
        if chemin_fichier_classes:
            with open(chemin_fichier_classes, 'w') as f:
                for nom in cible:
                    f.write(nom + '\n')
    
    return cible

class ChargerDonnees(Dataset):
    def __init__(self, chemin_images, cible, transformation=None):
        self.chemin_images = chemin_images
        self.cible = cible
        self.transformation = transformation
        
    def __len__(self): return len(self.chemin_images)
    
    def __getitem__(self, idx):
        image = Image.open(self.chemin_images[idx]).convert('RGB')
        cible = self.cible.index(Path(self.chemin_images[idx]).stem.split('.')[0])
        
        if self.transformation:
            image = self.transformation(image)
            
            return image, cible
        else:
            return image, cible
        

transformation = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))
])

chemin_images = obtenir_chemin_images(chemin_entrainement)
cible = encoder_cible(chemin_entrainement)

chemin_images_val = obtenir_chemin_images(chemin_test)
cible_val = encoder_cible(chemin_test) 

ds = ChargerDonnees(chemin_images, cible, transformation=transformation)
ds_val = ChargerDonnees(chemin_images_val, cible_val, transformation=transformation)

for i, (x, y) in enumerate(ds):
    print(x.shape, y, '------ ensemble d\'entrainement')
    if i ==6:
      break
    
for j, (q, w) in enumerate(ds_val):
    print(q.shape, w, '------ ensemble de validation')
    if j ==6:
      break

bs = 16

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

dispositif = 'cuda' if torch.cuda.is_available() else 'cpu'
modele = EfficientNet.from_pretrained("efficientnet-b2")

for param in modele.parameters():
    param.requires_grad = False
    
modele._fc = nn.Linear(1408, len(cible))

for param in modele.parameters():
    if param.requires_grad == True:
        print(param.shape)

modele = modele.to(dispositif)

optimiseur = optim.Adam(modele.parameters(), lr=1e-3)
critere = nn.CrossEntropyLoss()
planificateur_lr = optim.lr_scheduler.CyclicLR(optimiseur, base_lr=1e-3, max_lr=0.01, cycle_momentum=False)

def valider(dataloader):
  modele.eval()
  total, correct = 0, 0
  for donnees in tqdm(dataloader, total=len(dataloader), leave=False):
    entrees, etiquettes = donnees
    entrees, etiquettes = entrees.to(dispositif), etiquettes.to(dispositif)
    sorties = modele(entrees)
    _, pred = torch.max(sorties, 1)

    total += etiquettes.size(0)
    correct += (pred == etiquettes).sum().item()

  return critere(sorties, etiquettes), (correct/total * 100)

# Collecte de la répartition des classes
def collecter_repartition_classes(chemin_images):
    for path in chemin_images:
        nom_classe = Path(path).parent.stem
        repartition_classes[nom_classe] += 1

    # Sauvegarde dans un DataFrame
    repartition_classes_df = pd.DataFrame(list(repartition_classes.items()), columns=["Classe", "Nombre d'images"])
    repartition_classes_df.to_csv("./pdf/modeleImage/datasets/class_distribution.csv", index=False)

# Collecte des statistiques d'entraînement
def collecter_stats_entrainement(epoch, perte_entrainement, perte_val, accuracy_val):
    stats_entrainement.append({
        "epoch": epoch,
        "perte_entrainement": perte_entrainement,
        "perte_val": perte_val,
        "accuracy_val": accuracy_val
    })

# Sauvegarde des stats à la fin
def sauvegarder_stats_entrainement():
    stats_entrainement_df = pd.DataFrame(stats_entrainement)
    stats_entrainement_df.to_csv("./pdf/modeleImage/datasets/training_stats.csv", index=False)

# Collecte des données pour la répartition des classes
collecter_repartition_classes(chemin_images)

# Exemple d'utilisation dans la boucle d'entraînement
epochs = 3
for epoch in range(epochs):
    modele.train()
    for donnees, etiquettes in tqdm(chargeur_entrainement, total=len(chargeur_entrainement), leave=False):      
        optimiseur.zero_grad()
        
        sortie = modele(donnees.to(dispositif))
        perte = critere(sortie, etiquettes.to(dispositif))
        perte.backward()
        
        optimiseur.step()
        planificateur_lr.step()
    
    validation = valider(chargeur_test)
    
    print(f"Époque: {epoch+1}/{epochs}\tperte_entrainement: {perte.item()}\tperte_val: {validation[0].item()}\taccuracy_val: {validation[1]}")

for params in modele.parameters():
    params.requires_grad = True

optimiseur1 = optim.Adam(modele.parameters(), lr=1e-3)

stats_par_batch = []

# Exemple d'utilisation dans la boucle d'entraînement (seulement pour les époques multiples de 4)
epochs = 20
for epoch in range(epochs):
    modele.train()
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