import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import pandas as pd
from pathlib import Path
from scripts.saveClassNames import ClassNameGestion

# On charge les noms des classes depuis le fichier class_names.txt via la classe ClassNameGestion
className = ClassNameGestion(train_path="./dataset/train", class_file_path="./dataset/class_names.txt")

class_file_path = './dataset/class_names.txt'

# On crée une classe PokemonClassifier
class PokemonClassifier:
    """
    Classe pour classifier les Pokémon à partir d'une image
    """
    def __init__(self, model_path, csv_path):
        # ON défini les paramètres de l'appareil avant de charger le modèle
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # On charge les noms des classes
        self.class_names = className.load_class_names()
        # On charge le modèle EfficientNet pour avoir déjà les poids pré-entraînés
        self.model = EfficientNet.from_pretrained("efficientnet-b2")
        # On change la dernière couche pour qu'elle corresponde à notre nombre de classes
        self.model._fc = torch.nn.Linear(1408, len(self.class_names))
        # On charge les poids du modèle
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        # On envoie le modèle sur l'appareil
        self.model = self.model.to(self.device)
        # On met le modèle en mode évaluation
        self.model.eval() 
        # On charge les données des Pokémon (nom et numéro utile pour la traduction après prédictions)
        self.pokemon_data = pd.read_csv(csv_path)

    
    # On crée une méthode pour obtenir le numéro du Pokédex à partir du nom du Pokémon (Le modèle renvoie un nom de Pokémon en anglais)
    def get_pokedex_number(self, pokemon_name):
        """
        Méthode pour obtenir le numéro du Pokédex à partir du nom du Pokémon
        """

        # On cherche le nom du Pokémon dans les données, str.lower() permet de mettre en minuscule pour éviter les erreurs de casse
        pokemon_info = self.pokemon_data[self.pokemon_data['Name'].str.lower() == pokemon_name.lower()]
        if not pokemon_info.empty:
            # On retourne le numéro du Pokémon, iloc[0] permet de récupérer la première ligne uniquement
            return pokemon_info.iloc[0]['Number']
        else:
            return None  

    # On crée une méthode pour prédire le Pokémon à partir d'une image
    def predict(self, image_path):
        """
        Méthode pour prédire le Pokémon à partir d'une image
        """

        # On ouvre l'image avec PIL et on la convertit en RGB
        img = Image.open(image_path).convert('RGB')

        # On applique les transformations nécessaires pour l'image : redimensionnement, normalisation, etc.
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # On applique les transformations à l'image et on ajoute une dimension pour le batch
        # transform(img) permet de transformer l'image en tenseur
        # unsqueeze(0) permet d'ajouter une dimension pour le batch
        # to(self.device) permet d'envoyer le tenseur sur l'appareil
        img_tensor = transform(img).unsqueeze(0).to(self.device)  

        # On désactive le calcul du gradient pour la prédiction
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # On récupère la classe prédite
        _, predicted_class = torch.max(output, 1)
        # On récupère le nom du Pokémon prédit
        predicted_pokemon = self.class_names[predicted_class.item()]

        # On récupère le numéro du Pokédex correspondant
        pokedex_number = self.get_pokedex_number(predicted_pokemon)

        # On retourne le nom du Pokémon prédit et son numéro de Pokédex
        return predicted_pokemon, pokedex_number
