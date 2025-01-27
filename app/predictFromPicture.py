import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import pandas as pd
from pathlib import Path
from scripts.saveClassNames import ClassNameGestion

className = ClassNameGestion(train_path="./dataset/train", class_file_path="./dataset/class_names.txt")

class_file_path = './dataset/class_names.txt'

class PokemonClassifier:
    def __init__(self, model_path, csv_path):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_names = className.load_class_names()
        self.model = EfficientNet.from_pretrained("efficientnet-b2")
        self.model._fc = torch.nn.Linear(1408, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval() 
        
        self.pokemon_data = pd.read_csv(csv_path)

    

    def get_pokedex_number(self, pokemon_name):

        pokemon_info = self.pokemon_data[self.pokemon_data['Name'].str.lower() == pokemon_name.lower()]
        if not pokemon_info.empty:
            return pokemon_info.iloc[0]['Number']
        else:
            return None  

    def predict(self, image_path):

        img = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img_tensor = transform(img).unsqueeze(0).to(self.device)  

        with torch.no_grad():
            output = self.model(img_tensor)
        
        _, predicted_class = torch.max(output, 1)
        predicted_pokemon = self.class_names[predicted_class.item()]

        pokedex_number = self.get_pokedex_number(predicted_pokemon)

        return predicted_pokemon, pokedex_number
