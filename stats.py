import pandas as pd
import matplotlib.pyplot as plt
import json

class Pokemon:
    def __init__(self, data_row, dataset):
        
        self.dataset = dataset 
        self.number = int(data_row['Number'])
        self.name = data_row['Nom']
        self.type1 = data_row['Type1']
        self.type2 = data_row['Type2'] if pd.notna(data_row['Type2']) else None
        self.height = data_row['Height(m)']
        self.weight = data_row['Weight(kg)']
        self.sub_evolution = data_row['Sous_Evolution'] if pd.notna(data_row['Sous_Evolution']) else None
        self.evolution = data_row['Evolution'] if pd.notna(data_row['Evolution']) else None
        self.description = data_row['description'] if pd.notna(data_row['description']) else None

        # Trouver les IDs des sous-évolutions et des évolutions
        self.sub_evolution_id = self.get_pokemon_id_by_name(self.sub_evolution)
        self.evolution_id = self.get_pokemon_id_by_name(self.evolution)

        # Statistiques pour les graphiques
        self.radar_stats = {
            'Poison_Dmg': data_row['Poison_Dmg'],
            'Ground_Dmg': data_row['Ground_Dmg'],
            'Flying_Dmg': data_row['Flying_Dmg'],
            'Psychic_Dmg': data_row['Psychic_Dmg'],
            'Bug_Dmg': data_row['Bug_Dmg'],
            'Rock_Dmg': data_row['Rock_Dmg'],
            'Ghost_Dmg': data_row['Ghost_Dmg'],
            'Dragon_Dmg': data_row['Dragon_Dmg']
        }
        self.bar_stats = {
            'HP': data_row['HP'],
            'Attack': data_row['Attack'],
            'Defense': data_row['Defense'],
            'Special': data_row['Special'],
            'Speed': data_row['Speed']
        }

    # Fonction pour retrouver l'ID des pokemons en fonction de leur nom en français (utiliser pour retrouver les ids des évolutions)
    def get_pokemon_id_by_name(self, name):
        
        if name:
            row = self.dataset[self.dataset['Nom'] == name]
            if not row.empty:
                return int(row['Number'].iloc[0])
        return None

    def to_json(self, output_path="pokemon.json"):
        
        data = {
            "Number": self.number,
            "Name": self.name,
            "Type": f"{self.type1}" + (f" / {self.type2}" if self.type2 else ""),
            "Height": f"{self.height} m",
            "Weight": f"{self.weight} kg",
            "Sub_Evolution": self.sub_evolution if self.sub_evolution else "None",
            "Sub_Evolution_ID": self.sub_evolution_id,
            "Evolution": self.evolution if self.evolution else "None",
            "Evolution_ID": self.evolution_id,
            "Description": self.description if self.description else "No description available"
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return data

    # Graphique radar pour les dégâts reçus par type d'attaque
    def display_radar(self, output_path="radar_stats.png"):
        
        radar_values = list(self.radar_stats.values())
        radar_labels = list(self.radar_stats.keys())

        radar_values += radar_values[:1]
        angles = [n / float(len(radar_labels)) * 2 * 3.14159 for n in range(len(radar_labels))]
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], radar_labels, color='grey', size=8)

        ax.plot(angles, radar_values, linewidth=1, linestyle='solid')
        ax.fill(angles, radar_values, alpha=0.4)
        plt.title(f"Multiplicateur des dégats reçus par {self.name} selon le type de l'attaque", size=15, color='blue', y=1.1)

        plt.tight_layout()
        plt.savefig(output_path)  # Sauvegarder le graphique radar
        plt.close()

    # Graphique en barres pour les statistiques de base
    def display_bar(self, output_path="bar_stats.png"):
        
        bar_values = list(self.bar_stats.values())
        bar_labels = list(self.bar_stats.keys())

        plt.figure(figsize=(6, 6))
        plt.bar(bar_labels, bar_values, color='skyblue')
        plt.title(f"Statistiques de base de {self.name}", size=15, color='blue')
        plt.ylabel("Value")

        plt.tight_layout()
        plt.savefig(output_path)  # Sauvegarder le graphique en barres
        plt.close()

file_path = 'dataset/PokemonPremiereGen_2.csv' # Chemin du dataset
data = pd.read_csv(file_path, sep=';', encoding='latin1')

# Nettoyer les colonnes
data.columns = data.columns.str.strip()

pokemon_id = 1 # Variable à changer pour afficher un autre Pokémon
pokemon_row = data[data['Number'] == pokemon_id]

if not pokemon_row.empty:
    pokemon = Pokemon(pokemon_row.iloc[0], data)  
    json_data = pokemon.to_json(output_path="Bulbizarre.json")  # chemin pour l'export du JSON
    pokemon.display_radar(output_path="Bulbizarre_radar.png")  # chemin pour l'export du graphique radar
    pokemon.display_bar(output_path="Bulbizarre_bar.png")  # chemin pour l'export du graphique en barres
    print(json.dumps(json_data, ensure_ascii=False, indent=4))  # Affichage du JSON pour vérifier le contenu
else:
    print(f"No Pokémon found with ID {pokemon_id}")
