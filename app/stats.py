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
        self.hp = data_row['HP']
        self.sub_evolution = data_row['Sous_Evolution'] if pd.notna(data_row['Sous_Evolution']) else None
        self.evolution = data_row['Evolution'] if pd.notna(data_row['Evolution']) else None
        self.legendary = self.legendary_bool(data_row['Legendaire'])
        self.description = data_row['description'] if pd.notna(data_row['description']) else None

        # Trouver les IDs des sous-évolutions et des évolutions
        self.sub_evolution_id = self.get_pokemon_id_by_name(self.sub_evolution)
        self.evolution_id = self.get_pokemon_id_by_name(self.evolution)
        self.sub_evolution_2 = self.get_sub_evolution(self.sub_evolution) if self.sub_evolution else None
        self.sub_evolution_2_id = self.get_pokemon_id_by_name(self.sub_evolution_2) if self.sub_evolution_2 else None
        self.evolution_2 = self.get_evolution(self.evolution) if self.evolution else None
        self.evolution_2_id = self.get_pokemon_id_by_name(self.evolution_2) if self.evolution_2 else None
        self.step = self.step_evolution(self.sub_evolution, self.sub_evolution_2, self.legendary)

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
    
    def get_sub_evolution(self, name):
        row = self.dataset[self.dataset['Nom'] == name]
        if not row.empty:
            sub_evolution_2 = row['Sous_Evolution'].iloc[0]
            if sub_evolution_2 is not None:
                return sub_evolution_2
        
    def get_evolution(self, name):
        row = self.dataset[self.dataset['Nom'] == name]
        if not row.empty:
            evolution_2 = row['Evolution'].iloc[0]
            if evolution_2 is not None:
                return evolution_2
    
    def legendary_bool(self, val):
        if val == 1.0:
            return True
        else:
            return False
        
    def step_evolution(self, sub_evolution, sub_evolution_2, legendary):
        if sub_evolution is not None:
            step = 2
            if sub_evolution_2 is not None:
                step = 3
        elif legendary:
            step = 4
        else:
            step = 1
        return step

    def to_json(self, output_path="pokemon.json"):
        
        data = {
            "Number": self.number,
            "Name": self.name,
            "Type": f"{self.type1}" + (f" / {self.type2}" if self.type2 else ""),
            "Height": f"{self.height} m",
            "Weight": f"{self.weight} kg",
            "hp": self.hp,
            "Sub_Evolution_2": self.sub_evolution_2 if self.sub_evolution_2 else "None",
            "Sub_Evolution_2_ID": self.sub_evolution_2_id,
            "Sub_Evolution": self.sub_evolution if self.sub_evolution else "None",
            "Sub_Evolution_ID": self.sub_evolution_id,
            "Evolution": self.evolution if self.evolution else "None",
            "Evolution_ID": self.evolution_id,
            "Evolution_2": self.evolution_2 if self.evolution_2 else "None",
            "Evolution_2_ID": self.evolution_2_id,
            "Legendaire": self.legendary,
            "step": self.step,
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
        plt.title(f"Multiplicateur des dégats reçus par {self.name}\n selon le type de l'attaque", size=15, color='black', y=1.1)

        plt.tight_layout()
        plt.savefig(output_path)  # Sauvegarder le graphique radar
        plt.close()

    # Graphique en barres pour les statistiques de base
    def display_bar(self, output_path="bar_stats.png"):
        custom_labels = ['PV', 'Attaque', 'Défense', 'Spécial', 'Vitesse']
        
        ordered_stats = [
            self.bar_stats['HP'],       
            self.bar_stats['Attack'],   
            self.bar_stats['Defense'],  
            self.bar_stats['Special'],  
            self.bar_stats['Speed'],    
        ]
        
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FFD700']  

        plt.figure(figsize=(6, 6))
        plt.bar(custom_labels, ordered_stats, color=colors)  
        plt.title(f"Statistiques de base de {self.name}", size=15, color='black')
        plt.ylabel("Valeur", size=12)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
