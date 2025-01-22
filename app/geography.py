import pandas as pd
import json

class PokemonMap:
    def __init__(self, fichier_csv):
        # Charger le fichier CSV
        self.df = pd.read_csv(fichier_csv, sep=',')
        # Définir les limites géographiques approximatives de l'Île-de-France
        self.limites_ile_de_france = {
            'lat_min': 48.5, 
            'lat_max': 49.5,
            'lon_min': 2.2,
            'lon_max': 2.7
        }
    
    def obtenir_donnees_pokemon(self, id_pokemon, nb_echantillons=200):
        # Filtrer par l'ID du Pokémon
        df_pokemon = self.df[self.df['pokemonId'] == id_pokemon]
        
        # Filtrer les Pokémon en Île-de-France
        df_pokemon_ile_de_france = df_pokemon[
            (df_pokemon['latitude'] >= self.limites_ile_de_france['lat_min']) & 
            (df_pokemon['latitude'] <= self.limites_ile_de_france['lat_max']) & 
            (df_pokemon['longitude'] >= self.limites_ile_de_france['lon_min']) & 
            (df_pokemon['longitude'] <= self.limites_ile_de_france['lon_max'])
        ]
        
        # Calculer combien de lignes prendre d'Île-de-France et du reste
        nb_ile_de_france = min(len(df_pokemon_ile_de_france), nb_echantillons // 2)
        nb_restant = min(nb_echantillons - nb_ile_de_france, len(df_pokemon) - nb_ile_de_france)
        
        # Prendre un échantillon aléatoire de l'Île-de-France
        echantillon_ile_de_france = df_pokemon_ile_de_france.sample(n=nb_ile_de_france, random_state=1)
        
        # Prendre un échantillon aléatoire du reste
        echantillon_restant = df_pokemon.drop(df_pokemon_ile_de_france.index).sample(n=nb_restant, random_state=1)
        
        # Combiner les deux échantillons
        echantillon_combine = pd.concat([echantillon_ile_de_france, echantillon_restant])
        
        # Convertir en JSON
        donnees_json = echantillon_combine[['latitude', 'longitude']].to_dict(orient='records')

        print(donnees_json)
        
        # Retourner les données JSON
        return donnees_json
