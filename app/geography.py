import pandas as pd
import json

class PokemonMap:
    def __init__(self, fichier_csv):
        """
        Constructeur de la classe PokemonMap
        """
        # Nous chargeons les données depuis un fichier CSV
        self.df = pd.read_csv(fichier_csv, sep=',')
        # Pour cibler les Pokémon en Île-de-France, nous définissons les limites géographiques à partir de la longitude et de la latitude
        self.limites_ile_de_france = {
            'lat_min': 48.5, 
            'lat_max': 49.5,
            'lon_min': 2.2,
            'lon_max': 2.7
        }
    
    def obtenir_donnees_pokemon(self, id_pokemon, nb_echantillons=200):
        """
        Méthode pour obtenir les données d'un Pokémon donné
        """
        # On récupère toutes les lignes correspondant à un Pokémon donné
        df_pokemon = self.df[self.df['pokemonId'] == id_pokemon]
        
        # On récupère les lignes qui sont en Île-de-France dans un premier temps
        df_pokemon_ile_de_france = df_pokemon[
            (df_pokemon['latitude'] >= self.limites_ile_de_france['lat_min']) & 
            (df_pokemon['latitude'] <= self.limites_ile_de_france['lat_max']) & 
            (df_pokemon['longitude'] >= self.limites_ile_de_france['lon_min']) & 
            (df_pokemon['longitude'] <= self.limites_ile_de_france['lon_max'])
        ]
        
        # On calcule le nombre de lignes obtenu pour l'Île-de-France et le reste
        nb_ile_de_france = min(len(df_pokemon_ile_de_france), nb_echantillons // 2)
        nb_restant = min(nb_echantillons - nb_ile_de_france, len(df_pokemon) - nb_ile_de_france)
        
        # On prend un échantillon aléatoire en Île-de-France
        echantillon_ile_de_france = df_pokemon_ile_de_france.sample(n=nb_ile_de_france, random_state=1)
        
        # on prend un échantillon aléatoire du reste de la France
        echantillon_restant = df_pokemon.drop(df_pokemon_ile_de_france.index).sample(n=nb_restant, random_state=1)
        
        # on combine les deux échantillons avec concat : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
        echantillon_combine = pd.concat([echantillon_ile_de_france, echantillon_restant])
        
        # POur les récupérer dans le front, nous allons les transformer en JSON
        donnees_json = echantillon_combine[['latitude', 'longitude']].to_dict(orient='records')

        print(donnees_json)
        
        # On retourne les données au format JSON
        return donnees_json
