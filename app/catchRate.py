
import math as math

class catchRate:
    """
    Classe permettant de calculer le taux de capture d'un Pokémon
    """
    def __init__(self, PokemonValue, BallValue, LevelValue, PvValue, StatusValue, PvMax ):
        """
        Constructeur de la classe catchRate
        """
        self.PokemonValue = PokemonValue
        self.BallValue = BallValue
        self.LevelValue = LevelValue
        self.StatusValue = StatusValue

        if (PvValue > PvMax):
            self.PvValue = PvMax
        else:
            self.PvValue = PvValue

        self.PvMaxValue = PvMax + 2*LevelValue

    def ValueToImpact(self):
        """
        Méthode pour convertir les valeurs en impact
        """
        #Pokémon Value
        if (self.PokemonValue == 1): #Premier de la lignée
            self.PokemonImpact = 4
        elif (self.PokemonValue == 2): # Deuxième de la lignée
            self.PokemonImpact = 2
        elif (self.PokemonValue == 3): # Troisième de la lignée
            self.PokemonImpact = 1
        elif (self.PokemonValue == 4): # Cas du légendaire
            self.PokemonImpact = 0.25
        
        #Ball Value
        if 1 <= self.BallValue <= 3: #Pokéball =1/ Superball =2/ Hyperball = 3
            self.BallImpact = self.BallValue
        elif (self.BallValue == 4): # Ultraball = 5
            self.BallImpact = 5
        elif (self.BallValue == 5): # Masterball ne peut pas rater
            self.BallImpact = 255

        #Level Value
        self.LevelImpact = 1 + ((100 - self.LevelValue) / 100) / 4

        #PV Value
        self.PvImpact = (2 - self.PvValue / self.PvMaxValue) *4

        #Status Value
        if (self.StatusValue == 0): # Aucun status
            self.StatusImpact = 1
        elif (self.StatusValue >= 1 & self.StatusValue <= 3): # Status Paralysie Brulure ou poison
            self.StatusImpact = 1.5
        elif (self.StatusValue == 4 or self.StatusValue == 5): # Status Sommeil ou Gel
            self.StatusImpact = 2.5

        
    def calculCatchRate(self):
        """
        Méthode pour calculer le taux de capture
        """
        self.ValueToImpact()
        value = math.floor((1 * self.PokemonImpact * self.BallImpact * self.LevelImpact * self.PvImpact * self.StatusImpact) * 100) / 100
        print(value)
        return value
        #Valeur retournée arrondi a 10^-2 pour un affichage aux petits ognions
