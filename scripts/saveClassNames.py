from pathlib import Path

# On crée une classe pour générer les noms de classe et les enregistrer dans un fichier

class ClassNameGestion:
    """
    Classe pour gérer les noms de classe.
    """
    def __init__(self, train_path=None, class_file_path=None):
        """
        Initialise la classe ClassNameGestion.
        """
        self.train_path = train_path
        self.class_file_path = class_file_path

    def save_class_names(self, output_path):
        """
        Enregistre les noms de classe dans un fichier texte.
        """
        if not self.train_path:
            raise ValueError("Chemin d'entraînement non spécifié.")

        target = []
        for p in Path(self.train_path).glob('*'):
            if p.stem == 'NidoranF':
                target.append(p.stem[:-1])
            else:
                target.append(p.stem)

        with open(output_path, 'w') as f:
            for class_name in target:
                f.write(class_name + '\n')

    def load_class_names(self):
        """
        Charge les noms de classe à partir d'un fichier texte.
        """
        if not self.class_file_path:
            raise ValueError("Chemin du fichier de classes non spécifié.")

        with open(self.class_file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]