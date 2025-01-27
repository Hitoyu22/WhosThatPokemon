from pathlib import Path

class ClassNameGestion:
    def __init__(self, train_path=None, class_file_path=None):
        """
        Initialise l'objet avec les chemins des dossiers ou fichiers.

        :param train_path: Chemin vers le dossier contenant les fichiers d'entraînement (facultatif).
        :param class_file_path: Chemin vers le fichier contenant les noms des classes (facultatif).
        """
        self.train_path = train_path
        self.class_file_path = class_file_path

    def save_class_names(self, output_path):
        """
        Générez et enregistrez les noms de classe à partir des fichiers dans le dossier d'entraînement.

        :param output_path: Chemin où les noms des classes doivent être sauvegardés.
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
        Chargez les noms de classe à partir du fichier.

        :return: Liste des noms de classes.
        """
        if not self.class_file_path:
            raise ValueError("Chemin du fichier de classes non spécifié.")

        with open(self.class_file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]