import os
import shutil

# On crée une classe DirectoryCleaner pour nettoyer le répertoire des images uploadées
class DirectoryCleaner:
    """
    Classe pour nettoyer un répertoire.
    """
    def __init__(self, directory_path):
        """
        Initialise la classe DirectoryCleaner.
        """
        self.directory_path = directory_path

    # On crée une méthode clean pour nettoyer le répertoire
    def clean(self):
        """
        Nettoie le répertoire spécifié.
        """
        if not os.path.exists(self.directory_path):
            print(f"The directory {self.directory_path} does not exist.")
            return

        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":
    # Chemin du répertoire à nettoyer
    directory_path = "./app/static/uploads/"
    cleaner = DirectoryCleaner(directory_path)
    cleaner.clean()
