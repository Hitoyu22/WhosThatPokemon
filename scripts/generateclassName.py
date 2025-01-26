from saveClassNames import save_class_names

class ClassNameSaver:
    def __init__(self, dataset_directory, output_file):
        self.dataset_directory = dataset_directory
        self.output_file = output_file

    def save(self):
        save_class_names(self.dataset_directory, self.output_file)

if __name__ == "__main__":
    # Définir le répertoire du dataset et le fichier de sortie pour les noms de classes
    dataset_directory = './dataset/train'
    output_file = './dataset/class_names.txt'

    # Créer une instance de ClassNameSaver et appeler la méthode save
    class_name_saver = ClassNameSaver(dataset_directory, output_file)
    class_name_saver.save()
