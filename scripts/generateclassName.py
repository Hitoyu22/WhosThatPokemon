from saveClassNames import save_class_names


# On crée une classe ClassNameSaver pour sauvegarder les noms des classes
# L'ordre des classes varie d'un ordinateur à l'autre, il est donc important de les sauvegarder pour que les modèles soient toujours compatibles
class ClassNameSaver:
    def __init__(self, dataset_directory, output_file):
        self.dataset_directory = dataset_directory
        self.output_file = output_file

    # On crée une méthode save pour sauvegarder les noms des classes
    def save(self):
        save_class_names(self.dataset_directory, self.output_file)

if __name__ == "__main__":
    dataset_directory = './dataset/train'
    output_file = './dataset/class_names.txt'

    class_name_saver = ClassNameSaver(dataset_directory, output_file)
    class_name_saver.save()
