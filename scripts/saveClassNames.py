
from pathlib import Path

class_file_path = './dataset/class_names.txt'

def save_class_names(train_path, output_path):
        target = []
        for p in Path(train_path).glob('*'):
            if p.stem == 'NidoranF':
                target.append(p.stem[:-1])
            else:
                target.append(p.stem)

        with open(output_path, 'w') as f:
            for class_name in target:
                f.write(class_name + '\n')


save_class_names('dataset/images/train/', class_file_path)

def load_class_names(class_file_path):
    with open(class_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]