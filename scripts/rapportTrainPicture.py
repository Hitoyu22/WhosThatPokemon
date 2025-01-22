import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime


class RapportPDF:
    def __init__(self, chemin_repartition, chemin_statistiques, chemin_batch_stats, chemin_rapport):
        self.chemin_repartition = chemin_repartition
        self.chemin_statistiques = chemin_statistiques
        self.chemin_batch_stats = chemin_batch_stats
        self.chemin_rapport = chemin_rapport

        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

        # Dossier pour enregistrer les graphiques
        self.graphs_dir = "./pdf/modeleImage/graphs"
        # Créer le dossier des graphiques s'il n'existe pas
        os.makedirs(self.graphs_dir, exist_ok=True)

    def generer_graphiques(self):
        # Charger les données
        repartition_df = pd.read_csv(self.chemin_repartition)
        stats_df = pd.read_csv(self.chemin_statistiques)

        # Diviser les données de répartition en trois groupes
        num_classes = len(repartition_df)
        chunk_size = num_classes // 3

        repartition_graph_paths = []
        for i in range(3):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 2 else num_classes  # Le dernier graphique prend le reste des classes
            chunk_df = repartition_df.iloc[start:end]

            # Graphique de répartition des images par classe
            plt.figure(figsize=(10, 6))
            plt.bar(chunk_df['Classe'], chunk_df['Nombre d\'images'])
            plt.xticks(rotation=90)
            plt.xlabel("Classes")
            plt.ylabel("Nombre d'images")
            plt.title(f"Répartition des images par classe - Partie {i+1}")
            graph_path = os.path.join(self.graphs_dir, f"repartition_graph_part_{i+1}.png")
            plt.savefig(graph_path, bbox_inches="tight")
            plt.close()

            repartition_graph_paths.append(graph_path)

        # Graphique de l'évolution des performances du modèle
        plt.figure(figsize=(10, 6))
        plt.plot(stats_df['epoch'], stats_df['accuracy_val'], label="Précision de validation", color='blue')
        plt.plot(stats_df['epoch'], stats_df['perte_entrainement'], label="Loss d'entraînement", color='red')
        plt.plot(stats_df['epoch'], stats_df['perte_val'], label="Loss de validation", color='green')
        plt.xlabel("Epochs")
        plt.ylabel("Valeurs")
        plt.title("Évolution des performances du modèle")
        plt.legend()
        stats_graph_path = os.path.join(self.graphs_dir, "stats_graph.png")
        plt.savefig(stats_graph_path, bbox_inches="tight")
        plt.close()

        return repartition_graph_paths, stats_graph_path

    def generer_graphiques_epoch(self):
        # Charger les données de batch stats
        batch_stats_df = pd.read_csv(self.chemin_batch_stats)

        # Pour chaque époque, générer un graphique de perte et d'accuracy par rapport aux batches
        epoch_graph_paths = []
        for epoch in batch_stats_df['epoch'].unique():
            epoch_df = batch_stats_df[batch_stats_df['epoch'] == epoch]

            plt.figure(figsize=(10, 6))
            plt.plot(epoch_df['batch'], epoch_df['perte_entrainement'], label="Loss d'entraînement", color='red')
            plt.plot(epoch_df['batch'], epoch_df['accuracy_entrainement'], label="Accuracy d'entraînement", color='blue')
            plt.xlabel("Batch")
            plt.ylabel("Valeurs")
            plt.title(f"Évolution de la Perte et de l'Accuracy pour l'Époque {epoch}")
            plt.legend()

            epoch_graph_path = os.path.join(self.graphs_dir, f"epoch_{epoch}_graph.png")
            plt.savefig(epoch_graph_path, bbox_inches="tight")
            plt.close()

            epoch_graph_paths.append(epoch_graph_path)

        return epoch_graph_paths

    def ajouter_page_de_couverture(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=20, style="B")
        self.pdf.cell(0, 10, txt="Rapport d'Apprentissage du Modèle", ln=True, align="C")
        self.pdf.ln(20)  # Ajouter un espacement après le titre

        # Ajouter les noms des membres de l'équipe
        self.pdf.set_font("Arial", size=16)
        self.pdf.cell(0, 10, txt="Membres de l'équipe:", ln=True, align="C")
        self.pdf.ln(10)  # Ajouter un espacement avant la liste des noms

        # Ajouter les noms des membres un par un
        self.pdf.cell(0, 10, txt="Rémy THIBAUT", ln=True, align="C")
        self.pdf.cell(0, 10, txt="Quentin DELNEUF", ln=True, align="C")
        self.pdf.cell(0, 10, txt="Damien VAURETTE", ln=True, align="C")
        self.pdf.ln(20)

        self.pdf.set_font("Arial", size=12)
        date_rapport = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        self.pdf.cell(0, 10, txt=f"Généré le {date_rapport}", ln=True, align="C")
        self.pdf.ln(20)

        # Ajouter le texte explicatif sous le titre
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 8, txt="""
    Le modèle utilisé pour cette tâche de classification d'images repose sur l'architecture EfficientNet pré-entraînée, spécifiquement le modèle EfficientNet-B2. EfficientNet est une architecture de réseau neuronal convolutif (CNN) optimisée pour atteindre un compromis optimal entre efficacité et performance. Cette architecture a été choisie pour sa capacité à fournir des résultats précis tout en étant relativement plus légère que d'autres architectures classiques comme ResNet ou VGG.

    Le modèle a été ajusté pour cette tâche en modifiant sa couche de sortie. Par défaut, la couche de sortie d'EfficientNet-B2 contient 1408 unités, adaptées pour une tâche de classification sur un large nombre de classes. Pour cette application, cette couche a été remplacée par une nouvelle couche entièrement connectée (_fc) avec une sortie correspondant au nombre de classes dans l'ensemble de données, qui est dynamiquement déterminé à partir des sous-dossiers des images d'entraînement.

    Librairies utilisées :
    - PyTorch : Utilisé pour la construction, l'entraînement et la gestion du modèle. PyTorch est une bibliothèque flexible et puissante pour le calcul des gradients et l'optimisation des réseaux neuronaux.
    - EfficientNet-PyTorch : Une implémentation de l'architecture EfficientNet, permettant de charger directement des modèles pré-entraînés et de les adapter à des tâches spécifiques de classification.
    - PIL (Python Imaging Library) et Torchvision : Utilisées pour les transformations d'images, telles que le redimensionnement et la normalisation, avant d'envoyer les images dans le modèle.
    - TQDM : Une bibliothèque utilisée pour ajouter une barre de progression dans les boucles d'entraînement et de validation, facilitant le suivi de l'avancement de l'entraînement.

    Paramètres de l'entraînement :
    - Optimiseur : Adam, utilisé pour la mise à jour des poids du modèle.
    - Critère de perte : CrossEntropyLoss, adapté pour les tâches de classification multi-classes.
    - Scheduler : CyclicLR pour ajuster dynamiquement le taux d'apprentissage pendant l'entraînement.
    """)

    def ajouter_graphique(self, titre, image_path):
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=16, style="B")
        self.pdf.cell(0, 10, txt=titre, ln=True, align="C")
        self.pdf.ln(10)
        self.pdf.image(image_path, x=10, y=30, w=190)

    def ajouter_statistiques(self):
        stats_df = pd.read_csv(self.chemin_statistiques)

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=16, style="B")
        self.pdf.cell(0, 10, txt="Statistiques d'Apprentissage", ln=True, align="C")
        self.pdf.ln(10)

        # Réduire la taille de la police pour les lignes de statistiques
        self.pdf.set_font("Arial", size=12)
        
        # Ajouter un en-tête pour la section des statistiques
        self.pdf.cell(40, 10, 'Epoque', border=1, align='C')
        self.pdf.cell(40, 10, 'Perte d\'Entraînement', border=1, align='C')
        self.pdf.cell(40, 10, 'Perte de Validation', border=1, align='C')
        self.pdf.cell(60, 10, 'Précision de Validation (%)', border=1, align='C')
        self.pdf.ln()

        # Ajouter les données de chaque ligne dans le DataFrame
        for index, row in stats_df.iterrows():
            self.pdf.cell(40, 10, str(row['epoch']), border=1, align='C')
            self.pdf.cell(40, 10, f"{row['perte_entrainement']:.4f}", border=1, align='C')
            self.pdf.cell(40, 10, f"{row['perte_val']:.4f}", border=1, align='C')
            self.pdf.cell(60, 10, f"{row['accuracy_val']:.2f}", border=1, align='C')
            self.pdf.ln()

        # Ajouter un espacement après la section des statistiques
        self.pdf.ln(10)

    def generer_pdf(self):
        # Ajouter une page de couverture
        self.ajouter_page_de_couverture()

        # Générer les graphiques de répartition et de performance
        repartition_graph_paths, stats_graph_path = self.generer_graphiques()

        # Ajouter les graphiques au PDF
        for i, graph_path in enumerate(repartition_graph_paths):
            self.ajouter_graphique(f"Répartition des Images par Classe - Partie {i+1}", graph_path)

        self.ajouter_graphique("Évolution des Performances du Modèle", stats_graph_path)

        # Ajouter les graphiques de chaque époque
        epoch_graph_paths = self.generer_graphiques_epoch()
        for graph_path in epoch_graph_paths:
            self.ajouter_graphique(f"Évolution de la Perte et de l'Accuracy pour l'Époque", graph_path)

        # Ajouter les statistiques d'apprentissage
        self.ajouter_statistiques()

        # Sauvegarder le PDF
        self.pdf.output(self.chemin_rapport)
        print(f"Rapport PDF généré: {self.chemin_rapport}")


