import os
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
from app.predictFromPicture import PokemonClassifier
from app.stats import Pokemon
from app.geography import PokemonMap
import pandas as pd
UPLOAD_FOLDER = 'app/static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

train_path = 'dataset/images/train/'  
csv_path = 'dataset/FirstGenPokemon.csv' 



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    # Chemin du dossier contenant les modèles
    model_dir = 'model/'
    model_files = []

    # Parcourir les fichiers dans le dossier
    for filename in os.listdir(model_dir):
        if filename.startswith("model-") and filename.endswith(".pt"):
            # Extraire les informations du nom de fichier
            parts = filename.split('-')
            accuracy = parts[1]
            formatted_name = f"Modèle à {accuracy}%"
            
            model_files.append((filename, formatted_name))
    
    return render_template('index.html', model_files=model_files)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        model_path = request.form['model']

        entire_model_path = os.path.join('model/', model_path)
        classifier = PokemonClassifier(entire_model_path, csv_path)
        
        predicted_pokemon, pokedex_number = classifier.predict(file_path)

        datasetPokemonData = pd.read_csv('dataset/PokemonPremiereGen_2.csv', sep=';', encoding='latin-1')

        datasetPokemonData.columns = datasetPokemonData.columns.str.strip()
        pokemonSelect = datasetPokemonData[datasetPokemonData['Number'] == pokedex_number]

        pokemonData = Pokemon(pokemonSelect.iloc[0], datasetPokemonData)

        pokemonLocalisation = PokemonMap('dataset/localisationPokemon.csv')
        coordonneesPokemon = pokemonLocalisation.obtenir_donnees_pokemon(pokedex_number, nb_echantillons=200)

        # Lancement des différents traitements
        pokemonData.display_radar(output_path='app/static/radar_stats.png')
        pokemonData.display_bar(output_path='app/static/bar_stats.png')


        
        return render_template(
            'upload.html', 
            pokemon=predicted_pokemon, 
            number=pokedex_number, 
            image_url=file_path,
            pokemonData=pokemonData.to_json(output_path='app/static/pokemon.json'),
            coordonneesPokemon=coordonneesPokemon
        )

    return redirect(url_for('index'))
