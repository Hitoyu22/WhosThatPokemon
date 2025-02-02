import os
from flask import jsonify, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
from app.predictFromPicture import PokemonClassifier
from app.stats import Pokemon
from app.geography import PokemonMap
import pandas as pd
from app.catchRate import catchRate


# On définit le dossier où les images seront stockées
UPLOAD_FOLDER = 'app/static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

train_path = 'dataset/images/train/'  
csv_path = 'dataset/FirstGenPokemon.csv' 


# On définit une fonction pour vérifier si le fichier est autorisé
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# On définit la route pour la page d'accueil ("/")
@app.route('/')
def index():
    """
    Route pour la page d'accueil
    """


    # On définit le dossier où les modèles sont stockés
    model_dir = 'model/'
    model_files = []

    # On parcourt les fichiers du dossier
    for filename in os.listdir(model_dir):
        if filename.startswith("model-") and filename.endswith(".pt"):
            # On extrait le taux d'accuracy du nom du fichier
            parts = filename.split('-')
            accuracy = float(parts[1])
            formatted_name = f"Modèle à {accuracy}%"
            
            model_files.append((filename, formatted_name, accuracy))
    
    # On trie les fichiers par ordre décroissant d'accuracy
    # sorted() permet de trier une liste, key permet de définir la fonction de tri
    # On utilise une fonction lambda pour trier par le troisième élément du tuple (l'accuracy)
    model_files = sorted(model_files, key=lambda x: x[2], reverse=True)
    
    # On ne garde que le nom formaté et le nom du fichier
    model_files = [(filename, formatted_name) for filename, formatted_name, _ in model_files]
    
    # On retourne le template index.html en lui passant les modèles
    return render_template('index.html', model_files=model_files)

# On définit la route pour la page d'upload ("/upload") : on récupère l'image et on la prédit
@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Route pour uploader une image et prédire le Pokémon
    """


    # On vérifie si une image a été envoyée
    if 'file' not in request.files:
        return redirect(request.url)
    
    # On récupère le fichier
    file = request.files['file']
    
    # On vérifie si le fichier est vide
    if file and allowed_file(file.filename):
        # On sécurise le nom du fichier
        # secure_filename() permet de sécuriser le nom du fichier pour éviter des incompréhensions entre les différents programmes
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # On récupère le modèle choisi
        model_path = request.form['model']

        # On charge le modèle choisit
        entire_model_path = os.path.join('model/', model_path)
        # On crée une instance de PokemonClassifier
        classifier = PokemonClassifier(entire_model_path, csv_path)
        
        # On récupère le nom et le numéro du Pokémon prédit
        predicted_pokemon, pokedex_number = classifier.predict(file_path)

        datasetPokemonData = pd.read_csv('dataset/PokemonPremiereGen_2.csv', sep=';', encoding='latin-1')

        # On nettoie les colonnes pour éviter les problèmes de casses avec str.strip() qui permet de supprimer les espaces en début et fin de chaîne
        datasetPokemonData.columns = datasetPokemonData.columns.str.strip()
        pokemonSelect = datasetPokemonData[datasetPokemonData['Number'] == pokedex_number]

        pokemonData = Pokemon(pokemonSelect.iloc[0], datasetPokemonData)
        # On crée une instance de PokemonMap pour récupérer les données de localisation du pokémon prédit
        pokemonLocalisation = PokemonMap('dataset/localisationPokemon.csv')
        coordonneesPokemon = pokemonLocalisation.obtenir_donnees_pokemon(pokedex_number, nb_echantillons=200)

        # Lancement des différents traitements
        pokemonData.display_radar(output_path='app/static/radar_stats.png')
        pokemonData.display_bar(output_path='app/static/bar_stats.png')


        # On retourne le template upload.html en lui passant le nom du Pokémon prédit, le numéro du Pokédex, l'image et les données de localisation
        return render_template(
            'upload.html', 
            pokemon=predicted_pokemon, 
            number=pokedex_number, 
            image_url=file_path,
            pokemonData=pokemonData.to_json(output_path='app/static/pokemon.json'),
            coordonneesPokemon=coordonneesPokemon
        )

    return redirect(url_for('index'))

# On définit la route d'une API pour calculer le taux de capture d'un pokemon en fonction de différents paramètres, il ne s'agit pas d'une page mais d'un retour json uniquement
@app.route('/catch', methods=['POST'])
def catch():
    """
    Route pour calculer le taux de capture d'un Pokémon
    """

    # On récupère les données envoyées en POST
    data = request.json  
    pokemon_level = int(data.get('pokemonLevel', 0))  
    pokeball = int(data.get('pokeball',0)) 
    pokemon_health = int(data.get('pokemonHealth', 0))
    pokemon_status = int(data.get('pokemonStatus', 1))
    pokemon_value = int(data.get('pokemonValue', 1))
    pokemon_hp = int(data.get('pokemonHp', 0))

    print(pokemon_level, pokeball, pokemon_health, pokemon_status)

    # On crée une instance de catchRate avec les données récupérées
    catch = catchRate(pokemon_value, pokeball, pokemon_level, pokemon_health, pokemon_status, pokemon_hp)
    
    # On calcule le taux de capture
    result =  catch.calculCatchRate() 

    # On retourne le résultat au format JSON
    return jsonify(result=result)
