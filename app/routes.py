import os
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
from app.predictFromPicture import PokemonClassifier

UPLOAD_FOLDER = 'app/static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'model/model-94.5073.pt'  
train_path = 'dataset/images/train/'  
csv_path = 'dataset/FirstGenPokemon.csv' 

classifier = PokemonClassifier(model_path, train_path, csv_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        predicted_pokemon, pokedex_number = classifier.predict(file_path)
        
        return render_template('index.html', pokemon=predicted_pokemon, number=pokedex_number, image_url=file_path)

    return redirect(url_for('index'))
