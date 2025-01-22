const typeColors = {
    "normal": "#A8A77A",
    "fire": "#EE8130",
    "water": "#6390F0",
    "electric": "#F7D02C",
    "grass": "#7AC74C",
    "ice": "#96D9D6",
    "fighting": "#C22E28",
    "poison": "#A33EA1",
    "ground": "#E2BF65",
    "flying": "#A98FF3",
    "psychic": "#F95587",
    "bug": "#A6B91A",
    "rock": "#B6A136",
    "ghost": "#735797",
    "dragon": "#6F35FC",
    "dark": "#705746",
    "steel": "#B7B7CE",
    "fairy": "#D685AD"
};

// Récupérer les données du Pokémon de Flask

// Séparer la chaîne de types
const types = pokemonData.Type.split(" / ");

// Fonction pour afficher les images des types
const showTypesImages = () => {
    const typesContainer = document.getElementById('types-images');

    // Boucle pour ajouter une image pour chaque type
    types.forEach(type => {
        const typeId = type.toLowerCase();
        const typeImageUrl = `static/images/types/${getTypeImage(typeId)}.png`;

        const typeImageElement = document.createElement('img');
        typeImageElement.src = typeImageUrl;
        typeImageElement.alt = type;
        typeImageElement.classList.add('type-img');

        typesContainer.appendChild(typeImageElement);
    });
};

// Fonction pour obtenir le nom du fichier image selon le type
const getTypeImage = (type) => {
    const typeMapping = {
        "normal": "normal",
        "fire": "fire",
        "water": "water",
        "electric": "electric",
        "grass": "grass",
        "psychic": "psychic",
        "ice": "ice",
        "dragon": "dragon",
        "dark": "dark",
        "fairy": "fairy",
        "fighting": "fighting",
        "flying": "flying",
        "poison": "poison",
        "ground": "ground",
        "rock": "rock",
        "bug": "bug",
        "ghost": "ghost",
        "steel": "steel"
    };

    return typeMapping[type] || type; // Par défaut, on retourne le type
};

// Afficher les images des types
showTypesImages();

// Récupérer les couleurs des types
const type1Color = typeColors[types[0].toLowerCase()];
const type2Color = types[1] ? typeColors[types[1].toLowerCase()] : null;

// Appliquer le dégradé de couleurs
document.documentElement.style.setProperty('--type1-color', type1Color);
if (type2Color) {
    document.documentElement.style.setProperty('--type2-color', type2Color);
} else {
    document.documentElement.style.setProperty('--type2-color', type1Color); // Utiliser la même couleur si un seul type
}


console.log(coordonneesPokemon);

var map = L.map('map').setView([46.603354, 1.888334], 6);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: 'Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Ajouter les marqueurs sur la carte
if (Array.isArray(coordonneesPokemon) && coordonneesPokemon.length > 0) {
    coordonneesPokemon.forEach(function(point) {
        if (point.latitude && point.longitude) {
            var pokemonImageUrl = `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${number}.png`;

            var pokemonIcon = L.icon({
                iconUrl: pokemonImageUrl,
                iconSize: [80, 80], 
                iconAnchor: [16, 32], 
                popupAnchor: [0, -32] 
            });

            L.marker([point.latitude, point.longitude], { icon: pokemonIcon })
                .addTo(map)
                .bindPopup('Pokémon n° ' + number);
        } else {
            console.error('Coordonnée invalide ou manque de numéro Pokémon :', point);
        }
    });
} else {
    console.error('coordonneesPokemon n\'est pas un tableau valide ou est vide');
}