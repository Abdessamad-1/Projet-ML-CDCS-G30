from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

# Charger les modèles pré-entraînés
models = {
    "logistic_regression": pickle.load(open('logistic_regression.pkl', 'rb')),
    "decision_tree": pickle.load(open('decision_tree.pkl', 'rb')),
    "knn": pickle.load(open('knn.pkl', 'rb')),
    "svm": pickle.load(open('svm.pkl', 'rb'))
}

# Initialiser l'application Flask
app = Flask(__name__)

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour faire une prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'csvFile' in request.files:
        csv_file = request.files['csvFile']
        df = pd.read_csv(csv_file)

        # Récupérer les caractéristiques du fichier CSV
        features = df.values.tolist()

        # Faire des prédictions avec chaque modèle
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(features).tolist()

        return jsonify(predictions)

    return render_template('index.html', error="Veuillez sélectionner un fichier CSV")

if __name__ == "__main__":
    app.run(debug=True)
