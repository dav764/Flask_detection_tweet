from pydantic import BaseModel, ValidationError
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

import nltk
nltk.data.path.append("C:\\Users\\ACER\\myproject\\mon_dossier_nltk")
nltk.download('words', download_dir="C:\\Users\\ACER\\myproject\\mon_dossier_nltk")

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import words
english_words = set(words.words())

# Téléchargement des ressources NLTK
nltk.download('stopwords')
nltk.download('wordnet')


# Prétraitement des données
def preprocess_text(text):
    text = re.sub(r'@\w+|http\S+|#\w+', '', text)  # Supprimer mentions, hashtags, URL
    text = ''.join([ch for ch in text if ch not in string.punctuation])  # Supprimer ponctuation
    words = re.split('\W+', text)  # Tokenizer
    words = [word for word in words if word not in stopwords.words('english')]  # Supprimer stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word in english_words]  # Lemmatisation
    counts=count_vect.transform(words)
    tf=tf_idf.transform(counts)
    return tf

# Charger le modèle
modele = joblib.load('random_forest_model.pkl')
count_vect = joblib.load('count_vectorizer.pkl')
tf_idf = joblib.load('tfidf_tranformer.pkl')
# Définir le schéma de données d'entrée
class DonneesEntree(BaseModel):
    message: str

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html', prediction=None)

@app.route("/form")
def form():
    return render_template("formulaire.html")

# Endpoint pour afficher et traiter la prédiction
@app.route("/predire", methods=["GET", "POST"])
def predire():
    if not request.form or "message" not in request.form:
        return jsonify({"erreur": "Aucun texte fourni"}), 400

    try:
        # Récupération du texte depuis le formulaire
        text = request.form["message"]

        texte=preprocess_text(text)

        # Création d'un DataFrame avec le texte
        #donnees_df = pd.DataFrame([{"message": texte}])

        # Prédiction avec le modèle
        predictions = modele.predict(texte)

        # Transmission du résultat à `predic.html`
        return render_template('predict.html', predictions)#=predictions[0])

    except Exception as e:
        return jsonify({"erreur": "Erreur lors de la prédiction", "details": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
