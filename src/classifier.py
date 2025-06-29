# src/classifier.py

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def train_classifier(cv_texts, labels, model_path='classifier.pkl', vect_path='vectorizer.pkl'):
    """
    Entrena un clasificador supervisado para predecir el perfil técnico de los CVs.

    Parámetros:
    - cv_texts: dict {nombre_archivo: texto_limpio}
        Diccionario donde las claves son los nombres de los archivos PDF de los CVs
        y los valores son los textos procesados/limpios de cada CV.
    - labels: list
        Lista de etiquetas (perfiles técnicos) correspondientes a cada CV.
    - model_path: str
        Ruta donde se guardará el modelo entrenado (formato .pkl).
    - vect_path: str
        Ruta donde se guardará el vectorizador TF-IDF entrenado (formato .pkl).

    El modelo y el vectorizador quedan guardados en disco para uso futuro.
    """
    # Vectoriza los textos usando TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list(cv_texts.values()))
    # Entrena un clasificador de regresión logística
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)
    # Guarda el modelo entrenado y el vectorizador en archivos .pkl
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(vect_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_classifier(model_path='classifier.pkl', vect_path='vectorizer.pkl'):
    """
    Carga desde disco el modelo de clasificación y el vectorizador TF-IDF previamente entrenados.

    Parámetros:
    - model_path: str
        Ruta al archivo .pkl del clasificador.
    - vect_path: str
        Ruta al archivo .pkl del vectorizador.

    Retorna:
    - clf: Clasificador de sklearn entrenado (por defecto, Regresión Logística).
    - vectorizer: Objeto TfidfVectorizer entrenado.
    """
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return clf, vectorizer

def predict_profiles(cv_texts, clf, vectorizer):
    """
    Predice el perfil técnico de cada CV utilizando el modelo y el vectorizador entrenados.

    Parámetros:
    - cv_texts: dict {nombre_archivo: texto_limpio}
        Diccionario con los textos limpios de los CVs.
    - clf: Clasificador entrenado.
    - vectorizer: Vectorizador TF-IDF entrenado.

    Retorna:
    - Dict con pares {nombre_archivo: perfil_predicho}.
    """
    # Transforma los textos a vectores numéricos usando el vectorizador entrenado
    X = vectorizer.transform(list(cv_texts.values()))
    # Predice los perfiles técnicos usando el modelo entrenado
    preds = clf.predict(X)
    # Devuelve un diccionario mapeando cada archivo PDF al perfil predicho
    return dict(zip(cv_texts.keys(), preds))
