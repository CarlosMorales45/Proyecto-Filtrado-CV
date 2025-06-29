# src/classifier.py

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def train_classifier(cv_texts, labels, model_path='classifier.pkl', vect_path='vectorizer.pkl'):
    # cv_texts: dict {filename: cleaned_text}
    # labels: list of labels corresponding to the files
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list(cv_texts.values()))
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)
    # Guarda el modelo y el vectorizador para uso futuro
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(vect_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_classifier(model_path='classifier.pkl', vect_path='vectorizer.pkl'):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return clf, vectorizer

def predict_profiles(cv_texts, clf, vectorizer):
    # cv_texts: dict {filename: cleaned_text}
    X = vectorizer.transform(list(cv_texts.values()))
    preds = clf.predict(X)
    return dict(zip(cv_texts.keys(), preds))
