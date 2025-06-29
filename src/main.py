# src/main.py

import os
from utils import extract_and_clean_all_pdfs, unzip_cvs
from embedding import compute_semantic_scores
from classifier import load_classifier, predict_profiles

def main():
    zip_path = os.path.join('..', 'data', 'cvs_ingenieria_sistemas.zip')
    pdf_folder = os.path.join('..', 'data', 'cvs_pdfs')
    
    # Si la carpeta de PDFs no existe o está vacía, descomprime el ZIP
    if not os.path.exists(pdf_folder) or not os.listdir(pdf_folder):
        print("Extrayendo archivos PDF del ZIP...")
        unzip_cvs(zip_path, pdf_folder)

    print("Extrayendo y limpiando los CVs...")
    cv_texts = extract_and_clean_all_pdfs(pdf_folder)

    job_description = input("Pega aquí la descripción de la vacante:\n")

    print("\nCalculando afinidad semántica con la vacante...")
    semantic_scores = compute_semantic_scores(cv_texts, job_description)

    print("\nCargando clasificador de perfiles técnicos...")
    clf, vectorizer = load_classifier()  # Asegúrate de tener entrenado el modelo antes
    profiles = predict_profiles(cv_texts, clf, vectorizer)

    print("\nResultados (Top 10):\n")
    for pdf, score in semantic_scores[:10]:
        profile = profiles.get(pdf, "Desconocido")
        print(f"{pdf:20} | Afinidad: {score:.3f} | Perfil predicho: {profile}")

if __name__ == "__main__":
    main()
