import os
from utils import extract_and_clean_all_pdfs
from embedding import compute_semantic_scores
from classifier import load_classifier, predict_profiles

def main():
    """
    Script principal para analizar CVs.
    - Extrae y limpia los textos de los CVs desde PDFs.
    - Calcula la similitud semántica entre cada CV y la descripción de la vacante.
    - Predice el perfil técnico de cada CV usando un clasificador previamente entrenado.
    - Muestra el top 10 de CVs mejor rankeados.
    """

    pdf_folder = os.path.join('..', 'data', 'cvs_pdfs')

    print("Extrayendo y limpiando los CVs...")
    cv_texts = extract_and_clean_all_pdfs(pdf_folder)

    job_description = input("Pega aquí la descripción de la vacante:\n")

    print("\nCalculando afinidad semántica con la vacante...")
    semantic_scores = compute_semantic_scores(cv_texts, job_description)

    print("\nCargando clasificador de perfiles técnicos...")
    clf, vectorizer = load_classifier()
    profiles = predict_profiles(cv_texts, clf, vectorizer)

    print("\nResultados (Top 10):\n")
    for pdf, score in semantic_scores[:10]:
        profile = profiles.get(pdf, "Desconocido")
        print(f"{pdf:20} | Afinidad: {score:.3f} | Perfil predicho: {profile}")

if __name__ == "__main__":
    main()