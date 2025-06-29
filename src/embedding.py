# src/embedding.py

from sentence_transformers import SentenceTransformer, util

def compute_semantic_scores(cv_texts, job_description):
    """
    Calcula la similitud semántica entre cada CV y la descripción de una vacante.

    Parámetros:
    - cv_texts: dict {nombre_archivo: texto_limpio}
        Diccionario con los textos limpios de los CVs (valores) y sus nombres de archivo (claves).
    - job_description: str
        Descripción de la vacante o puesto de trabajo (en texto plano).

    Retorna:
    - results: list of tuples (nombre_archivo, score)
        Lista de tuplas, donde cada tupla contiene el nombre del CV y el puntaje de similitud semántica,
        ordenados de mayor a menor score.

    Detalles:
    - Utiliza el modelo preentrenado 'distiluse-base-multilingual-cased-v1' de Sentence Transformers,
      el cual genera embeddings (representaciones numéricas) tanto para la descripción del puesto como para cada CV.
    - La similitud se calcula usando el coseno entre los embeddings del CV y la vacante.
    - Un score mayor indica mayor afinidad semántica entre el CV y la vacante.
    """

    # Carga el modelo preentrenado de Sentence Transformers (soporta español e inglés)
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # Calcula el embedding de la descripción de la vacante
    emb_job = model.encode(job_description)
    results = []
    # Para cada CV, calcula el embedding y la similitud con la vacante
    for pdf, text in cv_texts.items():
        emb_cv = model.encode(text)
        # Similaridad por coseno (valor entre -1 y 1)
        score = util.cos_sim(emb_cv, emb_job).item()
        results.append((pdf, score))
    # Ordena los resultados de mayor a menor afinidad semántica
    results.sort(key=lambda x: x[1], reverse=True)
    return results