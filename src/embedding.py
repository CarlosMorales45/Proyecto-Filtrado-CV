# src/embedding.py

from sentence_transformers import SentenceTransformer, util

def compute_semantic_scores(cv_texts, job_description):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    emb_job = model.encode(job_description)
    results = []
    for pdf, text in cv_texts.items():
        emb_cv = model.encode(text)
        score = util.cos_sim(emb_cv, emb_job).item()
        results.append((pdf, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
