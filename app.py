import streamlit as st
import os
import pandas as pd
import sys

sys.path.append('./src')
from utils import extract_and_clean_all_pdfs, normalize_keyword, keywords_score
from classifier import load_classifier, predict_profiles
from embedding import compute_semantic_scores

st.set_page_config(page_title="Filtrado y Análisis Automatizado de CVs", layout="centered")
st.title("🧠 Filtrado y Análisis Automatizado de CVs usando IA")

# Carpetas/archivos de datos ya subidos al repo
pdf_folder = 'data/cvs_pdfs'
etiquetas_path = 'data/etiquetas.csv'
model_path = 'classifier.pkl'
vect_path = 'vectorizer.pkl'

# --- Verifica existencia de datos y modelo ---
if not (os.path.exists(pdf_folder) and os.listdir(pdf_folder)):
    st.error("❌ No se encontraron CVs en data/cvs_pdfs. Sube los archivos al repositorio.")
    st.stop()
if not os.path.exists(etiquetas_path):
    st.error("❌ No se encontró el archivo de etiquetas en data/etiquetas.csv. Sube el archivo al repositorio.")
    st.stop()
if not (os.path.exists(model_path) and os.path.exists(vect_path)):
    st.error("❌ No se encontraron los archivos del modelo (`classifier.pkl`, `vectorizer.pkl`). Sube ambos al repositorio.")
    st.stop()

# --- Paso 1: Palabras clave ---
st.header("1️⃣ Palabras clave de filtrado")
keywords_input = st.text_input(
    "Ingresa las palabras clave separadas por coma (ejemplo: python, docker, .net, ingles(nativo), sql)",
    value="python, docker, .net, ingles(nativo), sql"
)
keywords = [normalize_keyword(k) for k in keywords_input.split(",")]
st.write("Palabras clave normalizadas:", keywords)

# --- Paso 2: Descripción de la vacante ---
st.header("2️⃣ Descripción de la vacante")
job_description = st.text_area(
    "Pega aquí la descripción de la vacante:",
    value="Buscamos ingeniero backend con experiencia en Python, .NET, Docker, Linux y nivel de inglés nativo."
)

# --- Paso 3: Número de candidatos ---
n_top = st.number_input("¿Cuántas plazas deseas mostrar?", min_value=1, max_value=30, value=5)

# --- Paso 4: Análisis ---
if st.button("Analizar CVs"):
    st.info("Extrayendo y limpiando CVs...")
    cv_texts = extract_and_clean_all_pdfs(pdf_folder)

    kw_results = keywords_score(cv_texts, keywords)
    st.success("Puntaje por palabras clave calculado.")

    st.info("Calculando afinidad semántica (esto puede tardar unos segundos)...")
    semantic_scores = compute_semantic_scores(cv_texts, job_description)
    semantic_dict = dict(semantic_scores)

    st.info("Cargando clasificador de perfiles técnicos...")
    clf, vectorizer = load_classifier(model_path, vect_path)
    profiles = predict_profiles(cv_texts, clf, vectorizer)

    tabla = []
    for pdf in cv_texts.keys():
        kw_score, found_keywords = kw_results.get(pdf, (0, []))
        sem_score = semantic_dict.get(pdf, 0)
        perfil = profiles.get(pdf, "Desconocido")
        puntaje_total = kw_score + sem_score
        tabla.append({
            "CV": pdf,
            "Score Keywords": kw_score,
            "Keywords Encontradas": ", ".join(found_keywords),
            "Score Semántico": round(sem_score, 3),
            "Perfil Técnico Predicho": perfil,
            "Puntaje Total": round(puntaje_total, 3)
        })
    df = pd.DataFrame(tabla)
    df = df.sort_values("Puntaje Total", ascending=False).reset_index(drop=True)

    st.header("🏆 Ranking de CVs")
    st.dataframe(df.head(n_top), use_container_width=True)

    st.header("🔍 Detalle de coincidencias en el top")
    for i, row in df.head(n_top).iterrows():
        st.subheader(f"{i+1}. {row['CV']}")
        st.markdown(f"- **Palabras clave encontradas:** {row['Keywords Encontradas']}")
        st.markdown(f"- **Perfil técnico predicho:** `{row['Perfil Técnico Predicho']}`")
        st.markdown(f"- **Score total:** `{row['Puntaje Total']}`")
        st.markdown("---")
else:
    st.info("Completa los campos y pulsa 'Analizar CVs'.")