import streamlit as st
import os
import pandas as pd
import sys
import subprocess

sys.path.append('./src')
from utils import extract_and_clean_all_pdfs, normalize_keyword, keywords_score
from classifier import load_classifier, predict_profiles
from embedding import compute_semantic_scores

# Configuración de la página
st.set_page_config(page_title="Filtrado y Análisis Automatizado de CVs", layout="centered")
st.title("🧠 Filtrado y Análisis Automatizado de CVs usando IA")

# 1. Verifica/genera automáticamente los datos si no existen
pdf_folder = 'data/cvs_pdfs'
etiquetas_path = 'data/etiquetas.csv'

def generar_datos():
    st.warning("Generando datos de prueba automáticamente...")
    # Ejecuta scripts de generación y entrenamiento
    subprocess.run(["python", "tools/generador_cvs_etiquetados.py"])
    subprocess.run(["python", "tools/entrenar_clasificador.py"])
    st.success("¡CVs y modelo generados correctamente! Recarga la página para continuar.")

# Si no existe la carpeta de CVs o está vacía, genera los datos
if not os.path.exists(pdf_folder) or not os.listdir(pdf_folder):
    generar_datos()
    st.stop()
if not os.path.exists(etiquetas_path):
    generar_datos()
    st.stop()

# 2. Ingreso de palabras clave
st.header("1️⃣ Palabras clave de filtrado")
keywords_input = st.text_input(
    "Ingresa las palabras clave separadas por coma (ejemplo: python, docker, .net, ingles(nativo), sql)",
    value="python, docker, .net, ingles(nativo), sql"
)
keywords = [normalize_keyword(k) for k in keywords_input.split(",")]
st.write("Palabras clave normalizadas:", keywords)

# 3. Ingreso de descripción de vacante
st.header("2️⃣ Descripción de la vacante")
job_description = st.text_area(
    "Pega aquí la descripción de la vacante:",
    value="Buscamos ingeniero backend con experiencia en Python, .NET, Docker, Linux y nivel de inglés nativo."
)

# 4. ¿Cuántos candidatos deseas mostrar?
n_top = st.number_input("¿Cuántas plazas deseas mostrar?", min_value=1, max_value=30, value=5)

# 5. Procesamiento (al pulsar botón)
if st.button("Analizar CVs"):
    # Extracción y limpieza
    st.info("Extrayendo y limpiando CVs...")
    cv_texts = extract_and_clean_all_pdfs(pdf_folder)

    # Score por keywords
    kw_results = keywords_score(cv_texts, keywords)
    st.success("Puntaje por palabras clave calculado.")

    # Matching semántico
    st.info("Calculando afinidad semántica (esto puede tardar unos segundos)...")
    semantic_scores = compute_semantic_scores(cv_texts, job_description)
    semantic_dict = dict(semantic_scores)

    # Clasificación de perfiles técnicos
    st.info("Cargando clasificador de perfiles técnicos...")
    clf, vectorizer = load_classifier()
    profiles = predict_profiles(cv_texts, clf, vectorizer)

    # Tabla combinada y puntaje total
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

    # Mostrar detalle de coincidencias
    st.header("🔍 Detalle de coincidencias en el top")
    for i, row in df.head(n_top).iterrows():
        st.subheader(f"{i+1}. {row['CV']}")
        st.markdown(f"- **Palabras clave encontradas:** {row['Keywords Encontradas']}")
        st.markdown(f"- **Perfil técnico predicho:** `{row['Perfil Técnico Predicho']}`")
        st.markdown(f"- **Score total:** `{row['Puntaje Total']}`")
        st.markdown("---")

else:
    st.info("Completa los campos y pulsa 'Analizar CVs'.")