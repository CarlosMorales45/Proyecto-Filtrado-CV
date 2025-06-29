import streamlit as st
import os
import pandas as pd
import sys
import zipfile
import shutil
import subprocess
import time

sys.path.append('./src')
from utils import extract_and_clean_all_pdfs, normalize_keyword, keywords_score
from classifier import load_classifier, predict_profiles
from embedding import compute_semantic_scores

st.set_page_config(page_title="Filtrado y Análisis Automatizado de CVs", layout="centered")
st.title("🧠 Filtrado y Análisis Automatizado de CVs usando IA")

pdf_folder = 'data/cvs_pdfs'
etiquetas_path = 'data/etiquetas.csv'

def limpiar_carpeta(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# === 1. Selección de fuente de datos ===
st.header("📁 ¿Cómo deseas cargar los CVs?")
modo = st.radio(
    "Selecciona una opción:",
    [
        "Subir un ZIP con mis propios CVs PDF",
        "Generar CVs de prueba automáticamente"
    ]
)

if modo == "Subir un ZIP con mis propios CVs PDF":
    zip_file = st.file_uploader("Sube aquí tu archivo .zip con los CVs en PDF", type=["zip"])
    if zip_file is not None:
        limpiar_carpeta(pdf_folder)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(pdf_folder)
        st.success("¡CVs extraídos correctamente! Continúa con el análisis.")
        etiquetas_csv = st.file_uploader("Sube el archivo etiquetas.csv (archivo de perfiles)", type=["csv"])
        if etiquetas_csv is not None:
            with open(etiquetas_path, "wb") as f:
                f.write(etiquetas_csv.read())
            st.success("Archivo de etiquetas cargado correctamente.")
        # Recarga si ambos archivos existen (flujo más cómodo)
        if os.path.exists(pdf_folder) and os.listdir(pdf_folder) and os.path.exists(etiquetas_path):
            st.info("CVs y etiquetas cargados correctamente. ¡Ahora puedes analizarlos!")
            time.sleep(0.5)
            st.experimental_rerun()
else:
    if st.button("Generar CVs y etiquetas de prueba"):
        subprocess.run(["python", "tools/generador_cvs_etiquetados.py"])
        subprocess.run(["python", "tools/entrenar_clasificador.py"])
        st.success("CVs y modelo de prueba generados correctamente.")
        time.sleep(0.5)
        st.experimental_rerun()

# === Verifica si hay datos listos para procesar ===
datos_listos = os.path.exists(pdf_folder) and os.listdir(pdf_folder) and os.path.exists(etiquetas_path)

if datos_listos:
    # === 2. Palabras clave ===
    st.header("1️⃣ Palabras clave de filtrado")
    keywords_input = st.text_input(
        "Ingresa las palabras clave separadas por coma (ejemplo: python, docker, .net, ingles(nativo), sql)",
        value="python, docker, .net, ingles(nativo), sql"
    )
    keywords = [normalize_keyword(k) for k in keywords_input.split(",")]
    st.write("Palabras clave normalizadas:", keywords)

    # === 3. Descripción de vacante ===
    st.header("2️⃣ Descripción de la vacante")
    job_description = st.text_area(
        "Pega aquí la descripción de la vacante:",
        value="Buscamos ingeniero backend con experiencia en Python, .NET, Docker, Linux y nivel de inglés nativo."
    )

    # === 4. Número de candidatos ===
    n_top = st.number_input("¿Cuántas plazas deseas mostrar?", min_value=1, max_value=30, value=5)

    # === 5. Procesamiento ===
    if st.button("Analizar CVs"):
        st.info("Extrayendo y limpiando CVs...")
        cv_texts = extract_and_clean_all_pdfs(pdf_folder)

        kw_results = keywords_score(cv_texts, keywords)
        st.success("Puntaje por palabras clave calculado.")

        st.info("Calculando afinidad semántica (esto puede tardar unos segundos)...")
        semantic_scores = compute_semantic_scores(cv_texts, job_description)
        semantic_dict = dict(semantic_scores)

        st.info("Cargando clasificador de perfiles técnicos...")
        clf, vectorizer = load_classifier()
        profiles = predict_profiles(cv_texts, clf, vectorizer)

        # Tabla y visualización
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
else:
    st.warning("Carga o genera los CVs (y etiquetas) antes de analizar.")