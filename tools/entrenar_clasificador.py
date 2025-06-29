# entrenar_clasificador.py

import pandas as pd
import sys
sys.path.append('./src')  # Permite importar los módulos desde la carpeta src

from utils import extract_and_clean_all_pdfs
from classifier import train_classifier

# === Parámetros y rutas de entrada ===
pdf_folder = 'data/cvs_pdfs'        # Carpeta donde se encuentran los archivos PDF de los CVs
etiquetas_path = 'data/etiquetas.csv'  # Ruta del archivo CSV con las etiquetas (perfil) de cada CV

# === 1. Carga de etiquetas desde el archivo CSV ===
df = pd.read_csv(etiquetas_path)  # Lee el archivo de etiquetas (debe tener columnas 'archivo' y 'perfil')

# === 2. Extracción y limpieza de textos de los CVs ===
cv_texts = extract_and_clean_all_pdfs(pdf_folder)
# Solo se consideran los archivos que tengan etiqueta asociada en el CSV
cv_texts = {row['archivo']: cv_texts[row['archivo']]
            for idx, row in df.iterrows() if row['archivo'] in cv_texts}
labels = df.set_index('archivo').loc[list(cv_texts.keys()), 'perfil'].tolist()

print(f"Entrenando con {len(labels)} CVs etiquetados...")

# === 3. Entrenamiento y guardado del modelo y vectorizador ===
train_classifier(cv_texts, labels)
print("¡Clasificador entrenado y guardado correctamente!")

"""
Resumen del flujo:
------------------
1. Carga el archivo de etiquetas (cada CV tiene su perfil técnico real).
2. Extrae y normaliza los textos de los PDF de CVs.
3. Entrena el clasificador supervisado y el vectorizador TF-IDF.
4. Guarda ambos en disco para que el pipeline principal los use luego.

Este script solo se debe ejecutar cuando:
- Generas un nuevo set de CVs de ejemplo.
- Actualizas los perfiles o las etiquetas manualmente.
- Necesitas reentrenar el modelo con nuevos datos.

"""