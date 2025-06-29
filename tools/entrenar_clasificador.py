# entrenar_clasificador.py

import pandas as pd
import sys
sys.path.append('./src')

from utils import extract_and_clean_all_pdfs
from classifier import train_classifier

# Rutas
pdf_folder = 'data/cvs_pdfs'
etiquetas_path = 'data/etiquetas.csv'

# Cargar etiquetas
df = pd.read_csv(etiquetas_path)

# Extraer y limpiar textos de los CVs
cv_texts = extract_and_clean_all_pdfs(pdf_folder)

# Filtrar solo los archivos que tengan etiqueta
cv_texts = {row['archivo']: cv_texts[row['archivo']]
            for idx, row in df.iterrows() if row['archivo'] in cv_texts}
labels = df.set_index('archivo').loc[list(cv_texts.keys()), 'perfil'].tolist()

print(f"Entrenando con {len(labels)} CVs etiquetados...")

# Entrenar y guardar modelo y vectorizador
train_classifier(cv_texts, labels)
print("¡Clasificador entrenado y guardado correctamente!")