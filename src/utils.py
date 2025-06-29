# src/utils.py

import os
import re
import unicodedata
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de todas las páginas de un archivo PDF.

    Parámetros:
    - pdf_path: str
        Ruta al archivo PDF.

    Retorna:
    - text: str
        Texto extraído concatenado de todas las páginas del PDF.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def remove_accents(text):
    """
    Elimina acentos y caracteres diacríticos del texto, dejando solo letras base.

    Parámetros:
    - text: str
        Texto de entrada.

    Retorna:
    - str
        Texto normalizado sin acentos.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def clean_text(text):
    """
    Limpia y normaliza el texto extraído de un CV:
    - Convierte a minúsculas.
    - Elimina saltos de línea, retornos y tabulaciones.
    - Elimina caracteres especiales excepto letras, números, algunos símbolos y acentos.
    - Elimina acentos.
    - Reemplaza múltiples espacios por uno solo.

    Parámetros:
    - text: str
        Texto de entrada.

    Retorna:
    - str
        Texto limpio y normalizado.
    """
    text = text.lower()
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[^a-z0-9\+\#\.\-\_\(\)\sáéíóúñü]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_accents(text)
    return text

def extract_and_clean_all_pdfs(pdf_folder):
    """
    Procesa todos los archivos PDF en una carpeta:
    - Extrae el texto de cada PDF.
    - Limpia y normaliza cada texto.

    Parámetros:
    - pdf_folder: str
        Ruta a la carpeta donde están los PDFs.

    Retorna:
    - dict {nombre_archivo: texto_limpio}
        Diccionario con el nombre del PDF como clave y su texto limpio como valor.
    """
    cv_texts = {}
    for file in os.listdir(pdf_folder):
        if file.endswith('.pdf'):
            path = os.path.join(pdf_folder, file)
            raw_text = extract_text_from_pdf(path)
            cleaned = clean_text(raw_text)
            cv_texts[file] = cleaned
    return cv_texts

def normalize_keyword(k):
    """
    Normaliza una palabra clave:
    - Convierte a minúsculas.
    - Elimina espacios y acentos.

    Parámetros:
    - k: str
        Palabra clave de entrada.

    Retorna:
    - str
        Palabra clave normalizada.
    """
    k = k.lower().strip()
    k = remove_accents(k)
    return k

def keywords_score(cv_texts, keywords):
    """
    Calcula el puntaje de coincidencia de palabras clave para cada CV.
    Para cada CV, cuenta cuántas de las palabras clave se encuentran presentes
    y devuelve también la lista de palabras coincidentes.

    La función maneja:
    - Palabras clave con formato idioma(nivel) (permite espacios dentro del paréntesis).
    - Palabras con símbolos (como .net, c++, c#).
    - Palabras normales (usa límites de palabra \b).

    Parámetros:
    - cv_texts: dict {nombre_archivo: texto_limpio}
        Textos de CVs procesados.
    - keywords: list[str]
        Lista de palabras clave normalizadas.

    Retorna:
    - dict {nombre_archivo: (puntaje, [coincidencias])}
        Para cada CV, devuelve el puntaje y la lista de palabras clave encontradas.
    """
    results = {}
    for pdf, text in cv_texts.items():
        score = 0
        matches = []
        for kw in keywords:
            # Detecta si la palabra clave es un idioma con nivel (ej: ingles(nativo))
            if "(" in kw and ")" in kw:
                escaped = re.escape(kw)
                # Permite espacios variables dentro de los paréntesis
                pattern = escaped.replace(r'\(', r'\s*\(').replace(r'\)', r'\s*\)')
            # Detecta palabras clave con símbolos (como .net, c++, c#)
            elif any(ch in kw for ch in ".+#"):
                pattern = re.escape(kw)
            # Palabra clave normal (usa límites de palabra)
            else:
                pattern = rf"\b{re.escape(kw)}\b"
            # Busca el patrón en el texto del CV
            if re.search(pattern, text):
                score += 1
                matches.append(kw)
        results[pdf] = (score, matches)
    return results