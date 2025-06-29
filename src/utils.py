# src/utils.py

import os
import re
import unicodedata
import pdfplumber
import zipfile

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[^a-z0-9\+\#\.\-\_\(\)\sáéíóúñü]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_accents(text)
    return text

def extract_and_clean_all_pdfs(pdf_folder):
    cv_texts = {}
    for file in os.listdir(pdf_folder):
        if file.endswith('.pdf'):
            path = os.path.join(pdf_folder, file)
            raw_text = extract_text_from_pdf(path)
            cleaned = clean_text(raw_text)
            cv_texts[file] = cleaned
    return cv_texts

def normalize_keyword(k):
    k = k.lower().strip()
    k = remove_accents(k)
    return k

def keywords_score(cv_texts, keywords):
    results = {}
    for pdf, text in cv_texts.items():
        score = 0
        matches = []
        for kw in keywords:
            if "(" in kw and ")" in kw:  # idioma(nivel)
                escaped = re.escape(kw)
                pattern = escaped.replace(r'\(', r'\s*\(').replace(r'\)', r'\s*\)')
            elif any(ch in kw for ch in ".+#"):  # .net, c++, c#, etc.
                pattern = re.escape(kw)
            else:
                pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text):
                score += 1
                matches.append(kw)
        results[pdf] = (score, matches)
    return results