# toola/generador_cvs_etiquetados.py

import os
import random
import csv
from fpdf import FPDF

# === Parámetros generales ===
N_CVS = 50  # CVs por perfil (ajusta si quieres más)
OUTPUT_FOLDER = "data/cvs_pdfs"
ETIQUETAS_PATH = "data/etiquetas.csv"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# === Definición de perfiles y características ===
perfiles = {
    "backend": {
        "habilidades": ["Python", "Java", "SQL", "C#", ".NET", "Docker", "Linux"],
        "descripcion": "Especialista en desarrollo backend, bases de datos y servidores.",
        "proyectos": [
            "API RESTful para gestión de usuarios",
            "Optimización de consultas SQL en sistemas bancarios",
            "Migración de aplicaciones monolíticas a microservicios"
        ]
    },
    "frontend": {
        "habilidades": ["HTML", "CSS", "JavaScript", "React", "Angular", "Figma"],
        "descripcion": "Experto en desarrollo de interfaces y experiencia de usuario.",
        "proyectos": [
            "Desarrollo de dashboard en React para monitoreo en tiempo real",
            "Implementación de diseño responsive para e-commerce",
            "Maquetación y prototipado en Figma"
        ]
    },
    "data": {
        "habilidades": ["Python", "SQL", "Machine Learning", "Pandas", "Power BI", "Data Science"],
        "descripcion": "Profesional en análisis de datos y machine learning.",
        "proyectos": [
            "Modelo de predicción de ventas usando machine learning",
            "Dashboards de visualización de KPIs en Power BI",
            "Automatización de reportes con Python y Pandas"
        ]
    },
    "infraestructura": {
        "habilidades": ["Linux", "Docker", "Redes", "Cloud Computing", "AWS", "Azure", "Servidores"],
        "descripcion": "Especialista en infraestructura, redes y cloud.",
        "proyectos": [
            "Despliegue de infraestructura como código en AWS",
            "Configuración de redes seguras y VPNs",
            "Monitorización de servidores Linux"
        ]
    }
}

nombres = ["José Pérez", "Ana Gómez", "Carlos Ruiz", "Sofía Torres", "Miguel Fernández", "Lucía Sánchez", "Daniela Díaz", "María López", "Andrés García"]
instituciones = ["Universidad Nacional", "Universidad de Ingeniería", "Instituto Tecnológico"]
empresas = ["TechCorp", "SoftSolutions", "DataWorks", "InfraSystems"]
niveles_idioma = ["Nativo", "Avanzado", "Intermedio"]

def normaliza_idioma(s):
    """Convierte a minúsculas, elimina acentos y espacios."""
    s = s.lower()
    reemplazos = (
        ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n")
    )
    for a, b in reemplazos:
        s = s.replace(a, b)
    s = s.replace(" ", "")
    return s

def generar_cv_texto(nombre, perfil):
    datos = perfiles[perfil]
    habilidades = random.sample(datos["habilidades"], k=min(4, len(datos["habilidades"])))
    proyecto = random.choice(datos["proyectos"])
    institucion = random.choice(instituciones)
    empresa = random.choice(empresas)
    # Idiomas normalizados, formato: ingles(nativo)
    idiomas_lista = ["Español", "Inglés", "Francés", "Alemán", "Portugués", "Italiano"]
    num_idiomas = random.randint(1, 3)
    idiomas_elegidos = random.sample(idiomas_lista, num_idiomas)
    idiomas_formateados = []
    for idioma in idiomas_elegidos:
        nivel = random.choice(niveles_idioma)
        idioma_norm = normaliza_idioma(idioma)
        nivel_norm = normaliza_idioma(nivel)
        idiomas_formateados.append(f"{idioma_norm}({nivel_norm})")
    correo = nombre.lower().replace(" ", ".") + "@email.com"

    texto = f"""Nombre completo: {nombre}
Correo electrónico: {correo}
Teléfono: +51 9{random.randint(10000000,99999999)}
Perfil profesional:
{datos['descripcion']}

Educación:
- Grado en Ingeniería de Sistemas, {institucion}, 2015 - 2020

Experiencia laboral:
- {perfil.capitalize()} en {empresa}, 2020 - 2023
  Descripción: Trabajo enfocado en {', '.join(habilidades)}.
- Pasantía en {empresa}, 2019 - 2020

Habilidades técnicas:
""" + "\n".join(f"- {h}" for h in habilidades) + f"""

Idiomas:
""" + "\n".join(f"- {idioma}" for idioma in idiomas_formateados) + f"""

Proyectos relevantes:
- {proyecto}

Referencias:
- Juan García, Jefe en {empresa}, juan.garcia@{empresa.lower()}.com
"""
    return texto

def guardar_pdf(nombre_archivo, texto):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for linea in texto.split('\n'):
        pdf.multi_cell(0, 10, linea)
    pdf.output(nombre_archivo)

# === Generación de CVs y etiquetas ===
etiquetas = []
contador = 1

for perfil in perfiles.keys():
    for _ in range(N_CVS):
        nombre = random.choice(nombres)
        texto = generar_cv_texto(nombre, perfil)
        archivo_pdf = f"cv_{contador}.pdf"
        ruta_archivo = os.path.join(OUTPUT_FOLDER, archivo_pdf)
        guardar_pdf(ruta_archivo, texto)
        etiquetas.append([archivo_pdf, perfil])
        contador += 1

# === Guardar archivo de etiquetas ===
with open(ETIQUETAS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["archivo", "perfil"])
    writer.writerows(etiquetas)

print(f"Generados {contador-1} CVs en PDF y archivo de etiquetas guardado en {ETIQUETAS_PATH}")