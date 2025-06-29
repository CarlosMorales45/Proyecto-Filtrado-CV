# Filtrado y Análisis Automatizado de CVs usando IA
**Proyecto de Inteligencia Artificial y Principios Técnicos**

---

## Autores

- Morales Chunga Carlos  
- Cardenas Iglesias Hugo  
- Mauricci Becerra Leandro  
- Meza Cordova Aryel  

---

## Descripción

Este proyecto implementa un pipeline automatizado para el análisis y filtrado inteligente de currículums:

- Procesamiento de Lenguaje Natural (PLN) para la extracción y normalización de texto de los CVs en PDF.
- Matching semántico entre los CVs y una descripción de vacante usando embeddings de oraciones.
- Clasificación supervisada para predecir el perfil técnico del candidato (backend, frontend, data, infraestructura).
- Scoring de palabras clave definidas por el usuario (por ejemplo: `python`, `docker`, `inglés(nativo)`, `.net`, etc.).

El resultado final es un **ranking de los CVs más relevantes** según afinidad semántica, coincidencias de keywords y perfil técnico predicho.

---

## Estructura del repositorio

```
data/
├── cvs_pdfs/        # CVs en PDF generados
└── etiquetas.csv    # Archivo con etiquetas (perfil) de cada CV

src/
├── classifier.py
├── embedding.py
├── main.py
└── utils.py

tools/
├── generador_cvs_etiquetados.py
└── entrenar_clasificador.py

ModeloIACvs.ipynb    # Notebook principal de demo
requirements.txt     # Dependencias del proyecto
README.md            # Este archivo
.gitignore
```

---

## Requisitos

- Python 3.8+

*(Se recomienda crear y activar un entorno virtual antes de instalar dependencias)*

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

---

## Instrucciones de uso

### 1. Generar CVs de prueba y etiquetas

Para trabajar con datos de ejemplo, ejecuta el script:

```bash
python tools/generador_cvs_etiquetados.py
```
Esto generará PDFs ficticios de CV en `data/cvs_pdfs/` y el archivo de etiquetas en `data/etiquetas.csv`.

---

### 2. Entrenar el clasificador

Con los CVs ya generados, entrena el modelo supervisado:

```bash
python tools/entrenar_clasificador.py
```
Esto creará los archivos `classifier.pkl` y `vectorizer.pkl` en la raíz del proyecto.

---

### 3. Ejecutar el análisis y ranking

Puedes ejecutar el pipeline completo y probar la demo desde el notebook:

- Abre y ejecuta **ModeloIACvs.ipynb** (recomendado para explorar los resultados paso a paso).
- El sistema pedirá las palabras clave relevantes y las buscará en cada CV.
- Calcula la afinidad semántica entre el texto del CV y la descripción de la vacante.
- Predice el perfil técnico de cada CV.
- Combina todos los puntajes y muestra los candidatos más relevantes, indicando también las palabras clave encontradas.

---

## Ejemplo de ejecución

```
Ingresa las palabras clave separadas por coma:
python, sql, docker, ingles(nativo)

Pega aquí la descripción de la vacante:
Buscamos ingeniero de software con experiencia en backend, dominio de python y bases de datos SQL, familiaridad con Docker y buen nivel de inglés.
```

CVs mejor rankeados por afinidad semántica y perfil técnico:

| CV         | Score Keywords | Score Semántico | Perfil Técnico Predicho | Puntaje Total |
|------------|:-------------:|:---------------:|:----------------------:|:-------------:|
| cv_2.pdf   |       4       |      0.599      |       backend          |    4.599      |
| cv_26.pdf  |       4       |      0.597      |       backend          |    4.597      |
| ...        |      ...      |       ...       |        ...             |     ...       |

Palabras clave encontradas por CV (Top 10):

- cv_2.pdf: python, sql, docker, ingles(nativo)
- cv_26.pdf: python, sql, docker, ingles(nativo)
- ...

---

## Consideraciones

- El pipeline es modular: puedes adaptar los scripts para usar tus propios CVs, etiquetas o mejorar el entrenamiento.
- **No subas archivos generados ni PDFs reales** a tu repositorio; usa únicamente los de prueba.

---

## Conclusiones

Este proyecto demuestra cómo aplicar técnicas modernas de IA para el procesamiento y filtrado eficiente de grandes volúmenes de currículums, automatizando tareas tradicionalmente manuales. El uso de matching semántico y scoring de palabras clave permite obtener un ranking mucho más preciso y relevante para los requerimientos reales de una vacante, agilizando los procesos de selección en escenarios académicos y empresariales.

---

## Créditos

Elaborado para el curso de **Inteligencia Artificial y Principios Técnicos**, UPAO.

---
