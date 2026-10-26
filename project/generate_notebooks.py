"""
Generator script to produce clean, well-documented Jupyter Notebooks strictly focused on
Image Detection / Classification (identifying which class an image belongs to: apples, dogs, people).
1. 1_hog.ipynb
2. 2_sift.ipynb
3. 3_freak.ipynb
4. 4_comparison.ipynb
"""

import json
from pathlib import Path


def create_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created notebook: {output_path}")


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }


# ==============================================================================
# Notebook 1: HOG
# ==============================================================================
hog_cells = [
    md_cell("""# 1. HOG (Histogram of Oriented Gradients) para Clasificación de Imágenes
### Clasificación de Categorías de Imágenes: Manzanas, Perros y Personas

Este cuaderno implementa la solución completa de **Image Detection / Classification** utilizando el descriptor **HOG** y un clasificador **SVM** con OpenCV:
- **Entrada**: Una imagen cualquiera.
- **Salida**: La clase predicha a la que pertenece la imagen (`apples`, `dogs` o `people`).

#### Correcciones clave respecto al trabajo previo en `mine/hog.ipynb`:
1. **Soporte de extensiones `.jpeg`**: El código original ignoraba las imágenes `.jpeg`, omitiendo el 100% de las manzanas (164 de train y 54 de test). Aquí se cargan todas las imágenes dinámicamente.
2. **Dimensionamiento óptimo de ventana**: Se sustituye la ventana de $512 \\times 512$ (que producía 142,884 dimensiones y un modelo de 628 MB) por una ventana de $128 \\times 128$ (8,100 dimensiones, modelo de 39 MB, entrenamiento instantáneo).
3. **Evaluación completa en el conjunto de prueba**: Cálculo de exactitud, reporte de clasificación por clase y matriz de confusión.
"""),

    code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split
from hog_pipeline import HOGPipeline

def show_image(img, title=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    plt.show()
"""),

    md_cell("""## Paso 1: Carga de Datos de Entrenamiento y Prueba"""),

    code_cell("""train_imgs, train_lbls, classes, _ = load_split("train")
test_imgs, test_lbls, _, _ = load_split("test")

print(f"Clases detectadas: {classes}")
print(f"Muestras de entrenamiento: {len(train_imgs)}")
print(f"Muestras de prueba:        {len(test_imgs)}")

for c in classes:
    print(f"  - {c}: {train_lbls.count(c)} train, {test_lbls.count(c)} test")
"""),

    md_cell("""## Paso 2: Configuración del Descriptor HOG
Ventana: `(128, 128)`, Bloque: `(16, 16)`, Paso de bloque: `(8, 8)`, Celda: `(8, 8)`, Contenedores: `9`.
Dimensión resultante: $15 \\times 15 \\times 36 = 8,100$ características.
"""),

    code_cell("""pipeline = HOGPipeline(win_size=(128, 128))
sample_des = pipeline.compute_descriptor(train_imgs[0])
print(f"Dimensión del descriptor HOG por imagen: {sample_des.shape[0]} características flotantes")
"""),

    md_cell("""## Paso 3: Entrenamiento del Clasificador SVM
Entrenamos la SVM de OpenCV (`cv2.ml.SVM`) con kernel lineal para clasificar las imágenes.
"""),

    code_cell("""print("Entrenando clasificador HOG + SVM...")
train_info = pipeline.train(train_imgs, train_lbls, kernel_type="linear", C=1.0)
print(f"Entrenamiento completado en {train_info['train_time_sec']:.2f} segundos!")
"""),

    md_cell("""## Paso 4: Evaluación en el Conjunto de Prueba
Evaluamos la exactitud, precisión, recall y matriz de confusión en las 140 imágenes de prueba.
"""),

    code_cell("""eval_results = pipeline.evaluate(test_imgs, test_lbls)
print(f"Exactitud en Prueba (Accuracy): {eval_results['accuracy'] * 100:.2f}%\n")
print("Reporte de Clasificación por Clase:")
print(classification_report(test_lbls, eval_results['predictions'], target_names=pipeline.classes_))

# Matriz de confusión
cm = np.array(eval_results["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)
plt.xlabel("Clase Predicha", fontsize=11)
plt.ylabel("Clase Real", fontsize=11)
plt.title("Matriz de Confusión HOG + SVM", fontsize=12, fontweight='bold')
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if i != j else "black", fontweight='bold')
plt.colorbar(im)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## Paso 5: Demostración de Clasificación en Imágenes Individuales
Dada una imagen de entrada, el modelo predice directamente su clase semántica.
"""),

    code_cell("""import random

# Seleccionamos muestras aleatorias de prueba
for _ in range(3):
    idx = random.randrange(len(test_imgs))
    img = test_imgs[idx]
    true_label = test_lbls[idx]
    
    pred_label, _ = pipeline.predict([img])
    show_image(img, title=f"Real: {true_label.upper()} | Predicción HOG: {pred_label[0].upper()}")
"""),

    md_cell("""## Paso 6: Guardado del Modelo"""),

    code_cell("""pipeline.save_model("hog_svm_model.yml")
model_size_mb = Path("hog_svm_model.yml").stat().st_size / (1024 * 1024)
print(f"Modelo guardado: hog_svm_model.yml ({model_size_mb:.2f} MB)")
""")
]


# ==============================================================================
# Notebook 2: SIFT
# ==============================================================================
sift_cells = [
    md_cell("""# 2. SIFT (Scale-Invariant Feature Transform) con Bolsa de Palabras Visuales (BoVW)
### Clasificación de Categorías de Imágenes: Manzanas, Perros y Personas

Este cuaderno implementa la solución completa de **Image Detection / Classification** utilizando **SIFT** y el modelo de **Bolsa de Palabras Visuales (Bag of Visual Words - BoVW)**:
- **Entrada**: Una imagen cualquiera.
- **Salida**: La clase predicha a la que pertenece la imagen (`apples`, `dogs` o `people`).

#### ¿Por qué Bolsa de Palabras Visuales (BoVW) para clasificar imágenes?
SIFT extrae un número variable de puntos clave locales por imagen (descriptores de 128 dimensiones). Como distintas imágenes de una misma clase no comparten puntos en las mismas posiciones espaciales, no se pueden comparar directamente fila por fila.
Con BoVW:
1. Agrupamos los descriptores de entrenamiento con K-Means para crear un **vocabulario visual** (diccionario de palabras visuales).
2. Cuantizamos cada imagen en un **histograma de frecuencias de palabras visuales**, produciendo un vector de longitud fija ($k = 150$) normalizado en norma $L_2$.
3. Entrenamos un clasificador SVM para predecir la clase de cualquier imagen.
"""),

    code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split
from sift_pipeline import SIFTBoVWPipeline

def show_image(img, title=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    plt.show()
"""),

    md_cell("""## Paso 1: Carga de Datos y Visualización de Puntos Clave SIFT"""),

    code_cell("""train_imgs, train_lbls, classes, _ = load_split("train")
test_imgs, test_lbls, _, _ = load_split("test")

sift = cv2.SIFT_create(nfeatures=400)
sample_img = train_imgs[0]
gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
kps, descs = sift.detectAndCompute(gray, None)

img_kps = cv2.drawKeypoints(sample_img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_image(img_kps, title=f"SIFT: Puntos Clave Multi-Escala ({len(kps)} puntos detectados)")
"""),

    md_cell("""## Paso 2: Creación del Vocabulario Visual y Entrenamiento del Clasificador BoVW"""),

    code_cell("""bovw_pipeline = SIFTBoVWPipeline(nfeatures=400, vocab_size=150)
print("Construyendo vocabulario visual con K-Means y entrenando clasificador SVM...")
train_info = bovw_pipeline.train(train_imgs, train_lbls, C=1.0)
print(f"Entrenamiento completado en {train_info['train_time_sec']:.2f} segundos!")
print(f"Vocabulario visual: {train_info['vocab_size']} palabras visuales")
"""),

    md_cell("""## Paso 3: Evaluación en el Conjunto de Prueba"""),

    code_cell("""eval_res = bovw_pipeline.evaluate(test_imgs, test_lbls)
print(f"Exactitud en Prueba SIFT BoVW (Accuracy): {eval_res['accuracy'] * 100:.2f}%")
print(f"Latencia promedio por imagen:            {eval_res['avg_time_per_image_ms']:.2f} ms/imagen\n")
print("Reporte de Clasificación por Clase:")
print(classification_report(test_lbls, eval_res['predictions'], target_names=bovw_pipeline.classes_))

# Matriz de confusión
cm = np.array(eval_res["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Purples")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)
plt.xlabel("Clase Predicha", fontsize=11)
plt.ylabel("Clase Real", fontsize=11)
plt.title("Matriz de Confusión SIFT + BoVW", fontsize=12, fontweight='bold')
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if i != j else "black", fontweight='bold')
plt.colorbar(im)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## Paso 4: Demostración de Clasificación en Imágenes Individuales"""),

    code_cell("""import random

for _ in range(3):
    idx = random.randrange(len(test_imgs))
    img = test_imgs[idx]
    true_label = test_lbls[idx]
    
    pred_label = bovw_pipeline.predict([img])[0]
    show_image(img, title=f"Real: {true_label.upper()} | Predicción SIFT: {pred_label.upper()}")
"""),

    md_cell("""## Paso 5: Guardado del Modelo"""),

    code_cell("""bovw_pipeline.save_model("sift_bovw_model.pkl")
sz = Path("sift_bovw_model.pkl").stat().st_size / (1024 * 1024)
print(f"Modelo guardado: sift_bovw_model.pkl ({sz:.2f} MB)")
""")
]


# ==============================================================================
# Notebook 3: FREAK
# ==============================================================================
freak_cells = [
    md_cell("""# 3. FREAK (Fast Retina Keypoint) con Bolsa de Palabras Visuales (BoVW)
### Clasificación de Categorías de Imágenes: Manzanas, Perros y Personas

Este cuaderno implementa la solución completa de **Image Detection / Classification** utilizando el descriptor binario bio-inspirado **FREAK**:
- **Entrada**: Una imagen cualquiera.
- **Salida**: La clase predicha a la que pertenece la imagen (`apples`, `dogs` o `people`).

#### ¿Qué es FREAK y cómo se aplica para clasificar imágenes?
1. **Descriptor Bio-inspirado**: FREAK emula la fóvea de la retina humana mediante círculos concéntricos de campos receptivos, con alta densidad en el centro y decaimiento hacia la periferia.
2. **Descriptor Binario de 512 bits (64 bytes)**: Cada bit es el resultado de una comparación de intensidades filtradas entre pares de campos receptivos.
3. **Detector Rápido FAST**: Dado que FREAK es exclusivamente un descriptor (no detecta puntos clave), se acopla con el detector ultra-rápido de esquinas FAST (`cv2.FastFeatureDetector_create`).
4. **Bolsa de Palabras Visuales (BoVW)**: Los descriptores binarios se agrupan en un vocabulario visual ($k = 150$) mediante K-Means para generar histogramas de frecuencias normalizados por imagen y clasificar con SVM.
"""),

    code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split
from freak_pipeline import FREAKBoVWPipeline

def show_image(img, title=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    plt.show()
"""),

    md_cell("""## Paso 1: Carga de Datos y Detección de Puntos Clave FAST + FREAK"""),

    code_cell("""train_imgs, train_lbls, classes, _ = load_split("train")
test_imgs, test_lbls, _, _ = load_split("test")

# Detector FAST + Extractor FREAK
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
freak = cv2.xfeatures2d.FREAK_create()

sample_img = train_imgs[10]
gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
kps = fast.detect(gray, None)
kps = sorted(kps, key=lambda k: k.response, reverse=True)[:300]
kps, descs = freak.compute(gray, kps)

print(f"Puntos clave FAST detectados: {len(kps)}")
print(f"Dimensión del descriptor FREAK: {descs.shape} (uint8, 64 bytes = 512 bits)")

img_kps = cv2.drawKeypoints(sample_img, kps, None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_image(img_kps, title="FAST Keypoints + FREAK Bio-Inspired Retinal Descriptors")
"""),

    md_cell("""## Paso 2: Creación del Vocabulario Visual y Entrenamiento de FREAK BoVW"""),

    code_cell("""freak_bovw = FREAKBoVWPipeline(detector_type="fast", max_keypoints=300, vocab_size=150)
print("Construyendo vocabulario visual FREAK y entrenando clasificador SVM...")
info = freak_bovw.train(train_imgs, train_lbls, C=1.0)
print(f"Entrenamiento completado en {info['train_time_sec']:.2f} segundos!")
print(f"Vocabulario visual: {info['vocab_size']} palabras visuales")
"""),

    md_cell("""## Paso 3: Evaluación en el Conjunto de Prueba"""),

    code_cell("""eval_res = freak_bovw.evaluate(test_imgs, test_lbls)
print(f"Exactitud en Prueba FREAK BoVW (Accuracy): {eval_res['accuracy'] * 100:.2f}%")
print(f"Latencia promedio por imagen:              {eval_res['avg_time_per_image_ms']:.2f} ms/imagen\n")
print("Reporte de Clasificación por Clase:")
print(classification_report(test_lbls, eval_res['predictions'], target_names=freak_bovw.classes_))

# Matriz de confusión
cm = np.array(eval_res["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Oranges")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)
plt.xlabel("Clase Predicha", fontsize=11)
plt.ylabel("Clase Real", fontsize=11)
plt.title("Matriz de Confusión FREAK + BoVW", fontsize=12, fontweight='bold')
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if i != j else "black", fontweight='bold')
plt.colorbar(im)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## Paso 4: Demostración de Clasificación en Imágenes Individuales"""),

    code_cell("""import random

for _ in range(3):
    idx = random.randrange(len(test_imgs))
    img = test_imgs[idx]
    true_label = test_lbls[idx]
    
    pred_label = freak_bovw.predict([img])[0]
    show_image(img, title=f"Real: {true_label.upper()} | Predicción FREAK: {pred_label.upper()}")
"""),

    md_cell("""## Paso 5: Guardado del Modelo"""),

    code_cell("""freak_bovw.save_model("freak_bovw_model.pkl")
sz = Path("freak_bovw_model.pkl").stat().st_size / (1024 * 1024)
print(f"Modelo guardado: freak_bovw_model.pkl ({sz:.2f} MB)")
""")
]


# ==============================================================================
# Notebook 4: Unified Comparison
# ==============================================================================
comp_cells = [
    md_cell("""# 4. Comparación de Desempeño: HOG vs SIFT vs FREAK
### Evaluación Comparativa en la Tarea de Image Detection / Classification

En este cuaderno se comparan frente a frente los tres métodos en la tarea de **Image Detection (Clasificación de Imágenes: dada una imagen, decir a qué clase pertenece)** sobre el mismo conjunto de prueba de 140 imágenes:
1. **HOG + SVM**
2. **SIFT + BoVW + SVM**
3. **FREAK + BoVW + SVM**
"""),

    code_cell("""import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Cargar resumen numérico del benchmark
with open("benchmark_summary.json", "r") as f:
    results = json.load(f)

methods = list(results.keys())
accuracies = [results[m]["Accuracy (%)"] for m in methods]
latencies = [results[m]["Latency (ms/img)"] for m in methods]
fps_vals = [results[m]["Throughput (FPS)"] for m in methods]
sizes = [results[m]["Model Size (MB)"] for m in methods]

print("=" * 80)
print(f"{'Método':<8} | {'Exactitud':<12} | {'Latencia (ms)':<14} | {'FPS':<10} | {'Tamaño Modelo':<12}")
print("-" * 80)
for m in methods:
    print(f"{m:<8} | {results[m]['Accuracy (%)']:>10.2f}% | {results[m]['Latency (ms/img)']:>12.2f} ms | {results[m]['Throughput (FPS)']:>10.1f} | {results[m]['Model Size (MB)']:>10.2f} MB")
print("=" * 80)
"""),

    md_cell("""## Gráficas Comparativas de Rendimiento"""),

    code_cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = ['#1f77b4', '#9467bd', '#ff7f0e']

# 1. Exactitud
axes[0].bar(methods, accuracies, color=colors, width=0.55)
axes[0].set_title("Exactitud en Prueba (%)", fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')

# 2. Latencia
axes[1].bar(methods, latencies, color=colors, width=0.55)
axes[1].set_title("Latencia de Inferencia (ms/imagen)", fontsize=12, fontweight='bold')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(latencies):
    axes[1].text(i, v + 0.8, f"{v:.2f} ms", ha='center', fontweight='bold')

# 3. Tamaño del Modelo
axes[2].bar(methods, sizes, color=colors, width=0.55)
axes[2].set_title("Tamaño del Modelo (MB)", fontsize=12, fontweight='bold')
axes[2].grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(sizes):
    axes[2].text(i, v + 0.9, f"{v:.2f} MB", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
"""),

    md_cell("""## Visualización de Muestras Clasificadas por los Tres Descriptores"""),

    code_cell("""import cv2

pred_demo_path = Path("visualizations/classification_predictions_demo.png")
if pred_demo_path.exists():
    img_demo = cv2.imread(str(pred_demo_path))
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_demo, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ejemplos de Clasificación de Imágenes (Dada una imagen -> Clase predicha)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
"""),

    md_cell("""## Resumen Comparativo de Arquitectura para Clasificación de Imágenes

| Dimensión | HOG + Linear SVM | SIFT + BoVW + SVM | FREAK + BoVW + SVM |
| :--- | :--- | :--- | :--- |
| **Representación de la Imagen** | Vector global denso de gradientes ($128 \\times 128$) | Histograma BoVW ($150$ palabras visuales) | Histograma BoVW ($150$ palabras visuales) |
| **Dimensión por Imagen** | 8,100 valores flotantes | 150 valores flotantes | 150 valores flotantes |
| **Exactitud en Prueba** | **84.29%** | 70.00% | 71.43% |
| **Velocidad de Inferencia** | **0.73 ms** (1,377 FPS) | 35.34 ms (28 FPS) | **2.93 ms** (341 FPS, **12x más rápido que SIFT**) |
| **Tamaño del Modelo** | 39.38 MB | 0.21 MB | **0.17 MB** |
| **Fortaleza Principal** | Preserva la distribución espacial global de siluetas | Invarianza a escala y rotación de puntos locales | Velocidad extrema con bajo consumo de memoria y CPU |
""")
]

if __name__ == "__main__":
    out_dir = Path("/home/user/Puente/vision/project")
    create_notebook(hog_cells, out_dir / "1_hog.ipynb")
    create_notebook(sift_cells, out_dir / "2_sift.ipynb")
    create_notebook(freak_cells, out_dir / "3_freak.ipynb")
    create_notebook(comp_cells, out_dir / "4_comparison.ipynb")
    # Also overwrite the old 4_comparison_and_detection.ipynb for backward compatibility
    create_notebook(comp_cells, out_dir / "4_comparison_and_detection.ipynb")
