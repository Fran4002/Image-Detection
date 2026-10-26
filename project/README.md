# Clasificación de Imágenes (Image Detection / Classification) con OpenCV
## HOG, SIFT y FREAK

Este proyecto implementa y compara experimentalmente los tres descriptores solicitados para la tarea de **Image Detection / Classification** (dada una imagen, decir a qué clase pertenece: `apples`, `dogs` o `people`):
- **HOG** (Histogram of Oriented Gradients) + Linear SVM
- **SIFT** (Scale-Invariant Feature Transform) + Bag of Visual Words (BoVW) + Linear SVM
- **FREAK** (Fast Retina Keypoint) + FAST + Bag of Visual Words (BoVW) + Linear SVM

El proyecto se enfoca estrictamente en **clasificar la imagen completa** (no localización de objetos internos ni cajas delimitadoras).

---

## 1. Resumen de Desempeño en el Conjunto de Prueba (140 imágenes)

| Método | Exactitud (*Accuracy*) | F1 Macro | Latencia por Imagen | FPS | Tamaño Modelo | Característica Principal |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HOG + Linear SVM** | **84.29%** | **0.833** | **0.73 ms** | **1,377.8 FPS** | 39.38 MB | Mejor desempeño global capturando siluetas y bordes completos |
| **SIFT + BoVW** | 70.00% | 0.682 | 35.34 ms | 28.3 FPS | 0.21 MB | Invarianza local a escala y rotación con vocabulario visual |
| **FREAK + BoVW** | 71.43% | 0.708 | **2.93 ms** | **341.5 FPS** | **0.17 MB** | **12x más rápido que SIFT**, huella binaria ultraligera |

---

## 2. Estructura del Directorio `project/`

```
project/
├── 1_hog.ipynb             # Cuaderno interactivo de HOG para clasificar imágenes
├── 2_sift.ipynb            # Cuaderno interactivo de SIFT BoVW para clasificar imágenes
├── 3_freak.ipynb           # Cuaderno interactivo de FREAK BoVW para clasificar imágenes
├── 4_comparison.ipynb      # Cuaderno comparativo con métricas y gráficas
├── dataset.py              # Cargador robusto de imágenes (.jpg, .jpeg, .png)
├── hog_pipeline.py         # Extracción de HOG y clasificador SVM de imagen completa
├── sift_pipeline.py        # Extracción SIFT, vocabulario K-Means y clasificador BoVW
├── freak_pipeline.py       # Detector FAST + FREAK, vocabulario K-Means y clasificador BoVW
├── evaluate_all.py         # Script de benchmark automatizado
├── benchmark_summary.json  # Métricas numéricas del benchmark
├── hog_svm_model.yml       # Modelo entrenado de HOG SVM (39.38 MB)
├── sift_bovw_model.pkl     # Modelo entrenado de SIFT BoVW (0.21 MB)
├── freak_bovw_model.pkl    # Modelo entrenado de FREAK BoVW (0.17 MB)
├── REPORTE.md              # Reporte técnico completo con secciones a, b, c, d, e
└── visualizations/         # Gráficas y capturas de resultados
    ├── benchmark_metrics.png
    ├── confusion_matrices.png
    ├── feature_keypoints_demo.png
    └── classification_predictions_demo.png
```

---

## 3. Instrucciones de Ejecución

Activar el entorno virtual:
```bash
source ../.venv/bin/activate
```

Ejecutar la evaluación comparativa completa:
```bash
python evaluate_all.py
```

O abrir y ejecutar los cuadernos en Jupyter / VS Code / Antigravity.
