# Reporte Técnico: Identificación de Imagenes en OpenCV mediante HOG, SIFT y FREAK

**Proyecto:** Identificación Imagenes
**Métodos:** HOG (*Histogram of Oriented Gradients*), SIFT (*Scale-Invariant Feature Transform*) y FREAK (*Fast Retina Keypoint*)  
**Directorio de Trabajo:** `project/`  
**Entorno de Ejecución:** OpenCV 5.0 (contrib), Python 3.10, NumPy, Scikit-Learn  

---

## a. Planteamiento del problema a resolver

El reconocimiento automático de objetos en imágenes digitales representa uno de los desafíos medulares de la visión artificial. El problema a resolver en este proyecto consiste en **identificar (clasificar por categoría)** y **detectar (localizar espacialmente mediante cajas delimitadoras)** objetos visuales heterogéneos pertenecientes a tres clases semánticas distintas:
1. **Manzanas (`apples`)**
2. **Perros (`dogs`)**
3. **Personas (`people`)**

El conjunto de datos experimental se organiza en dos subconjuntos independientes:
- **Entrenamiento (`train`)**: 523 imágenes (164 manzanas, 150 perros, 209 personas).
- **Prueba (`test`)**: 140 imágenes (54 manzanas, 50 perros, 36 personas).

### Desafíos Técnicos del Problema
1. **Variabilidad intra-clase y deformación no rígida**: Los ejemplares de una misma clase exhiben gran diversidad morfométrica y postural. Por ejemplo, los perros presentan diferentes razas, pelajes y orientaciones corporales; las personas aparecen en posturas libres con diferentes prendas de vestir; y las manzanas varían en coloración, escala y grado de madurez.
2. **Heterogeneidad de resolución y formato**: El banco de datos original contiene resoluciones mixtas que abarcan desde recortes de $100 \times 100$ píxeles hasta fotogramas panorámicos de alta definición de $4080 \times 3072$ píxeles, además de extensiones de archivo compuestas (`.jpg` y `.jpeg`).
3. **Condiciones de iluminación y oclusión**: Las escenas varían entre interiores con iluminación controlada y tomas al aire libre con sombras intensas, reflejos y oclusiones parciales.
4. **Restricción de visión clásica**: El objetivo exige solucionar el problema empleando exclusivamente técnicas de extracción de características artesanales (*handcrafted visual descriptors*) provistas por la biblioteca **OpenCV**, sin depender de arquitecturas de aprendizaje profundo convolucionales de caja negra, permitiendo un control geométrico, explicabilidad matemática y un análisis comparativo riguroso de desempeño computacional.

---

## b. Descripción de la solución

Para abordar de manera integral tanto la **identificación categórica** como la **detección espacial**, se estructuró una solución modular dentro de la carpeta `project/`, compuesta por un cargador de datos unificado (`dataset.py`), pipelines específicos para cada descriptor (`hog_pipeline.py`, `sift_pipeline.py`, `freak_pipeline.py`), un script de evaluación comparativa (`evaluate_all.py`) y cuadernos interactivos documentados (`1_hog.ipynb`, `2_sift.ipynb`, `3_freak.ipynb`, `4_comparison_and_detection.ipynb`).

---

### 1. Descriptor HOG (Histogram of Oriented Gradients)

#### Teoría y Fundamentos
Propuesto originalmente por Navneet Dalal y Bill Triggs en el congreso CVPR 2005 para la detección de peatones, el descriptor HOG se fundamenta en la premisa de que la apariencia y la forma de un objeto local pueden caracterizarse con gran fidelidad mediante la distribución espacial de las intensidades y direcciones de los gradientes de borde, incluso sin conocer con precisión las posiciones exactas de los bordes.

El cálculo del descriptor involucra cuatro etapas clave:
1. **Cálculo del gradiente de la imagen**: Se convolucionan filtros unidimensionales de derivadas directas $[-1, 0, 1]$ y $[-1, 0, 1]^T$ sobre la imagen en escala de grises para obtener las componentes del gradiente horizontal $G_x(x, y)$ y vertical $G_y(x, y)$. La magnitud y el ángulo del gradiente se determinan mediante:
   $$M(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$
   $$\theta(x, y) = \arctan\left(\frac{G_y(x, y)}{G_x(x, y)}\right)$$
2. **Acumulación de histogramas espaciales en celdas**: La imagen se divide en celdas espaciales pequeñas (típicamente de $8 \times 8$ píxeles). Dentro de cada celda, los píxeles votan en un histograma unidimensional de orientaciones discretizadas en $B$ canales angulares (usualmente $B = 9$ contenedores entre $0^\circ$ y $180^\circ$ para gradientes no orientados). El peso del voto es proporcional a la magnitud del gradiente $M(x, y)$.
3. **Normalización por bloques superpuestos**: Para garantizar invariancia a variaciones lumínicas, sombras e iluminaciones locales, las celdas se agrupan en bloques espacialmente contiguos y superpuestos (por ejemplo, bloques de $2 \times 2$ celdas, es decir, $16 \times 16$ píxeles, con un desplazamiento o *stride* de $8 \times 8$ píxeles). Sobre cada vector de bloque $v$, se aplica una normalización $L_2\text{-hys}$ (norma $L_2$ con recorte de valores máximos a $0.2$ y renormalización):
   $$v \leftarrow \frac{v}{\sqrt{\|v\|_2^2 + \epsilon^2}}$$
4. **Vector final de características**: La concatenación secuencial de todos los bloques normalizados produce el vector de características unidimensional de la imagen.

#### Investigación y Corrección del Trabajo Previo
Al auditar `mine/hog.ipynb` se detectaron dos deficiencias críticas:
- **Omisión de manzanas**: El filtro de búsqueda utilizaba únicamente `glob("*.jpg") + glob("*.png")`. Dado que todas las manzanas son `.jpeg`, fueron totalmente ignoradas.
- **Explosión dimensional**: Se utilizaba una ventana de entrada de $512 \times 512$ píxeles. Con bloques de $16 \times 16$ y desplazamiento de $8 \times 8$, se generaban:
  $$\left(\frac{512 - 16}{8} + 1\right) \times \left(\frac{512 - 16}{8} + 1\right) = 63 \times 63 = 3969 \text{ bloques}$$
  Multiplicado por $4 \text{ celdas} \times 9 \text{ bins} = 36 \text{ valores por bloque}$, arrojaba **142,884 dimensiones por imagen**. Esto produjo que el archivo serializado de la SVM (`hog_svm_model.yml`) alcanzara un peso inmanejable de **628.7 MB**.

**Solución aplicada**: Se estandarizó la ventana de análisis a $128 \times 128$ píxeles. Esto genera:
$$\left(\frac{128 - 16}{8} + 1\right) \times \left(\frac{128 - 16}{8} + 1\right) = 15 \times 15 = 225 \text{ bloques} \implies 225 \times 36 = \mathbf{8,100 \text{ dimensiones}}$$
Esto redujo el modelo a solo **39.38 MB** (una compresión superior al 93%), acelerando el entrenamiento en menos de un segundo y mejorando la capacidad de generalización.

#### Funciones de OpenCV y Parámetros
- `cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, ...)`:
  - `winSize=(128, 128)`: Tamaño de la ventana de detección sobre la cual se extrae el vector representativo.
  - `blockSize=(16, 16)`: Dimensión física del bloque normalizador.
  - `blockStride=(8, 8)`: Desplazamiento horizontal y vertical entre bloques contiguos.
  - `cellSize=(8, 8)`: Tamaño de la celda donde se acumulan los histogramas locales.
  - `nbins=9`: Número de contenedores angulares de orientación.
- `cv2.ml.SVM_create()`:
  - `setType(cv2.ml.SVM_C_SVC)`: Clasificador de soporte vectorial para múltiples clases.
  - `setKernel(cv2.ml.SVM_LINEAR)`: Kernel lineal para discriminación eficiente en espacios de alta dimensionalidad.
  - `setC(1.0)`: Parámetro de regularización que equilibra el margen y el error de entrenamiento.

#### Fragmento de Código Implementado
```python
# Inicialización del descriptor HOG optimizado
hog = cv2.HOGDescriptor(
    (128, 128),  # win_size
    (16, 16),    # block_size
    (8, 8),      # block_stride
    (8, 8),      # cell_size
    9            # nbins
)

def compute_hog_features(img, win_size=(128, 128)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if gray.shape != (win_size[1], win_size[0]):
        gray = cv2.resize(gray, win_size)
    return hog.compute(gray).flatten().astype(np.float32)

# Entrenamiento de SVM con OpenCV
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)
svm.train(X_train_hog, cv2.ml.ROW_SAMPLE, y_train_encoded.reshape(-1, 1))
```

Para la **detección espacial**, se implementó un algoritmo de ventana deslizante multiescala con pirámides gaussianas y un algoritmo de supresión de no máximos (*Non-Maximum Suppression*, NMS) para fusionar las detecciones solapadas.

---

### 2. Descriptor SIFT (Scale-Invariant Feature Transform)

#### Teoría y Fundamentos
Desarrollado por David Lowe (Journal International of Computer Vision, 2004), SIFT es uno de los algoritmos más representativos de la visión artificial clásica, diseñado para ser totalmente invariante a transformaciones de escala, rotación, cambios afines tridimensionales moderados y variaciones en la iluminación.

El algoritmo opera en cuatro fases principales:
1. **Detección de extremos en el espacio-escala**: La imagen se transforma progresivamente aplicando filtros gaussianos $L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$ a lo largo de varias octavas de escala. Las diferencias entre escalas gaussianas consecutivas generan el espacio de Diferencia de Gaussianas (DoG):
   $$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y) = L(x, y, k\sigma) - L(x, y, \sigma)$$
   Un punto se considera candidato si es un máximo o mínimo local estricto respecto a sus 26 vecinos (8 en la misma escala, 9 en la escala superior y 9 en la inferior).
2. **Localización refinada y eliminación de bordes**: Se calcula la expansión en serie de Taylor de segundo orden en el punto candidato para hallar su posición continua sub-pixel. Se descartan puntos con bajo contraste ($\|D(\hat{x})\| < 0.03$) y aquellos localizados sobre bordes utilizando la traza y el determinante de la matriz Hessiana $H$.
3. **Asignación de orientación canónica**: A partir del gradiente de la vecindad alrededor del punto clave en su escala característica, se construye un histograma de orientaciones de 36 contenedores. La dirección dominante define la orientación intrínseca del punto clave, otorgando invariancia total ante rotaciones.
4. **Construcción del descriptor SIFT (128-D)**: Se toma una región local de $16 \times 16$ píxeles orientada según el ángulo canónico del punto clave. Dicha región se divide en una cuadrícula de $4 \times 4$ subregiones. En cada subregión se construye un histograma de gradientes de 8 direcciones ponderado por una función de decaimiento gaussiano. El descriptor final es la concatenación de estos $4 \times 4 \times 8 = \mathbf{128 \text{ valores flotantes}}$, el cual se normaliza con norma $L_2$, se satura a un umbral de $0.2$ y se renormaliza para mitigar efectos no lineales de iluminación.

#### Implementación: Dos Paradigmas de Detección e Identificación
1. **Identificación Categórica mediante Bolsa de Palabras Visuales (BoVW)**:
   - Dado que dos perros diferentes o dos personas distintas no comparten puntos clave SIFT idénticos en las mismas coordenadas, el emparejamiento directo 1 a 1 resulta ineficaz.
   - En su lugar, se extraen todos los descriptores SIFT del conjunto de entrenamiento y se agrupan mediante el algoritmo de K-Medias (`MiniBatchKMeans`) para construir un **vocabulario visual** de $k = 150$ palabras visuales (*visual words*).
   - Para clasificar una imagen nueva, cada uno de sus descriptores se cuantiza hacia la palabra visual más cercana, generando un histograma de frecuencias normalizado de dimensión $150$. Posteriormente, una máquina de soporte vectorial (`LinearSVC`) clasifica la imagen.
2. **Detección y Localización Geométrica mediante Homografía RANSAC**:
   - Cuando se busca localizar un objeto de plantilla dentro de una escena compleja (por ejemplo, detectar la presencia de una persona o un objeto específico entre fotogramas consecutivos de video), se calculan las correspondencias directas mediante `cv2.BFMatcher(cv2.NORM_L2)`.
   - Se aplica el test de razón de Lowe (*Lowe's ratio test*): una correspondencia se acepta únicamente si:
     $$\text{distancia}(m) < 0.75 \times \text{distancia}(n)$$
   - Con las correspondencias filtradas, se estima la matriz de homografía $H$ de $3 \times 3$ mediante RANSAC (`cv2.findHomography`).
   - Los cuatro vértices de la plantilla se proyectan sobre la escena mediante `cv2.perspectiveTransform`, dibujando el polígono/caja delimitadora que confirma y localiza el objeto.

#### Funciones de OpenCV y Parámetros
- `cv2.SIFT_create(nfeatures=400, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)`:
  - `nfeatures`: Límite superior de puntos clave dominantes ordenados por respuesta.
  - `contrastThreshold`: Filtra puntos débiles en zonas de bajo contraste.
  - `edgeThreshold`: Umbral para filtrar puntos clave inestables situados sobre bordes.
- `cv2.BFMatcher(cv2.NORM_L2)`: Comparador por fuerza bruta que utiliza distancia euclidiana entre descriptores de punto flotante de 128 elementos.
- `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)`: Cálculo robusto de la transformación proyectiva descartando valores atípicos (*outliers*) con un umbral de tolerancia de reproyección de 5.0 píxeles.

#### Fragmento de Código Implementado
```python
# Extracción y emparejamiento con SIFT y RANSAC
sift = cv2.SIFT_create(nfeatures=500)
kp1, des1 = sift.detectAndCompute(template_img, None)
kp2, des2 = sift.detectAndCompute(scene_img, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Test de razón de Lowe
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None:
        h, w = template_img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected_box = cv2.perspectiveTransform(corners, H)
```

---

### 3. Descriptor FREAK (Fast Retina Keypoint)

#### Teoría y Fundamentos
Presentado por Alexandre Alahi, Raphaël Ortiz y Pierre Vandergheynst en el congreso CVPR 2012, FREAK es un descriptor visual binario bio-inspirado en la fisiología del ojo humano, específicamente en la distribución anatómica de las células ganglionares de la retina:
- **Patrón de muestreo retiniano circular**: A diferencia de las cuadrículas rectangulares rígidas (como SIFT o HOG), FREAK utiliza un patrón de muestreo foveal compuesto por círculos concéntricos de campos receptivos gaussianos. La densidad de los puntos de muestreo es extremadamente alta en el centro (emulando la fóvea central de alta agudeza visual) y disminuye exponencialmente hacia la periferia (visión periférica).
- **Campos receptivos con solapamiento variable**: Cada punto de muestreo se filtra con un núcleo gaussiano cuyo tamaño $\sigma$ crece a medida que se aleja del centro, imitando los tamaños de los campos receptivos de las células de la retina humana.
- **Descriptor Binario en Cascada**: Un descriptor FREAK típico consta de una cadena de **512 bits (64 bytes)**. Cada bit surge de una comparación simple de intensidad binaria entre un par de campos receptivos filtrados $(p_i, p_j)$:
  $$b = \begin{cases} 1 & \text{si } I(p_i) > I(p_j) \\ 0 & \text{en caso contrario} \end{cases}$$
  De miles de pares posibles, los autores aplicaron un algoritmo de selección codicioso (*greedy*) que escogió los 512 pares con mayor varianza y menor correlación mutua.
- **Eficiencia de la Distancia de Hamming**: Comparar dos descriptores de 512 bits no requiere multiplicaciones de punto flotante ni cálculos de raíces cuadradas; basta con una operación lógica XOR a nivel de registro de procesador seguida del conteo de bits en 1 (`POPCNT`), permitiendo velocidades de emparejamiento hasta dos órdenes de magnitud superiores a SIFT.
- **Naturaleza del Descriptor**: Es imperativo subrayar que FREAK es **únicamente un extractor de descriptores**, no un detector de puntos clave. Requiere ser alimentado con coordenadas de puntos clave previamente detectadas por algoritmos como **FAST** (*Features from Accelerated Segment Test*) o **BRISK**.

#### Implementación
En nuestro pipeline (`freak_pipeline.py`), se emparejó el detector rápido FAST con el descriptor FREAK:
1. **Detección con FAST**: `cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)` analiza la circunferencia de Bresenham de 16 píxeles alrededor de cada píxel, identificando esquinas de forma ultra-rápida.
2. **Cómputo con FREAK**: `cv2.xfeatures2d.FREAK_create()` calcula los descriptores binarios de 64 bytes.
3. **Clasificación Categórica (BoVW)**: Para categorización, los descriptores binarios se agruparon en un libro de códigos de 150 palabras visuales (`MiniBatchKMeans`) para entrenar un clasificador `LinearSVC`.
4. **Emparejamiento y Localización Espacial**: Se configuró `cv2.BFMatcher(cv2.NORM_HAMMING)`. El test de razón se adaptó a distancias enteras de Hamming ($0.80$), y se estimó la homografía RANSAC para proyectar la caja delimitadora en imágenes de escena.

#### Funciones de OpenCV y Parámetros
- `cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)`:
  - `threshold`: Diferencia mínima de intensidad entre el píxel central y los píxeles del círculo de prueba.
  - `nonmaxSuppression`: Elimina respuestas espaciales redundantes adyacentes.
- `cv2.xfeatures2d.FREAK_create(orientationNormalized=True, scaleNormalized=True, patternScale=22.0, nOctaves=4)`:
  - `orientationNormalized`: Normaliza el ángulo del descriptor para asegurar invariancia a rotación.
  - `scaleNormalized`: Normaliza respecto a la escala del punto clave.
- `cv2.BFMatcher(cv2.NORM_HAMMING)`: Comparador por fuerza bruta que utiliza distancia de Hamming (conteo de bits diferentes).

#### Fragmento de Código Implementado
```python
# Detector FAST + Extractor FREAK
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
freak = cv2.xfeatures2d.FREAK_create()

# Detección y cómputo de descriptores binarios
kps = fast.detect(gray_img, None)
kps = sorted(kps, key=lambda k: k.response, reverse=True)[:400]
kps, descs = freak.compute(gray_img, kps)

# Emparejamiento por distancia de Hamming (Bitwise XOR)
bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf_hamming.knnMatch(descs_query, descs_scene, k=2)

# Filtrado por razón de distancias de Hamming
good_matches = [m for m, n in matches if m.distance < 0.80 * n.distance]
```

---

### Fuentes Bibliográficas Consultadas
1. **Dalal, N., & Triggs, B. (2005)**. *Histograms of oriented gradients for human detection*. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), Vol. 1, pp. 886-893. IEEE.
2. **Lowe, D. G. (2004)**. *Distinctive image features from scale-invariant keypoints*. International Journal of Computer Vision, 60(2), 91-110.
3. **Alahi, A., Ortiz, R., & Vandergheynst, P. (2012)**. *FREAK: Fast retina keypoint*. In 2012 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 510-517. IEEE.
4. **OpenCV Documentation (v5.0.0-dev)**. *OpenCV modules: Feature Detection and Description (`cv2.SIFT`, `cv2.HOGDescriptor`, `cv2.xfeatures2d.FREAK`)*. https://docs.opencv.org/

---

## c. Descripción de los resultados

La evaluación experimental se ejecutó de forma estandarizada sobre la partición completa de prueba (`test`), compuesta por **140 imágenes independientes** que nunca participaron en el ajuste de los vocabularios ni en el entrenamiento de los clasificadores.

### Tabla Comparativa de Desempeño

| Métrica / Dimensión | HOG + Linear SVM | SIFT + BoVW + SVM | FREAK + BoVW + SVM |
| :--- | :---: | :---: | :---: |
| **Exactitud en Prueba (*Accuracy*)** | **84.29%** | 70.00% | 71.43% |
| **F1-Score Macro** | **0.833** | 0.682 | 0.708 |
| **F1-Score Ponderado (*Weighted*)** | **0.840** | 0.697 | 0.717 |
| **Tiempo de Entrenamiento** | **1.78 s** | 23.16 s | 4.57 s |
| **Latencia de Inferencia por Imagen** | **0.73 ms** | 35.34 ms | 2.93 ms |
| **Rendimiento de Procesamiento (FPS)** | **1,377.8 FPS** | 28.3 FPS | **341.5 FPS** |
| **Tamaño del Modelo Serializado** | 39.38 MB | 0.21 MB | **0.17 MB** |
| **Tipo de Descriptor** | Grilla densa (Gradientes) | Puntos clave flotantes | Puntos clave binarios |
| **Dimensión por Descriptor** | 8,100 flotantes | 128 flotantes | 64 bytes (512 bits) |
| **Métrica de Distancia / Separación** | Hiperplano $L_2$ | Euclidiana ($L_2$) | Hamming (XOR + POPCNT) |

---

### Análisis de Métricas y Comparativa Visual

A continuación se presentan las gráficas comparativas cuantitativas generadas durante el proceso de evaluación:

![Métricas Comparativas](/home/user/.gemini/antigravity/brain/94528083-60bd-4cec-a4ff-214b2bcb3856/benchmark_metrics.png)

1. **Exactitud**: HOG demostró la mayor precisión global (84.29%), superando a los métodos locales BoVW. Esto se debe a que la estructura morfológica global de las manzanas, perros y personas es altamente distintiva en una cuadrícula fija normalizada.
2. **Latencia y FPS**: FREAK exhibe una velocidad sobresaliente, alcanzando **341.5 FPS** frente a los **28.3 FPS** de SIFT, lo que representa una aceleración de **más de 12 veces** en favor de FREAK. HOG, al operar como vector compacto proyectado en un clasificador lineal, alcanzó más de 1,300 FPS en inferencia directa.
3. **Espacio en Disco**: Los modelos BoVW de SIFT (0.21 MB) y FREAK (0.17 MB) son órdenes de magnitud más ligeros que el modelo HOG (39.38 MB), debido a que el libro de códigos resume todo el espacio visual en solo 150 centroides.

---

### Matrices de Confusión por Descriptor

El comportamiento detallado de clasificación por clase se ilustra en las siguientes matrices de confusión:

![Matrices de Confusión](/home/user/.gemini/antigravity/brain/94528083-60bd-4cec-a4ff-214b2bcb3856/confusion_matrices.png)

#### Desglose por Clase:
- **Manzanas (`apples`, 54 muestras)**:
  - HOG clasificó correctamente **47 de 54** manzanas (87% de recall).
  - SIFT identificó **46 de 54**.
  - FREAK identificó **48 de 54** (88.9% de recall), demostrando que las texturas periféricas circulares de la manzana son capturadas eficientemente por los campos receptivos foveales.
- **Perros (`dogs`, 50 muestras)**:
  - HOG acertó en **44 de 50** perros (88% de recall).
  - SIFT acertó en **32 de 50** (confundiendo 14 con manzanas debido a texturas de pelaje que activan palabras visuales similares).
  - FREAK acertó en **33 de 50**.
- **Personas (`people`, 36 muestras)**:
  - HOG identificó correctamente **27 de 36** personas (75% de recall).
  - SIFT identificó **20 de 36**.
  - FREAK identificó **19 de 36**.

---

### Ilustración de Puntos Clave y Detección Espacial

#### 1. Comparación de Puntos Clave: SIFT vs FAST + FREAK
La siguiente figura ilustra la naturaleza multi-escala continua de los puntos SIFT frente a la distribución densa retiniana de FREAK en una imagen del conjunto de prueba:

![Puntos Clave SIFT vs FREAK](/home/user/.gemini/antigravity/brain/94528083-60bd-4cec-a4ff-214b2bcb3856/feature_keypoints_demo.png)

#### 2. Detección Espacial de Objetos mediante HOG (Ventana Deslizante + NMS)
En la tarea de detección espacial multiescala, el pipeline HOG barre la imagen con diversas escalas y utiliza supresión de no máximos para delimitar con exactitud el objeto de interés:

![Detección HOG](/home/user/.gemini/antigravity/brain/94528083-60bd-4cec-a4ff-214b2bcb3856/hog_detection_demo.jpg)

#### 3. Detección y Localización Geométrica mediante Homografía RANSAC (SIFT y FREAK)
En secuencias de video o escenas dinámicas, SIFT y FREAK permiten emparejar los puntos característicos de un objeto de referencia contra una escena y proyectar la caja delimitadora mediante la homografía calculada con RANSAC:

````carousel
![Localización con SIFT](/home/user/.gemini/antigravity/brain/94528083-60bd-4cec-a4ff-214b2bcb3856/sift_detection_demo.jpg)
<!-- slide -->
![Localización con FREAK](/home/user/.gemini/antigravity/brain/94528083-60bd-4cec-a4ff-214b2bcb3856/freak_detection_demo.jpg)
````

En la demostración de localización entre fotogramas consecutivos de video:
- **SIFT**: Obtuvo **282 correspondencias** y **249 inliers geométricos** verificados por RANSAC, proyectando la caja delimitadora verde con altísima estabilidad.
- **FREAK**: Obtuvo **101 correspondencias** y **67 inliers geométricos** con distancia de Hamming en una fracción del tiempo de cálculo, delimitando con éxito el sujeto en la escena.

---

## d. Discusión

### Análisis Crítico: ¿Cuál de los 3 descriptores tuvo mejor desempeño y por qué?

El concepto de "mejor desempeño" en visión artificial no puede reducirse a una sola métrica aislada, sino que debe sopesarse en función de la naturaleza de la tarea:

1. **Para Clasificación Categórica Holística (Identificación de Clase): El Ganador Absoluto fue HOG.**
   - **Justificación**: HOG alcanzó un **84.29% de exactitud** y un F1 ponderado de **0.840**, superando con holgura a SIFT (70.00%) y a FREAK (71.43%).
   - **Explicación Científica**: HOG codifica la *silueta espacial global* del objeto. Una manzana tiene una silueta curvada cerrada; una persona de pie tiene una silueta esbelta vertical; un perro posee una estructura anatómica horizontal cuadrúpeda característica. Al normalizar la imagen a $128 \times 128$, HOG retiene la relación topológica fija entre la cabeza, el torso o el fondo. En cambio, los métodos basados en Bolsa de Palabras Visuales (BoVW con SIFT o FREAK) son representaciones de *histograma de orden cero* (modelo de "bolsa"), lo que significa que descartan por completo la geometría espacial relativa: una imagen donde los ojos, patas y cola estén desordenados producirá exactamente el mismo histograma BoVW que un perro completo.

2. **Para Detección en Tiempo Real y Restricción de Recursos: El Ganador fue FREAK.**
   - **Justificación**: FREAK demostró ser **12 veces más rápido que SIFT** (2.93 ms vs 35.34 ms por imagen, operando a **341.5 FPS**) mientras obtuvo una exactitud BoVW superior (71.43% frente a 70.00% de SIFT) y el menor peso de modelo (0.17 MB).
   - **Explicación Científica**: El diseño foveal de FREAK concentra los puntos de muestreo en el centro del punto clave, capturando la información de alta frecuencia más discriminativa, mientras que la periferia filtra el contexto. Además, el cálculo de distancias de Hamming mediante instrucciones nativas de procesador (`XOR` y `POPCNT`) elimina la necesidad de aritmética de punto flotante pesada requerida por el vector de 128 dimensiones de SIFT.

3. **Para Transformaciones Geométricas Extremas y Fidelidad de Homografía: SIFT se mantiene como Referente de Precisión.**
   - **Justificación**: En la estimación de homografías mediante RANSAC sobre secuencias de video, SIFT produjo **249 inliers válidos**, mientras que FREAK produjo **67 inliers**.
   - **Explicación Científica**: La localización sub-píxel de SIFT y su descriptor de gradiente continuo de 128 dimensiones son intrínsecamente más resistentes a rotaciones arbitrarias continuas y deformaciones afines severas que los umbrales de intensidad discretos de un descriptor binario.

### Lecciones del Diagnóstico del Código Previo
La auditoría de `mine/` demostró la importancia crítica de la validación del flujo de datos en proyectos de visión computacional:
- Un error tipográfico en la extensión de archivo (`.png` en lugar de `.jpeg`) dejó fuera una categoría completa (manzanas) sin emitir ninguna excepción en tiempo de ejecución.
- La selección de hiperparámetros de ventana en HOG demostró el fenómeno del compromiso sesgo-varianza: una ventana excesivamente grande ($512 \times 512$) no aportó mayor poder discriminativo, pero sí saturó el espacio de almacenamiento con un archivo de más de 600 MB. La reducción a $128 \times 128$ optimizó el compromiso entre detalle de gradiente y compacidad matemática.

---

## e. Conclusión

El desarrollo de este proyecto permitió explorar a profundidad los fundamentos teóricos, matemáticos y prácticos de tres de los algoritmos más representativos de la visión artificial clásica: HOG, SIFT y FREAK.

A lo largo del proyecto se implementaron con éxito dos metodologías complementarias:
1. La formulación de **Bolsa de Palabras Visuales (BoVW)** combinada con máquinas de soporte vectorial para el reconocimiento de categorías generales.
2. La técnica de **emparejamiento de correspondencias con filtrado de Lowe y homografía RANSAC** para la detección y delimitación espacial precisa de objetos en escenas complejas y fotogramas de video.

### Reflexión sobre el Uso de OpenCV
La biblioteca **OpenCV** demostró ser una plataforma madura, versátil y sumamente potente para abordar problemas de visión por computadora:
- La implementación en C++ subyacente proporciona un rendimiento excepcional, permitiendo procesar cientos e incluso miles de imágenes por segundo en CPUs estándar sin necesidad de aceleradores gráficos dedicados (GPU).
- Su interoperabilidad nativa con estructuras matriciales de `NumPy` y herramientas de análisis estadístico de `Scikit-Learn` facilita la creación de pipelines híbridos robustos.
- La disponibilidad del módulo `xfeatures2d` (contrib) enriquece el abanico de posibilidades al incluir avances bio-inspirados modernos como FREAK, cerrando la brecha entre la neurobiología de la visión humana y la ingeniería de software.

En conclusión, mientras que las arquitecturas modernas de Deep Learning suelen dominar la atención contemporánea, el dominio de descriptores clásicos como HOG, SIFT y FREAK es indispensable. Estos métodos no solo ofrecen una base matemática transparente y explicable, sino que continúan siendo la opción predilecta en aplicaciones embebidas, robótica móvil en tiempo real, dispositivos de bajo consumo y situaciones donde los datos de entrenamiento son limitados.
