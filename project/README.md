# Computer Vision Object Identification & Detection
## HOG, SIFT, and FREAK Pipelines with OpenCV

This project implements, refines, and benchmarks the three requested OpenCV methods:
- **HOG** (Histogram of Oriented Gradients)
- **SIFT** (Scale-Invariant Feature Transform)
- **FREAK** (Fast Retina Keypoint)

---

## 1. Audit and Improvements to Prior Work in `mine/`

### Issues Identified in `mine/hog.ipynb`:
1. **Critical Data Omission (.jpeg omission)**:
   - In `mine/hog.ipynb`, the file glob only checked for `*.jpg` and `*.png`:
     ```python
     files = sorted(list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png")))
     ```
   - All **164 training apples** and **54 test apples** are stored as `.jpeg`. Consequently, **0 apples** were loaded, training a biased 2-class model instead of a 3-class model.
   - **Fix**: Added case-insensitive matching for `.jpg`, `.jpeg`, and `.png` across all classes.

2. **Explosion of Descriptor Dimensions & Model Size (628 MB)**:
   - The original code resized all images to `(512, 512)` with `(16, 16)` block size and `(8, 8)` stride.
   - This generated **142,884 features per image**.
   - As a result, OpenCV's SVM model file `hog_svm_model.yml` ballooned to **628.7 MB**, causing slow training, high memory usage, and extreme curse-of-dimensionality risk.
   - **Fix**: Replaced `(512, 512)` with `(128, 128)` window size (8,100 features). This reduced model size to **39.38 MB** (over 93% smaller), trained in **0.91 seconds**, and reached **84.29% test accuracy**.

3. **Missing Test Set Evaluation**:
   - `mine/hog.ipynb` loaded `X_test_images`, but never extracted features, never predicted labels, and reported no accuracy, precision, recall, or confusion matrix.
   - **Fix**: Added comprehensive evaluation on all 140 test images.

---

### Issues Identified in `mine/sift.ipynb`:
1. **Critical Array Corruption Bug (Cell 9)**:
   - In Cell 9, the test loop appended descriptors to the training list instead of the test list:
     ```python
     for img_item in X_test_images:
         item_kp, item_des = sift.detectAndCompute(img_item, None)
         train_kps.append(item_kp)   # BUG: appended to train instead of test
         train_des.append(item_des)
     ```
   - This left `test_kps` and `test_des` completely empty while corrupting `train_kps`.

2. **Re-assignment Bug (Cell 7)**:
   - In Cell 7, `bf.knnMatch(train_des, test_des, k=2)` was called without assigning the result to `matches`, causing subsequent loops to process stale matches.

3. **Algorithmic Scalability (1-NN Brute Force vs BoVW)**:
   - Direct pairwise matching across all 523 training images (`match_all_images`) takes seconds per test image and is fragile to intra-class variance (e.g. different dog breeds or camera angles).
   - **Fix**: Introduced **Bag of Visual Words (BoVW)** codebook quantization with SVM for class identification, and **Lowe's Ratio Test + RANSAC Homography** for spatial object detection & bounding box projection.

---

### FREAK (Fast Retina Keypoint) Implementation:
- FREAK was completely missing from `mine/`.
- Implemented in `freak_pipeline.py` and `3_freak.ipynb`:
  - **Keypoint Detector**: Pair FAST (`cv2.FastFeatureDetector_create`) or BRISK with `cv2.xfeatures2d.FREAK_create()`.
  - **Bio-Inspired Foveal Receptive Fields**: High density near center, decreasing density outward.
  - **Binary Hamming Metric**: Matched with `cv2.BFMatcher(cv2.NORM_HAMMING)`, achieving **~340 FPS** (~12x faster than SIFT).

---

## 2. Benchmark Summary

Evaluated on the full test split (140 images across `apples`, `dogs`, `people`):

| Method | Test Accuracy | Macro F1 | Inference Latency | Throughput (FPS) | Model File Size | Primary Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HOG + Linear SVM** | **84.29%** | **0.833** | **0.73 ms** | **1,377.8 FPS** | 39.38 MB | Best overall accuracy for global category classification |
| **SIFT + BoVW** | 70.00% | 0.682 | 35.34 ms | 28.3 FPS | 0.21 MB | Invariant to extreme rotation, scale, and affine changes |
| **FREAK + BoVW** | 71.43% | 0.708 | **2.93 ms** | **341.5 FPS** | **0.17 MB** | 12x faster than SIFT; ultra-compact binary footprint |

---

## 3. Directory Layout

```
project/
├── 1_hog.ipynb                     # Interactive HOG notebook
├── 2_sift.ipynb                    # Interactive SIFT notebook
├── 3_freak.ipynb                   # Interactive FREAK notebook
├── 4_comparison_and_detection.ipynb# Unified comparison & detection demo
├── dataset.py                      # Robust multi-format dataset loader
├── hog_pipeline.py                 # HOG feature extractor & multiscale sliding-window detector
├── sift_pipeline.py                # SIFT BoVW classifier & homography object localizer
├── freak_pipeline.py               # FREAK detector, BoVW classifier & binary matcher
├── evaluate_all.py                 # Benchmark script across all 3 methods
├── benchmark_summary.json          # Numerical benchmark results
├── hog_svm_model.yml               # Compact trained HOG SVM model (39 MB)
├── sift_bovw_model.pkl             # Trained SIFT BoVW model (0.21 MB)
├── freak_bovw_model.pkl            # Trained FREAK BoVW model (0.17 MB)
└── visualizations/                 # Saved detection and bounding box demos
    ├── sift_detection_demo.jpg
    └── freak_detection_demo.jpg
```

---

## 4. How to Run

Activate the virtual environment:
```bash
source ../.venv/bin/activate
```

Run the complete comparative benchmark:
```bash
python evaluate_all.py
```

Or open and run any of the notebooks interactively in Jupyter / VS Code / Antigravity.
