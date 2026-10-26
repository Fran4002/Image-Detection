"""
Generator script to produce clean, well-documented Jupyter Notebooks for:
1. 1_hog.ipynb
2. 2_sift.ipynb
3. 3_freak.ipynb
4. 4_comparison_and_detection.ipynb
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
    md_cell("""# 1. Histogram of Oriented Gradients (HOG) & SVM Pipeline
### Improved Object Identification and Detection

This notebook addresses and fixes the key limitations found in the original `mine/hog.ipynb`:
1. **Missing Classes**: Fixed the `.jpeg` file glob bug that previously omitted all **164 training apples** and **54 test apples**.
2. **Dimension & Model Explosion**: Replaced the huge `(512, 512)` window size (which generated **142,884 features** and a **628 MB** SVM model!) with a standardized `(128, 128)` window (8,100 features, ~39 MB model, fast training and inference).
3. **Rigorous Test Evaluation**: Added systematic test evaluation on the full test set with **Accuracy**, **Precision**, **Recall**, **F1-score**, and a **Confusion Matrix**.
4. **Object Detection**: Implemented multiscale sliding-window detection with **Non-Maximum Suppression (NMS)** to draw localized bounding boxes around detected objects.
"""),

    code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split
from hog_pipeline import HOGPipeline, non_max_suppression_fast

def show_rgb(img, title=None):
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12)
    plt.show()
"""),

    md_cell("""## Step 1: Load Dataset (with .jpeg support)
Notice that the dataset contains `apples`, `dogs`, and `people`. All 3 classes are loaded dynamically.
"""),

    code_cell("""# Load train and test splits
train_imgs, train_lbls, classes, _ = load_split("train")
test_imgs, test_lbls, _, _ = load_split("test")

print(f"Classes Discovered: {classes}")
print(f"Training set: {len(train_imgs)} images")
print(f"Testing set:  {len(test_imgs)} images")

for c in classes:
    print(f"  - {c}: {train_lbls.count(c)} train, {test_lbls.count(c)} test")
"""),

    md_cell("""## Step 2: Configure HOG Descriptor
We use:
- `win_size = (128, 128)`
- `block_size = (16, 16)`
- `block_stride = (8, 8)`
- `cell_size = (8, 8)`
- `nbins = 9`

Feature Dimension = $((128-16)/8 + 1) \\times ((128-16)/8 + 1) \\times 4 \\times 9 = 15 \\times 15 \\times 36 = 8,100$.
"""),

    code_cell("""pipeline = HOGPipeline(win_size=(128, 128))
sample_des = pipeline.compute_descriptor(train_imgs[0])
print(f"HOG Descriptor dimension per image: {sample_des.shape[0]} features")
"""),

    md_cell("""## Step 3: Train SVM Classifier
We train OpenCV's SVM (`cv2.ml.SVM`) with a Linear kernel.
"""),

    code_cell("""print("Training HOG + SVM on training set...")
train_info = pipeline.train(train_imgs, train_lbls, kernel_type="linear", C=1.0)
print(f"Training completed in {train_info['train_time_sec']:.2f} seconds!")
print(f"Feature dimension: {train_info['feature_dim']}")
"""),

    md_cell("""## Step 4: Evaluate on Full Test Set
We evaluate accuracy, per-class precision/recall, and the confusion matrix on all 140 test images.
"""),

    code_cell("""eval_results = pipeline.evaluate(test_imgs, test_lbls)
print(f"HOG Test Accuracy: {eval_results['accuracy'] * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(test_lbls, eval_results['predictions'], target_names=pipeline.classes_))

# Plot Confusion Matrix
cm = np.array(eval_results["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("HOG Confusion Matrix")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if i != j else "black")
plt.colorbar(im)
plt.show()
"""),

    md_cell("""## Step 5: Save Model & Verify File Footprint
The model is saved to `hog_svm_model.yml`. Notice how compact it is compared to the original 628 MB model!
"""),

    code_cell("""pipeline.save_model("hog_svm_model.yml")
model_size_mb = Path("hog_svm_model.yml").stat().st_size / (1024 * 1024)
print(f"Saved HOG SVM Model: hog_svm_model.yml")
print(f"Model File Size:     {model_size_mb:.2f} MB")
"""),

    md_cell("""## Step 6: Object Detection in Scenes (Sliding Window & Bounding Box)
We can scan sliding windows over a scene image to detect and localize objects.
"""),

    code_cell("""# Test sliding-window detector on an apple test image
test_scene = test_imgs[0].copy()
detections = pipeline.detect_multiscale(test_scene, min_win=(64, 64), max_win=(160, 160), step_size=16)

annotated = test_scene.copy()
for det in detections:
    x1, y1, x2, y2 = det['box']
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated, f"{det['class']}: {det['score']:.1f}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

show_rgb(annotated, title=f"HOG Multiscale Detections ({len(detections)} found)")
""")
]


# ==============================================================================
# Notebook 2: SIFT
# ==============================================================================
sift_cells = [
    md_cell("""# 2. SIFT (Scale-Invariant Feature Transform) Pipeline
### Category Identification (BoVW) & Pairwise Object Detection (Homography)

This notebook addresses and fixes the bugs and limitations in `mine/sift.ipynb`:
1. **Critical Bug Fixed**: In `mine/sift.ipynb` Cell 9, test descriptors were appended to `train_kps` (`train_kps.append(item_kp)`), corrupting the training array and leaving `test_kps` empty.
2. **Re-assignment Bug Fixed**: In Cell 7, `bf.knnMatch` result was not stored, causing the loop to use obsolete matches.
3. **Bag of Visual Words (BoVW)**: Brute-force 1-NN pairwise template matching across 500 images is slow and fails for intra-class variation. We implement **BoVW** (visual codebook clustering + SVM) for robust category identification.
4. **Spatial Object Detection**: For detecting specific object instances, we implement **Lowe's Ratio Test + RANSAC Homography + Bounding Box Projection**.
"""),

    code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split
from sift_pipeline import SIFTBoVWPipeline, SIFTFeatureMatcher

def show_rgb(img, title=None):
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12)
    plt.show()
"""),

    md_cell("""## Step 1: Load Data & Visualize SIFT Keypoints"""),

    code_cell("""train_imgs, train_lbls, classes, _ = load_split("train")
test_imgs, test_lbls, _, _ = load_split("test")

sift = cv2.SIFT_create(nfeatures=500)
sample_img = train_imgs[0]
gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
kps, descs = sift.detectAndCompute(gray, None)

img_kps = cv2.drawKeypoints(sample_img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_rgb(img_kps, title=f"SIFT Rich Keypoints (Detected: {len(kps)})")
"""),

    md_cell("""## Step 2: Method A - Bag of Visual Words (BoVW) for Object Identification
1. Extract SIFT descriptors across the training set.
2. Cluster descriptors using KMeans into a visual codebook (e.g. 150 visual words).
3. Build normalized visual word frequency histograms.
4. Train an SVM classifier.
"""),

    code_cell("""bovw_pipeline = SIFTBoVWPipeline(nfeatures=400, vocab_size=150)
print("Training SIFT BoVW Pipeline...")
train_info = bovw_pipeline.train(train_imgs, train_lbls, C=1.0)
print(f"Trained visual codebook and SVM in {train_info['train_time_sec']:.2f} seconds!")
"""),

    md_cell("""## Step 3: Evaluate SIFT BoVW on Test Set"""),

    code_cell("""eval_res = bovw_pipeline.evaluate(test_imgs, test_lbls)
print(f"SIFT BoVW Test Accuracy: {eval_res['accuracy'] * 100:.2f}%")
print(f"Average Inference Latency: {eval_res['avg_time_per_image_ms']:.2f} ms/image\n")
print(classification_report(test_lbls, eval_res['predictions'], target_names=bovw_pipeline.classes_))

# Plot confusion matrix
cm = np.array(eval_res["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Purples")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("SIFT BoVW Confusion Matrix")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if i != j else "black")
plt.colorbar(im)
plt.show()
"""),

    md_cell("""## Step 4: Method B - Pairwise Feature Matching & Homography Object Detection
When detecting a specific template or tracking an object across frames:
1. Match keypoints with `cv2.BFMatcher(cv2.NORM_L2)`.
2. Apply Lowe's ratio test (`m.distance < 0.75 * n.distance`).
3. Compute geometric transformation using RANSAC Homography (`cv2.findHomography`).
4. Project bounding box corners (`cv2.perspectiveTransform`) to localize the object!
"""),

    code_cell("""matcher = SIFTFeatureMatcher(nfeatures=500, ratio_thresh=0.75)

# Test matching between two consecutive video frames from train/people
f0 = cv2.imread("../data/train/people/02022_0954.MOV_Bielsko-Biała_Pride_frame_000.jpg")
f1 = cv2.imread("../data/train/people/02022_0954.MOV_Bielsko-Biała_Pride_frame_001.jpg")

loc_res = matcher.localize_object(f0, f1, min_inliers=10)
print(f"Matches: {loc_res['num_matches']} | Verified Inliers: {loc_res['num_inliers']} | Object Detected: {loc_res['detected']}")

# Draw matched keypoints and projected bounding box
if loc_res['detected']:
    scene_annotated = f1.copy()
    pts = np.int32(loc_res['dst_corners']).reshape(-1, 1, 2)
    cv2.polylines(scene_annotated, [pts], True, (0, 255, 0), 4, cv2.LINE_AA)
    
    matches_img = cv2.drawMatches(
        f0, loc_res['kp_template'],
        scene_annotated, loc_res['kp_scene'],
        loc_res['good_matches'][:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    show_rgb(matches_img, title=f"SIFT Keypoint Matching & Object Localization ({loc_res['num_inliers']} Inliers)")
"""),

    md_cell("""## Step 5: Save Model"""),

    code_cell("""bovw_pipeline.save_model("sift_bovw_model.pkl")
sz = Path("sift_bovw_model.pkl").stat().st_size / (1024 * 1024)
print(f"Saved SIFT Model to sift_bovw_model.pkl ({sz:.2f} MB)")
""")
]


# ==============================================================================
# Notebook 3: FREAK
# ==============================================================================
freak_cells = [
    md_cell("""# 3. FREAK (Fast Retina Keypoint) Pipeline
### Bio-Inspired Binary Descriptor for Object Identification & Detection

### What is FREAK?
**FREAK** (*Fast Retina Keypoint*) was introduced by Alahi et al. (CVPR 2012). It is a fast, robust **binary** descriptor inspired by the human visual system:
- **Retinal Sampling Grid**: Mimics the distribution of photoreceptors in the human retina (fovea), with a high density of receptive fields near the center and decreasing density toward the periphery.
- **Binary Descriptor**: 64 bytes (512 bits) of intensity difference tests, comparing pairs of retinal receptive fields.
- **Detector-Free Extractor**: FREAK **only computes descriptors**; it requires a separate detector (such as **FAST** or **BRISK**) to identify keypoints.
- **Hamming Distance**: Matching uses bitwise XOR (`cv2.NORM_HAMMING`), executing orders of magnitude faster than floating-point L2 norms!
"""),

    code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split
from freak_pipeline import FREAKBoVWPipeline, FREAKFeatureMatcher

def show_rgb(img, title=None):
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12)
    plt.show()
"""),

    md_cell("""## Step 1: Detect Keypoints with FAST and Compute FREAK Descriptors"""),

    code_cell("""train_imgs, train_lbls, classes, _ = load_split("train")
test_imgs, test_lbls, _, _ = load_split("test")

# FAST detector + FREAK descriptor
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
freak = cv2.xfeatures2d.FREAK_create()

sample_img = train_imgs[10]
gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
kps = fast.detect(gray, None)
kps, descs = freak.compute(gray, kps)

print(f"FAST Detected: {len(kps)} keypoints")
print(f"FREAK Descriptors Shape: {descs.shape} (dtype: {descs.dtype})")
print(f"Descriptor bit length: {descs.shape[1] * 8} bits (64 bytes)")

img_kps = cv2.drawKeypoints(sample_img, kps, None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_rgb(img_kps, title="FAST Keypoints + FREAK Bio-Inspired Retinal Descriptors")
"""),

    md_cell("""## Step 2: Method A - FREAK Bag of Visual Words (BoVW) for Object Identification"""),

    code_cell("""freak_bovw = FREAKBoVWPipeline(detector_type="fast", max_keypoints=300, vocab_size=150)
print("Training FREAK BoVW Pipeline...")
info = freak_bovw.train(train_imgs, train_lbls, C=1.0)
print(f"Trained in {info['train_time_sec']:.2f} seconds!")
"""),

    md_cell("""## Step 3: Evaluate FREAK BoVW on Full Test Set"""),

    code_cell("""eval_res = freak_bovw.evaluate(test_imgs, test_lbls)
print(f"FREAK BoVW Test Accuracy: {eval_res['accuracy'] * 100:.2f}%")
print(f"Average Inference Latency: {eval_res['avg_time_per_image_ms']:.2f} ms/image\n")
print(classification_report(test_lbls, eval_res['predictions'], target_names=freak_bovw.classes_))

# Plot confusion matrix
cm = np.array(eval_res["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Oranges")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("FREAK BoVW Confusion Matrix")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="blue" if i != j else "black")
plt.colorbar(im)
plt.show()
"""),

    md_cell("""## Step 4: Method B - Binary Feature Matching (cv2.NORM_HAMMING) & Homography Detection"""),

    code_cell("""freak_matcher = FREAKFeatureMatcher(detector_type="fast", max_keypoints=500, ratio_thresh=0.80)

f0 = cv2.imread("../data/train/people/02022_0954.MOV_Bielsko-Biała_Pride_frame_000.jpg")
f1 = cv2.imread("../data/train/people/02022_0954.MOV_Bielsko-Biała_Pride_frame_001.jpg")

loc_res = freak_matcher.localize_object(f0, f1, min_inliers=10)
print(f"Hamming Matches: {loc_res['num_matches']} | Verified Inliers: {loc_res['num_inliers']} | Object Detected: {loc_res['detected']}")

if loc_res['detected']:
    scene_annotated = f1.copy()
    pts = np.int32(loc_res['dst_corners']).reshape(-1, 1, 2)
    cv2.polylines(scene_annotated, [pts], True, (255, 128, 0), 4, cv2.LINE_AA)

    matches_img = cv2.drawMatches(
        f0, loc_res['kp_template'],
        scene_annotated, loc_res['kp_scene'],
        loc_res['good_matches'][:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    show_rgb(matches_img, title=f"FREAK Hamming Matching & Object Localization ({loc_res['num_inliers']} Inliers)")
"""),

    md_cell("""## Step 5: Save FREAK Model"""),

    code_cell("""freak_bovw.save_model("freak_bovw_model.pkl")
sz = Path("freak_bovw_model.pkl").stat().st_size / (1024 * 1024)
print(f"Saved FREAK Model: freak_bovw_model.pkl ({sz:.2f} MB)")
""")
]


# ==============================================================================
# Notebook 4: Unified Comparison and Detection
# ==============================================================================
comp_cells = [
    md_cell("""# 4. Method Comparison & Object Detection Benchmark
### Comprehensive Evaluation of HOG vs SIFT vs FREAK

This notebook presents a direct, head-to-head comparison between the three OpenCV feature methods:
1. **HOG (Histogram of Oriented Gradients)**: Dense gradient orientation representation.
2. **SIFT (Scale-Invariant Feature Transform)**: Sparse multi-scale floating-point keypoint descriptors.
3. **FREAK (Fast Retina Keypoint)**: Bio-inspired binary foveal descriptor.
"""),

    code_cell("""import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark results generated by evaluate_all.py
summary_path = Path("benchmark_summary.json")
if not summary_path.exists():
    from evaluate_all import run_benchmark
    run_benchmark()

with open(summary_path, "r") as f:
    results = json.load(f)

methods = list(results.keys())
accuracies = [results[m]["Accuracy (%)"] for m in methods]
latencies = [results[m]["Latency (ms/img)"] for m in methods]
fps_vals = [results[m]["Throughput (FPS)"] for m in methods]
sizes = [results[m]["Model Size (MB)"] for m in methods]

print("=" * 80)
print(f"{'Method':<8} | {'Accuracy':<12} | {'Latency (ms)':<14} | {'FPS':<10} | {'Model Size':<10}")
print("-" * 80)
for m in methods:
    print(f"{m:<8} | {results[m]['Accuracy (%)']:>10.2f}% | {results[m]['Latency (ms/img)']:>12.2f} ms | {results[m]['Throughput (FPS)']:>10.1f} | {results[m]['Model Size (MB)']:>8.2f} MB")
print("=" * 80)
"""),

    md_cell("""## Visual Comparison Charts"""),

    code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Accuracy
colors = ['#2b5c8f', '#6c4b8b', '#d95f02']
axes[0].bar(methods, accuracies, color=colors)
axes[0].set_title("Test Accuracy (%)", fontsize=13)
axes[0].set_ylim(0, 100)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

# 2. Latency / Inference Time
axes[1].bar(methods, latencies, color=colors)
axes[1].set_title("Inference Latency (ms / image)", fontsize=13)
for i, v in enumerate(latencies):
    axes[1].text(i, v + 0.5, f"{v:.2f} ms", ha='center', fontweight='bold')

# 3. Model Size
axes[2].bar(methods, sizes, color=colors)
axes[2].set_title("Model File Footprint (MB)", fontsize=13)
for i, v in enumerate(sizes):
    axes[2].text(i, v + 1, f"{v:.2f} MB", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
"""),

    md_cell("""## Visualizing Object Detection Demos (SIFT vs FREAK)"""),

    code_cell("""import cv2

sift_demo = Path("visualizations/sift_detection_demo.jpg")
freak_demo = Path("visualizations/freak_detection_demo.jpg")

if sift_demo.exists() and freak_demo.exists():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    im_s = cv2.cvtColor(cv2.imread(str(sift_demo)), cv2.COLOR_BGR2RGB)
    im_f = cv2.cvtColor(cv2.imread(str(freak_demo)), cv2.COLOR_BGR2RGB)

    axes[0].imshow(im_s)
    axes[0].set_title("SIFT Object Detection & Homography Bounding Box", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(im_f)
    axes[1].set_title("FREAK Binary Object Detection & Homography Bounding Box", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
"""),

    md_cell("""## Architectural Summary & Decision Matrix

| Dimension | HOG | SIFT | FREAK |
| :--- | :--- | :--- | :--- |
| **Descriptor Type** | Dense gradient orientations (8,100 floats) | Sparse keypoints (128 floats / point) | Sparse keypoints (64 bytes / point) |
| **Keypoint Detector** | None (dense grid) | Built-in DoG (Difference of Gaussians) | External (FAST, BRISK, AGAST) |
| **Distance Metric** | L2 / Linear Hyperplane | L2 Norm | Hamming Distance (bitwise XOR) |
| **Classification Accuracy**| **84.29%** (Best on global shape) | 70.00% (BoVW) | 71.43% (BoVW) |
| **Inference Speed** | **0.73 ms** (~1,370 FPS) | 35.34 ms (~28 FPS) | **2.93 ms** (~340 FPS, 12x faster than SIFT!) |
| **Model Size** | 39.38 MB | 0.21 MB | **0.17 MB** |
| **Invariance** | Illumination, local edge gradients | Scale, Rotation, Illumination | Scale, Rotation, Illumination |
| **Best Use Case** | Whole-image category classification & pedestrian/rigid object sliding window | High-precision instance recognition under severe scale/rotation | Real-time / mobile / embedded object detection & SLAM tracking |
""")
]

if __name__ == "__main__":
    out_dir = Path("/home/user/Puente/vision/project")
    create_notebook(hog_cells, out_dir / "1_hog.ipynb")
    create_notebook(sift_cells, out_dir / "2_sift.ipynb")
    create_notebook(freak_cells, out_dir / "3_freak.ipynb")
    create_notebook(comp_cells, out_dir / "4_comparison_and_detection.ipynb")
