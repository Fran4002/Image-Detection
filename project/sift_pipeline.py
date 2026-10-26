"""
SIFT (Scale-Invariant Feature Transform) Object Identification and Detection Pipeline.

Includes:
1. SIFT feature detection and 128-D descriptor extraction (cv2.SIFT_create).
2. Bag of Visual Words (BoVW) for general category identification / classification:
   - Visual vocabulary clustering (MiniBatchKMeans).
   - Image visual word histogram extraction & L2 normalization.
   - Classifier training (SVM).
   - Full evaluation on test set (accuracy, precision, recall, confusion matrix).
3. Pairwise Feature Matching & Geometric Object Localization:
   - BFMatcher / FLANN with Lowe's ratio test.
   - Homography estimation with RANSAC (cv2.findHomography).
   - Bounding box projection (cv2.perspectiveTransform) for spatial object detection.
   - Exemplar-based multi-class object identification via inlier voting.
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split, find_data_dir


class SIFTBoVWPipeline:
    """
    Category-level Object Identification using SIFT and Bag of Visual Words (BoVW).
    """
    def __init__(
        self,
        nfeatures: int = 400,
        vocab_size: int = 150,
        random_state: int = 42
    ):
        self.nfeatures = nfeatures
        self.vocab_size = vocab_size
        self.random_state = random_state

        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.classifier: Optional[LinearSVC] = None
        self.label_encoder = LabelEncoder()
        self.classes_: List[str] = []

    def extract_sift_descriptors(
        self,
        images: List[np.ndarray],
        max_dim: int = 640
    ) -> List[Optional[np.ndarray]]:
        """Extract SIFT descriptors for a list of images (optionally downscaled for efficiency)."""
        descriptors_list = []
        for img in images:
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                img_scaled = img

            gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY) if len(img_scaled.shape) == 3 else img_scaled
            _, des = self.sift.detectAndCompute(gray, None)
            descriptors_list.append(des if des is not None else np.empty((0, 128), dtype=np.float32))
        return descriptors_list

    def build_vocabulary(
        self,
        train_descriptors: List[np.ndarray],
        max_descriptors_per_img: int = 60
    ) -> MiniBatchKMeans:
        """Sample descriptors across training set and fit KMeans visual vocabulary."""
        sampled = []
        for des in train_descriptors:
            if des is not None and len(des) > 0:
                n = len(des)
                step = max(1, n // max_descriptors_per_img)
                sampled.append(des[::step])

        if not sampled:
            raise ValueError("No descriptors found to build visual vocabulary.")

        descriptors_pool = np.vstack(sampled).astype(np.float32)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            random_state=self.random_state,
            batch_size=512,
            n_init=3
        )
        self.kmeans.fit(descriptors_pool)
        return self.kmeans

    def compute_bovw_histogram(self, des: Optional[np.ndarray]) -> np.ndarray:
        """Convert an image's SIFT descriptors into a normalized visual word histogram."""
        hist = np.zeros(self.vocab_size, dtype=np.float32)
        if des is not None and len(des) > 0 and self.kmeans is not None:
            words = self.kmeans.predict(des.astype(np.float32))
            np.add.at(hist, words, 1.0)
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
        return hist

    def train(
        self,
        train_images: List[np.ndarray],
        train_labels: List[str],
        C: float = 1.0
    ) -> Dict[str, Any]:
        """Train SIFT BoVW classifier."""
        t0 = time.time()
        train_descs = self.extract_sift_descriptors(train_images)
        self.build_vocabulary(train_descs)

        X_train = np.array([self.compute_bovw_histogram(d) for d in train_descs], dtype=np.float32)
        self.label_encoder.fit(train_labels)
        self.classes_ = list(self.label_encoder.classes_)
        y_train = self.label_encoder.transform(train_labels)

        self.classifier = LinearSVC(C=C, random_state=self.random_state, max_iter=2000)
        self.classifier.fit(X_train, y_train)
        train_time = time.time() - t0

        return {
            "num_samples": len(train_images),
            "vocab_size": self.vocab_size,
            "train_time_sec": train_time,
            "classes": self.classes_
        }

    def predict(self, images: List[np.ndarray]) -> List[str]:
        """Predict class labels for a list of images."""
        if self.classifier is None or self.kmeans is None:
            raise ValueError("Model is not trained yet.")
        descs = self.extract_sift_descriptors(images)
        X = np.array([self.compute_bovw_histogram(d) for d in descs], dtype=np.float32)
        pred_indices = self.classifier.predict(X)
        return list(self.label_encoder.inverse_transform(pred_indices))

    def evaluate(
        self,
        test_images: List[np.ndarray],
        test_labels: List[str]
    ) -> Dict[str, Any]:
        """Evaluate SIFT BoVW model on test set."""
        t0 = time.time()
        preds = self.predict(test_images)
        eval_time = time.time() - t0

        acc = float(accuracy_score(test_labels, preds))
        report = classification_report(test_labels, preds, target_names=self.classes_, output_dict=True)
        cm = confusion_matrix(test_labels, preds, labels=self.classes_).tolist()

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "eval_time_sec": eval_time,
            "avg_time_per_image_ms": (eval_time / len(test_images)) * 1000 if test_images else 0.0,
            "predictions": preds
        }

    def save_model(self, filepath: str):
        """Save BoVW model (KMeans, LinearSVC, LabelEncoder)."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "nfeatures": self.nfeatures,
                "vocab_size": self.vocab_size,
                "kmeans": self.kmeans,
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "classes": self.classes_
            }, f)

    def load_model(self, filepath: str):
        """Load BoVW model."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.nfeatures = data["nfeatures"]
        self.vocab_size = data["vocab_size"]
        self.kmeans = data["kmeans"]
        self.classifier = data["classifier"]
        self.label_encoder = data["label_encoder"]
        self.classes_ = data["classes"]
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)


class SIFTFeatureMatcher:
    """
    SIFT Feature Matching and Geometric Object Detection (Homography & Bounding Box).
    """
    def __init__(self, nfeatures: int = 1000, ratio_thresh: float = 0.75):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.ratio_thresh = ratio_thresh
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def detect_and_compute(self, img: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return self.sift.detectAndCompute(gray, None)

    def match_features(
        self,
        des1: Optional[np.ndarray],
        des2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        """Compute KNN matches and filter using Lowe's ratio test."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []

        knn_matches = self.bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
        return good_matches

    def localize_object(
        self,
        template_img: np.ndarray,
        scene_img: np.ndarray,
        min_inliers: int = 8
    ) -> Dict[str, Any]:
        """
        Match template image to scene image, find homography with RANSAC,
        and project bounding box corners onto the scene.
        """
        kp1, des1 = self.detect_and_compute(template_img)
        kp2, des2 = self.detect_and_compute(scene_img)

        good_matches = self.match_features(des1, des2)

        if len(good_matches) < 4:
            return {
                "detected": False,
                "num_matches": len(good_matches),
                "num_inliers": 0,
                "homography": None,
                "dst_corners": None,
                "good_matches": good_matches,
                "kp_template": kp1,
                "kp_scene": kp2
            }

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return {
                "detected": False,
                "num_matches": len(good_matches),
                "num_inliers": 0,
                "homography": None,
                "dst_corners": None,
                "good_matches": good_matches,
                "kp_template": kp1,
                "kp_scene": kp2
            }

        inliers = int(mask.ravel().sum()) if mask is not None else 0
        h_t, w_t = template_img.shape[:2]
        pts = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(pts, H)

        detected = inliers >= min_inliers

        return {
            "detected": detected,
            "num_matches": len(good_matches),
            "num_inliers": inliers,
            "homography": H,
            "dst_corners": dst_corners,
            "good_matches": good_matches,
            "inlier_mask": mask,
            "kp_template": kp1,
            "kp_scene": kp2
        }


if __name__ == "__main__":
    print("Loading data for SIFT BoVW...")
    train_imgs, train_lbls, classes, _ = load_split("train")
    test_imgs, test_lbls, _, _ = load_split("test")

    pipeline = SIFTBoVWPipeline(nfeatures=400, vocab_size=150)
    print("Training SIFT BoVW pipeline...")
    info = pipeline.train(train_imgs, train_lbls)
    print(f"Trained in {info['train_time_sec']:.2f}s, Vocab size: {info['vocab_size']}")

    print("Evaluating SIFT BoVW on test set...")
    eval_res = pipeline.evaluate(test_imgs, test_lbls)
    print(f"SIFT BoVW Test Accuracy: {eval_res['accuracy'] * 100:.2f}%")
    print(f"Average Inference Time: {eval_res['avg_time_per_image_ms']:.2f} ms/image")

    model_path = Path("sift_bovw_model.pkl")
    pipeline.save_model(str(model_path))
    print(f"Saved SIFT model to {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
