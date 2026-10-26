"""
HOG (Histogram of Oriented Gradients) Classification and Object Detection Pipeline.
Includes:
- Feature extraction using cv2.HOGDescriptor
- Classifier training using cv2.ml.SVM
- Model saving/loading with compact footprint
- Full test evaluation (accuracy, precision, recall, F1, confusion matrix)
- Multiscale sliding-window object detection with Non-Maximum Suppression (NMS)
"""

import os
import json
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset import load_split, find_data_dir


def non_max_suppression_fast(boxes: np.ndarray, overlap_thresh: float = 0.3) -> List[int]:
    """
    Standard fast vectorized Non-Maximum Suppression (NMS).
    boxes: array of shape (N, 5): [x1, y1, x2, y2, score]
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    pick = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    return pick


class HOGPipeline:
    def __init__(
        self,
        win_size: Tuple[int, int] = (128, 128),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        cell_size: Tuple[int, int] = (8, 8),
        nbins: int = 9
    ):
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins

        self.hog = cv2.HOGDescriptor(
            self.win_size,
            self.block_size,
            self.block_stride,
            self.cell_size,
            self.nbins
        )
        self.svm: Optional[cv2.ml.SVM] = None
        self.label_encoder = LabelEncoder()
        self.classes_: List[str] = []

    def compute_descriptor(self, img: np.ndarray) -> np.ndarray:
        """Resize image to win_size and compute normalized 1D HOG feature vector."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if gray.shape != (self.win_size[1], self.win_size[0]):
            gray = cv2.resize(gray, self.win_size)

        des = self.hog.compute(gray)
        return des.flatten().astype(np.float32)

    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract HOG descriptors for a list of images."""
        features = [self.compute_descriptor(img) for img in images]
        return np.array(features, dtype=np.float32)

    def train(
        self,
        train_images: List[np.ndarray],
        train_labels: List[str],
        kernel_type: str = "linear",
        C: float = 1.0,
        gamma: float = 0.5
    ) -> Dict[str, Any]:
        """
        Train an OpenCV SVM classifier on HOG descriptors.
        kernel_type: 'linear' or 'rbf'
        """
        t0 = time.time()
        X_train = self.extract_features(train_images)
        y_train_enc = self.label_encoder.fit_transform(train_labels).astype(np.int32)
        self.classes_ = list(self.label_encoder.classes_)

        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)

        if kernel_type == "rbf":
            self.svm.setKernel(cv2.ml.SVM_RBF)
            self.svm.setGamma(gamma)
        else:
            self.svm.setKernel(cv2.ml.SVM_LINEAR)

        self.svm.setC(C)
        self.svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train_enc.reshape(-1, 1))
        train_time = time.time() - t0

        return {
            "num_samples": len(train_images),
            "feature_dim": X_train.shape[1],
            "train_time_sec": train_time,
            "classes": self.classes_
        }

    def predict(self, images: List[np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """Predict class labels and raw response values for a batch of images."""
        if self.svm is None:
            raise ValueError("Model is not trained yet.")
        X = self.extract_features(images)
        _, responses = self.svm.predict(X)
        pred_indices = responses.flatten().astype(int)
        pred_labels = self.label_encoder.inverse_transform(pred_indices).tolist()
        return pred_labels, responses

    def evaluate(
        self,
        test_images: List[np.ndarray],
        test_labels: List[str]
    ) -> Dict[str, Any]:
        """Run full evaluation on test set and return metrics."""
        t0 = time.time()
        pred_labels, _ = self.predict(test_images)
        eval_time = time.time() - t0

        acc = float(accuracy_score(test_labels, pred_labels))
        report = classification_report(test_labels, pred_labels, target_names=self.classes_, output_dict=True)
        cm = confusion_matrix(test_labels, pred_labels, labels=self.classes_).tolist()

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "eval_time_sec": eval_time,
            "avg_time_per_image_ms": (eval_time / len(test_images)) * 1000 if test_images else 0.0,
            "predictions": pred_labels
        }

    def save_model(self, filepath: str):
        """Save SVM model and metadata (classes and config)."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.svm.save(str(p))

        meta_path = p.with_suffix(".json")
        meta = {
            "win_size": self.win_size,
            "block_size": self.block_size,
            "block_stride": self.block_stride,
            "cell_size": self.cell_size,
            "nbins": self.nbins,
            "classes": self.classes_
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_model(self, filepath: str):
        """Load SVM model and metadata."""
        p = Path(filepath)
        self.svm = cv2.ml.SVM_load(str(p))

        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.win_size = tuple(meta["win_size"])
            self.block_size = tuple(meta["block_size"])
            self.block_stride = tuple(meta["block_stride"])
            self.cell_size = tuple(meta["cell_size"])
            self.nbins = meta["nbins"]
            self.classes_ = meta["classes"]
            self.label_encoder.fit(self.classes_)
            self.hog = cv2.HOGDescriptor(
                self.win_size,
                self.block_size,
                self.block_stride,
                self.cell_size,
                self.nbins
            )

    def detect_multiscale(
        self,
        image: np.ndarray,
        target_class: Optional[str] = None,
        scale_factor: float = 1.25,
        step_size: int = 16,
        min_win: Tuple[int, int] = (64, 64),
        max_win: Optional[Tuple[int, int]] = None,
        nms_overlap: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Multiscale sliding window object detector.
        Scans pyramids and sliding windows, classifies each window with SVM,
        and applies Non-Maximum Suppression to localize detected objects.
        """
        if self.svm is None:
            raise ValueError("Model is not trained yet.")

        h_orig, w_orig = image.shape[:2]
        if max_win is None:
            max_win = (min(w_orig, 400), min(h_orig, 400))

        detected_boxes = []

        curr_w, curr_h = min_win
        while curr_w <= max_win[0] and curr_h <= max_win[1]:
            # Step through image
            for y in range(0, h_orig - curr_h + 1, step_size):
                for x in range(0, w_orig - curr_w + 1, step_size):
                    window = image[y:y + curr_h, x:x + curr_w]
                    patch_feat = self.compute_descriptor(window).reshape(1, -1)
                    # OpenCV SVM predict: flag 1 returns signed decision value / margin
                    res = self.svm.predict(patch_feat, flags=cv2.ml.StatModel_RAW_OUTPUT)[1]
                    pred_idx = int(self.svm.predict(patch_feat)[1][0, 0])
                    class_name = self.classes_[pred_idx]

                    if target_class is None or class_name == target_class:
                        # Use negative raw margin or decision distance as score
                        score = float(np.abs(res[0, 0]))
                        detected_boxes.append([x, y, x + curr_w, y + curr_h, score, class_name])

            curr_w = int(curr_w * scale_factor)
            curr_h = int(curr_h * scale_factor)

        if not detected_boxes:
            return []

        # Filter boxes per class with NMS
        final_detections = []
        for cls in set([b[5] for b in detected_boxes]):
            cls_boxes = np.array([[b[0], b[1], b[2], b[3], b[4]] for b in detected_boxes if b[5] == cls])
            keep_indices = non_max_suppression_fast(cls_boxes, overlap_thresh=nms_overlap)
            for idx in keep_indices:
                b = cls_boxes[idx]
                final_detections.append({
                    "box": (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                    "score": float(b[4]),
                    "class": cls
                })

        return final_detections


if __name__ == "__main__":
    print("Loading data for HOG...")
    train_imgs, train_lbls, classes, _ = load_split("train")
    test_imgs, test_lbls, _, _ = load_split("test")

    pipeline = HOGPipeline(win_size=(128, 128))
    print(f"Training HOG Linear SVM on {len(train_imgs)} samples...")
    train_info = pipeline.train(train_imgs, train_lbls, kernel_type="linear", C=1.0)
    print(f"Trained in {train_info['train_time_sec']:.2f}s, feature dim: {train_info['feature_dim']}")

    print("Evaluating HOG on test set...")
    eval_res = pipeline.evaluate(test_imgs, test_lbls)
    print(f"HOG Test Accuracy: {eval_res['accuracy'] * 100:.2f}%")
    print(f"Average Inference Time: {eval_res['avg_time_per_image_ms']:.2f} ms/image")

    model_path = Path("hog_svm_model.yml")
    pipeline.save_model(str(model_path))
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Saved model to {model_path} (Size: {model_size_mb:.2f} MB)")
