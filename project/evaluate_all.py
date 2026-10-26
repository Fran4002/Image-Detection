"""
Unified Comprehensive Benchmark and Comparative Evaluation Script.
Compares HOG, SIFT, and FREAK on the dataset across:
1. Classification Accuracy, Precision, Recall, Macro/Weighted F1
2. Training Time (seconds)
3. Inference Latency (ms per image) and Throughput (FPS)
4. Model Footprint / Storage Size (MB)
5. Object Detection & Localization Demo (saving annotated visualizations)
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any

os.environ['MPLCONFIGDIR'] = '/tmp'
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_split
from hog_pipeline import HOGPipeline
from sift_pipeline import SIFTBoVWPipeline, SIFTFeatureMatcher
from freak_pipeline import FREAKBoVWPipeline, FREAKFeatureMatcher


def run_benchmark():
    print("=" * 70)
    print("     COMPREHENSIVE BENCHMARK: HOG vs SIFT vs FREAK")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[1/5] Loading train and test datasets...")
    train_imgs, train_lbls, classes, _ = load_split("train")
    test_imgs, test_lbls, _, _ = load_split("test")
    print(f"Discovered Classes: {classes}")
    print(f"Training Set: {len(train_imgs)} images")
    print(f"Testing Set:  {len(test_imgs)} images")

    results = {}

    # 2. Benchmark HOG
    print("\n[2/5] Benchmarking HOG + SVM Pipeline...")
    hog_p = HOGPipeline(win_size=(128, 128))
    t0 = time.time()
    hog_train_info = hog_p.train(train_imgs, train_lbls, kernel_type="linear", C=1.0)
    hog_train_time = time.time() - t0

    hog_eval = hog_p.evaluate(test_imgs, test_lbls)
    hog_model_path = Path("hog_svm_model.yml")
    hog_p.save_model(str(hog_model_path))
    hog_size_mb = hog_model_path.stat().st_size / (1024 * 1024)

    results["HOG"] = {
        "Accuracy (%)": hog_eval["accuracy"] * 100,
        "Macro F1": hog_eval["classification_report"]["macro avg"]["f1-score"],
        "Weighted F1": hog_eval["classification_report"]["weighted avg"]["f1-score"],
        "Train Time (s)": hog_train_time,
        "Latency (ms/img)": hog_eval["avg_time_per_image_ms"],
        "Throughput (FPS)": 1000.0 / hog_eval["avg_time_per_image_ms"] if hog_eval["avg_time_per_image_ms"] > 0 else 0,
        "Model Size (MB)": hog_size_mb,
        "Report": hog_eval["classification_report"]
    }
    print(f"  -> HOG Accuracy: {results['HOG']['Accuracy (%)']:.2f}% | Latency: {results['HOG']['Latency (ms/img)']:.2f} ms")

    # 3. Benchmark SIFT
    print("\n[3/5] Benchmarking SIFT + BoVW Pipeline...")
    sift_p = SIFTBoVWPipeline(nfeatures=400, vocab_size=150)
    t0 = time.time()
    sift_train_info = sift_p.train(train_imgs, train_lbls, C=1.0)
    sift_train_time = time.time() - t0

    sift_eval = sift_p.evaluate(test_imgs, test_lbls)
    sift_model_path = Path("sift_bovw_model.pkl")
    sift_p.save_model(str(sift_model_path))
    sift_size_mb = sift_model_path.stat().st_size / (1024 * 1024)

    results["SIFT"] = {
        "Accuracy (%)": sift_eval["accuracy"] * 100,
        "Macro F1": sift_eval["classification_report"]["macro avg"]["f1-score"],
        "Weighted F1": sift_eval["classification_report"]["weighted avg"]["f1-score"],
        "Train Time (s)": sift_train_time,
        "Latency (ms/img)": sift_eval["avg_time_per_image_ms"],
        "Throughput (FPS)": 1000.0 / sift_eval["avg_time_per_image_ms"] if sift_eval["avg_time_per_image_ms"] > 0 else 0,
        "Model Size (MB)": sift_size_mb,
        "Report": sift_eval["classification_report"]
    }
    print(f"  -> SIFT Accuracy: {results['SIFT']['Accuracy (%)']:.2f}% | Latency: {results['SIFT']['Latency (ms/img)']:.2f} ms")

    # 4. Benchmark FREAK
    print("\n[4/5] Benchmarking FREAK + BoVW Pipeline...")
    freak_p = FREAKBoVWPipeline(detector_type="fast", max_keypoints=300, vocab_size=150)
    t0 = time.time()
    freak_train_info = freak_p.train(train_imgs, train_lbls, C=1.0)
    freak_train_time = time.time() - t0

    freak_eval = freak_p.evaluate(test_imgs, test_lbls)
    freak_model_path = Path("freak_bovw_model.pkl")
    freak_p.save_model(str(freak_model_path))
    freak_size_mb = freak_model_path.stat().st_size / (1024 * 1024)

    results["FREAK"] = {
        "Accuracy (%)": freak_eval["accuracy"] * 100,
        "Macro F1": freak_eval["classification_report"]["macro avg"]["f1-score"],
        "Weighted F1": freak_eval["classification_report"]["weighted avg"]["f1-score"],
        "Train Time (s)": freak_train_time,
        "Latency (ms/img)": freak_eval["avg_time_per_image_ms"],
        "Throughput (FPS)": 1000.0 / freak_eval["avg_time_per_image_ms"] if freak_eval["avg_time_per_image_ms"] > 0 else 0,
        "Model Size (MB)": freak_size_mb,
        "Report": freak_eval["classification_report"]
    }
    print(f"  -> FREAK Accuracy: {results['FREAK']['Accuracy (%)']:.2f}% | Latency: {results['FREAK']['Latency (ms/img)']:.2f} ms")

    # 5. Object Detection & Localization Visual Demo
    print("\n[5/5] Generating Object Detection & Localization Visual Artifacts...")
    vis_dir = Path("visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Demonstration on two consecutive video frames from train/people
    f0_path = Path("../data/train/people/02022_0954.MOV_Bielsko-Biała_Pride_frame_000.jpg")
    f1_path = Path("../data/train/people/02022_0954.MOV_Bielsko-Biała_Pride_frame_001.jpg")
    if f0_path.exists() and f1_path.exists():
        f0 = cv2.imread(str(f0_path))
        f1 = cv2.imread(str(f1_path))

        # SIFT Localization
        s_matcher = SIFTFeatureMatcher(nfeatures=500)
        s_loc = s_matcher.localize_object(f0, f1, min_inliers=10)
        if s_loc["detected"] and s_loc["dst_corners"] is not None:
            vis_s = f1.copy()
            pts = np.int32(s_loc["dst_corners"]).reshape(-1, 1, 2)
            cv2.polylines(vis_s, [pts], True, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(vis_s, f"SIFT Detected (Inliers: {s_loc['num_inliers']})", (pts[0, 0, 0], pts[0, 0, 1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(str(vis_dir / "sift_detection_demo.jpg"), vis_s)

        # FREAK Localization
        f_matcher = FREAKFeatureMatcher(max_keypoints=500)
        f_loc = f_matcher.localize_object(f0, f1, min_inliers=10)
        if f_loc["detected"] and f_loc["dst_corners"] is not None:
            vis_f = f1.copy()
            pts = np.int32(f_loc["dst_corners"]).reshape(-1, 1, 2)
            cv2.polylines(vis_f, [pts], True, (255, 128, 0), 4, cv2.LINE_AA)
            cv2.putText(vis_f, f"FREAK Detected (Inliers: {f_loc['num_inliers']})", (pts[0, 0, 0], pts[0, 0, 1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
            cv2.imwrite(str(vis_dir / "freak_detection_demo.jpg"), vis_f)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<8} | {'Accuracy':<12} | {'Macro F1':<10} | {'Latency (ms)':<14} | {'FPS':<10} | {'Model Size':<10}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<8} | {m['Accuracy (%)']:>10.2f}% | {m['Macro F1']:>10.3f} | {m['Latency (ms/img)']:>12.2f} ms | {m['Throughput (FPS)']:>10.1f} | {m['Model Size (MB)']:>8.2f} MB")
    print("=" * 80)

    # Save summary json
    summary_path = Path("benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Convert numpy types to python natives
        clean_res = {
            k: {
                sub_k: (sub_v if not isinstance(sub_v, (np.float32, np.float64, np.int32, np.int64)) else float(sub_v))
                for sub_k, sub_v in v.items() if sub_k != "Report"
            }
            for k, v in results.items()
        }
        json.dump(clean_res, f, indent=2)
    print(f"\nSaved benchmark results to {summary_path}")


if __name__ == "__main__":
    run_benchmark()
