"""
train_and_tune.py
==================
Trains the fracture/non-fracture classifier ONCE on real ground-truth
labels (matched via IoU against your COCO annotations), tunes the
confidence threshold on a held-out validation split, and saves both
the model and the tuned threshold to disk.

Run this ONCE. After this, evaluate_and_report.py / batch inference
just load the saved model -- no more re-fitting a classifier per image.

Usage:
    python train_and_tune.py

Edit CONFIG below before running.
"""

import json
import joblib
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from common_pipeline import (load_coco_ground_truth, detect_candidate_regions,
                              region_bbox_to_xyxy, iou_xyxy)

NUM_WORKERS = max(1, mp.cpu_count() - 1)  # leave 1 core free for the OS/main process

CONFIG = {
    "imagesFolder_fractured":     "FracAtlas/images/Fractured",
    "imagesFolder_nonfractured":  "FracAtlas/images/Non_fractured",
    "cocoJsonPath":                "FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json",  # <-- fix filename to match yours
    "outputFolder":                "training_output",

    "maxImageDim":                 1024,
    "useAdvancedFeatures":         True,

    "iouMatchThreshold":           0.30,   # candidate region counts as a real fracture if IoU with a GT box >= this
    "trainFrac":                   0.70,
    "valFrac":                     0.15,
    "testFrac":                    0.15,   # test split is NOT touched here -- only in evaluate_and_report.py
    "randomSeed":                  42,
}


def list_all_images(config):
    frac_dir = Path(config["imagesFolder_fractured"])
    nonfrac_dir = Path(config["imagesFolder_nonfractured"])
    frac_imgs = sorted([p for p in frac_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    nonfrac_imgs = sorted([p for p in nonfrac_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    # (path, image_level_label) -- 1 = fractured, 0 = non-fractured
    all_imgs = [(p, 1) for p in frac_imgs] + [(p, 0) for p in nonfrac_imgs]
    return all_imgs


def match_regions_to_ground_truth(det_result, gt_boxes_orig, scale, iou_thresh):
    """
    det_result: output of detect_candidate_regions()
    gt_boxes_orig: list of [x0,y0,x1,y1] in ORIGINAL image coordinates
    scale: the resize scale detect_candidate_regions applied (so we rescale GT boxes to match)

    Returns: region_labels (0/1 array, one per candidate region),
             n_gt_matched (how many distinct GT boxes were hit by >=1 region -- for recall diagnostics)
    """
    props = det_result["props"]
    n = len(props)
    if n == 0:
        return np.array([], dtype=int), 0

    gt_boxes = [[c * scale for c in box] for box in gt_boxes_orig]  # rescale to match resized image

    region_labels = np.zeros(n, dtype=int)
    matched_gt = set()

    for i, p in enumerate(props):
        region_box = region_bbox_to_xyxy(p.bbox)
        best_iou, best_gt_idx = 0.0, -1
        for gi, gbox in enumerate(gt_boxes):
            iou = iou_xyxy(region_box, gbox)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gi
        if best_iou >= iou_thresh:
            region_labels[i] = 1
            matched_gt.add(best_gt_idx)

    return region_labels, len(matched_gt)


def _process_one_image(args):
    """
    Top-level (picklable) worker function -- runs in a separate process.
    Does candidate detection + IoU label matching for ONE image and returns
    a small, easily-serialized result instead of the full det dict.
    """
    img_path, gt_boxes, config = args
    fname = Path(img_path).name
    det = detect_candidate_regions(str(img_path), config)
    if det["status"] != "ok" or det["features"] is None:
        return {"image": fname, "status": det["status"], "features": None, "labels": None, "n_matched": 0}

    labels, n_matched = match_regions_to_ground_truth(det, gt_boxes, det["scale"], config["iouMatchThreshold"])
    return {"image": fname, "status": "ok", "features": det["features"], "labels": labels, "n_matched": n_matched}


def build_training_table(image_list, gt_dict, config, num_workers=None):
    """
    Runs candidate detection on every image in image_list IN PARALLEL across
    CPU cores, matches regions to ground truth, and returns a stacked
    feature matrix + label vector, plus diagnostics (candidate recall =
    fraction of real GT fractures that the classical CV step even proposed
    a region for -- this is the ceiling on what any classifier trained on
    these candidates can achieve).
    """
    num_workers = num_workers or NUM_WORKERS
    X_parts, y_parts = [], []
    total_gt_boxes = 0
    total_gt_matched = 0
    per_image_log = []

    tasks = []
    for img_path, img_level_label in image_list:
        gt_boxes = gt_dict.get(img_path.name, [])
        total_gt_boxes += len(gt_boxes)
        tasks.append((str(img_path), gt_boxes, config))

    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_one_image, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if done % 200 == 0 or done == len(futures):
                print(f"    processed {done}/{len(futures)} images...")

            if result["status"] != "ok":
                per_image_log.append({"image": result["image"], "status": result["status"], "n_regions": 0})
                continue

            total_gt_matched += result["n_matched"]
            X_parts.append(result["features"])
            y_parts.append(result["labels"])
            per_image_log.append({"image": result["image"], "status": "ok",
                                   "n_regions": len(result["labels"]),
                                   "n_positive": int(result["labels"].sum())})

    X = np.vstack(X_parts) if X_parts else np.empty((0, 0))
    y = np.concatenate(y_parts) if y_parts else np.array([])

    candidate_recall = (total_gt_matched / total_gt_boxes) if total_gt_boxes > 0 else float("nan")

    diagnostics = {
        "n_images": len(image_list),
        "n_regions_total": len(y),
        "n_positive_regions": int(y.sum()) if len(y) else 0,
        "total_gt_boxes": total_gt_boxes,
        "gt_boxes_matched_by_any_candidate": total_gt_matched,
        "candidate_region_recall": candidate_recall,  # IMPORTANT diagnostic -- see note printed below
    }

    return X, y, diagnostics, per_image_log


def main():
    out_dir = Path(CONFIG["outputFolder"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COCO ground truth...")
    gt_dict = load_coco_ground_truth(CONFIG["cocoJsonPath"])
    print(f"  Ground truth entries: {len(gt_dict)} images referenced in COCO JSON")

    print("Listing images...")
    all_imgs = list_all_images(CONFIG)
    print(f"  Fractured: {sum(1 for _, l in all_imgs if l == 1)}   "
          f"Non-fractured: {sum(1 for _, l in all_imgs if l == 0)}")

    labels_for_split = [l for _, l in all_imgs]
    train_imgs, temp_imgs = train_test_split(
        all_imgs, train_size=CONFIG["trainFrac"], stratify=labels_for_split,
        random_state=CONFIG["randomSeed"])
    val_frac_of_temp = CONFIG["valFrac"] / (CONFIG["valFrac"] + CONFIG["testFrac"])
    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=val_frac_of_temp,
        stratify=[l for _, l in temp_imgs], random_state=CONFIG["randomSeed"])

    print(f"  Train: {len(train_imgs)}   Val: {len(val_imgs)}   Test: {len(test_imgs)} (test untouched until evaluate_and_report.py)")

    # Save the split so evaluate_and_report.py uses the EXACT same test set (no leakage)
    split_record = {
        "train": [str(p) for p, _ in train_imgs],
        "val":   [str(p) for p, _ in val_imgs],
        "test":  [str(p) for p, _ in test_imgs],
    }
    with open(out_dir / "image_split.json", "w") as f:
        json.dump(split_record, f, indent=2)

    print("\nExtracting candidate regions + features + true labels for TRAIN split...")
    X_train, y_train, diag_train, _ = build_training_table(train_imgs, gt_dict, CONFIG)
    print(f"  {diag_train}")
    if diag_train["candidate_region_recall"] < 0.5:
        print("  ⚠ WARNING: candidate_region_recall is low. This means the classical CV "
              "step itself is proposing regions for fewer than half of real fractures -- "
              "no classifier can recover fractures that were never proposed as candidates. "
              "This is a ceiling on your final recall and is worth reporting honestly in your writeup.")

    print("\nExtracting candidate regions + features + true labels for VAL split...")
    X_val, y_val, diag_val, _ = build_training_table(val_imgs, gt_dict, CONFIG)
    print(f"  {diag_val}")

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training labels are all one class -- check your COCO JSON path/format and iouMatchThreshold.")

    # ---- Train RandomForest ONCE, with cross-validated hyperparameter search ----
    print("\nTraining RandomForest with GridSearchCV (this happens ONCE, not per image)...")
    param_grid = {
        "n_estimators": [150, 300],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"],  # fracture regions are the minority class -- important
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["randomSeed"])
    # IMPORTANT: n_jobs=-1 lives on GridSearchCV only. Also setting n_jobs=-1 on
    # RandomForestClassifier would cause nested parallelism (each CV fold spawning
    # its own set of parallel workers inside an already-parallel outer loop),
    # which multiplies memory usage (each worker needs its own copy of the training
    # data) without actually running faster on a 2-4 core machine. n_jobs=1 here
    # keeps memory bounded and lets GridSearchCV be the only parallelism layer.
    grid = GridSearchCV(RandomForestClassifier(random_state=CONFIG["randomSeed"], n_jobs=1),
                         param_grid, scoring="f1", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1 (train folds): {grid.best_score_:.3f}")

    # ---- Tune confidence threshold on VAL set (never seen during training) ----
    print("\nTuning confidence threshold on VAL split...")
    val_probs = clf.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.30, 0.91, 0.02)
    best_thresh, best_f1 = 0.5, -1
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    val_preds = (val_probs >= best_thresh).astype(int)
    print(f"  Best threshold: {best_thresh:.2f}  (val F1={best_f1:.3f}, "
          f"precision={precision_score(y_val, val_preds, zero_division=0):.3f}, "
          f"recall={recall_score(y_val, val_preds, zero_division=0):.3f})")
    print("\n  Full val classification report:")
    print(classification_report(y_val, val_preds, target_names=["not_fracture", "fracture"]))

    # ---- Save everything ----
    joblib.dump(clf, out_dir / "fracture_classifier.joblib")
    with open(out_dir / "tuned_threshold.json", "w") as f:
        json.dump({"confidenceThreshold": float(best_thresh)}, f, indent=2)

    # Feature importances -- useful for your report (which features actually matter)
    importances = pd.DataFrame({
        "feature": [f"f{i}" for i in range(X_train.shape[1])],
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "feature_importances.csv", index=False)

    print(f"\nSaved model      -> {out_dir / 'fracture_classifier.joblib'}")
    print(f"Saved threshold  -> {out_dir / 'tuned_threshold.json'}")
    print(f"Saved split      -> {out_dir / 'image_split.json'}")
    print(f"Saved importances-> {out_dir / 'feature_importances.csv'}")
    print("\nNext step: run evaluate_and_report.py to get final metrics on the untouched test split.")


if __name__ == "__main__":
    main()
