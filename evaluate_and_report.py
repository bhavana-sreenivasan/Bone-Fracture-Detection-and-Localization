"""
evaluate_and_report.py
=======================
Loads the model trained ONCE by train_and_tune.py and evaluates it on the
TEST split only (never touched during training or threshold tuning).
Produces both region-level (IoU-matched, like object detection mAP-style
metrics) and image-level (fractured / non-fractured, like a classifier
confusion matrix) results -- this is the honest, defensible evaluation
you want in your report/README.

Run AFTER train_and_tune.py.
Usage:
    python evaluate_and_report.py
"""

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from common_pipeline import load_coco_ground_truth, region_bbox_to_xyxy, iou_xyxy, apply_nms
from train_and_tune import CONFIG as TRAIN_CONFIG, match_regions_to_ground_truth
from common_pipeline import detect_candidate_regions

OUT_DIR = Path(TRAIN_CONFIG["outputFolder"])


def main():
    gt_dict = load_coco_ground_truth(TRAIN_CONFIG["cocoJsonPath"])

    with open(OUT_DIR / "image_split.json") as f:
        split = json.load(f)
    test_paths = [Path(p) for p in split["test"]]
    print(f"Evaluating on {len(test_paths)} held-out TEST images (never seen during training/tuning).")

    clf = joblib.load(OUT_DIR / "fracture_classifier.joblib")
    with open(OUT_DIR / "tuned_threshold.json") as f:
        conf_thresh = json.load(f)["confidenceThreshold"]
    print(f"Using tuned confidence threshold: {conf_thresh:.2f}")

    region_y_true, region_y_pred = [], []
    image_rows = []

    for img_path in test_paths:
        fname = img_path.name
        gt_boxes_orig = gt_dict.get(fname, [])
        image_is_fractured_gt = len(gt_boxes_orig) > 0

        det = detect_candidate_regions(str(img_path), TRAIN_CONFIG)
        if det["status"] != "ok" or det["features"] is None:
            # No candidate regions proposed at all.
            image_rows.append({
                "image": fname, "gt_fractured": image_is_fractured_gt,
                "pred_fractured": False, "n_gt_boxes": len(gt_boxes_orig),
                "n_predicted_regions": 0, "mean_iou_of_matches": None,
            })
            continue

        # True region labels (for computing region-level precision/recall against GT)
        true_labels, _ = match_regions_to_ground_truth(
            det, gt_boxes_orig, det["scale"], TRAIN_CONFIG["iouMatchThreshold"])

        # Model prediction
        probs = clf.predict_proba(det["features"])[:, 1]
        pred_labels = (probs >= conf_thresh).astype(int)
        pred_mask = apply_nms(det["props"], pred_labels.astype(bool), probs, nms_distance=30)

        region_y_true.extend(true_labels.tolist())
        region_y_pred.extend(pred_mask.astype(int).tolist())

        # Image-level: did the model predict ANY fracture in this image?
        image_pred_fractured = bool(pred_mask.any())

        # Mean IoU of predicted-positive regions against their best-matching GT box
        ious = []
        if image_is_fractured_gt:
            gt_scaled = [[c * det["scale"] for c in b] for b in gt_boxes_orig]
            for i, is_pos in enumerate(pred_mask):
                if not is_pos:
                    continue
                rbox = region_bbox_to_xyxy(det["props"][i].bbox)
                best = max((iou_xyxy(rbox, g) for g in gt_scaled), default=0.0)
                ious.append(best)

        image_rows.append({
            "image": fname, "gt_fractured": image_is_fractured_gt,
            "pred_fractured": image_pred_fractured, "n_gt_boxes": len(gt_boxes_orig),
            "n_predicted_regions": int(pred_mask.sum()),
            "mean_iou_of_matches": float(np.mean(ious)) if ious else None,
        })

    # ---------------- REGION-LEVEL METRICS ----------------
    region_y_true = np.array(region_y_true)
    region_y_pred = np.array(region_y_pred)
    print("\n" + "=" * 60)
    print("REGION-LEVEL METRICS (candidate region vs IoU-matched ground truth)")
    print("=" * 60)
    if len(region_y_true):
        print(f"  Precision: {precision_score(region_y_true, region_y_pred, zero_division=0):.3f}")
        print(f"  Recall:    {recall_score(region_y_true, region_y_pred, zero_division=0):.3f}")
        print(f"  F1:        {f1_score(region_y_true, region_y_pred, zero_division=0):.3f}")
    else:
        print("  No regions to evaluate.")

    # ---------------- IMAGE-LEVEL METRICS ----------------
    df = pd.DataFrame(image_rows)
    print("\n" + "=" * 60)
    print("IMAGE-LEVEL METRICS (fractured vs non-fractured, whole image)")
    print("=" * 60)
    cm = confusion_matrix(df["gt_fractured"], df["pred_fractured"], labels=[True, False])
    print("  Confusion matrix [rows=actual, cols=predicted], order=[Fractured, Non-fractured]:")
    print(f"    {cm}")
    print(f"  Precision: {precision_score(df['gt_fractured'], df['pred_fractured'], zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(df['gt_fractured'], df['pred_fractured'], zero_division=0):.3f}")
    print(f"  F1:        {f1_score(df['gt_fractured'], df['pred_fractured'], zero_division=0):.3f}")

    mean_iou = df["mean_iou_of_matches"].dropna().mean()
    print(f"\n  Mean IoU of correctly-flagged fracture regions: {mean_iou:.3f}" if not np.isnan(mean_iou)
          else "\n  Mean IoU: N/A (no correct region matches)")

    df.to_csv(OUT_DIR / "test_results_per_image.csv", index=False)
    print(f"\nPer-image results saved -> {OUT_DIR / 'test_results_per_image.csv'}")
    print("Use these numbers directly in your report/README -- they're honest, held-out metrics.")


if __name__ == "__main__":
    main()
