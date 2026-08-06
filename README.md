# Fracture Detection — Fixed Pipeline

## What changed vs. your original script

1. **Real ground-truth labels, not self-generated ones.** Your original
   `labels_train` was computed from the same hand-picked thresholds used
   elsewhere in the script — the classifier was learning to imitate a rule,
   not learning from real fractures. Now labels come from matching detected
   regions against your actual FracAtlas COCO bounding boxes via IoU.

2. **Train once, not per image.** Your original code called `clf.fit(...)`
   inside `detect_fractures()`, refitting a brand-new RandomForest from a
   handful of regions every single image. Now there's one training run
   (`train_and_tune.py`) that produces a saved model used for all inference.

3. **Proper train/val/test split at the image level**, saved to
   `image_split.json` so the test set is never touched until final evaluation.

4. **Thresholds tuned against real labels**, not hand-picked constants.
   `confidenceThreshold` is now chosen by maximizing F1 on the val split.

5. **New texture features: GLCM + LBP**, added alongside your existing
   Gabor + HOG features, to help catch fractures with weak edges/gradients.

6. **Honest, IoU-based evaluation** (region-level precision/recall/F1 like
   object detection, plus image-level fractured/non-fractured classification
   metrics) instead of just a raw fracture count.

## Files

- `common_pipeline.py` — candidate region detection + feature extraction
  (classical CV, deterministic) + COCO loading + IoU utilities. No training
  or classification happens here.
- `train_and_tune.py` — run **once**. Splits images, builds real labels,
  trains RandomForest with `GridSearchCV`, tunes confidence threshold on
  val, saves `fracture_classifier.joblib` + `tuned_threshold.json`.
- `evaluate_and_report.py` — run **after** training. Loads the frozen model,
  evaluates on the untouched test split, prints/saves final metrics.

## Before running

Edit the top of `train_and_tune.py`:

```python
CONFIG = {
    "imagesFolder_fractured":    "FracAtlas/images/Fractured",
    "imagesFolder_nonfractured": "FracAtlas/images/Non_fractured",
    "cocoJsonPath":               "FracAtlas/Annotations/COCO JSON/<your_actual_filename>.json",
    ...
}
```

Open your `Annotations/COCO JSON` folder and put the exact filename in —
I don't have your dataset locally so I couldn't confirm it. If the COCO file
turns out to have a slightly different schema (some COCO exports nest things
differently), send me an error message and I'll patch `load_coco_ground_truth()`.

## How to run

```bash
pip install scikit-learn scikit-image opencv-python scipy pandas joblib tqdm --break-system-packages

python train_and_tune.py         # trains + tunes, ~1x pass over train+val images
python evaluate_and_report.py    # final honest metrics on held-out test set
```

Watch the console output of `train_and_tune.py` for the
`candidate_region_recall` warning — if it's low, it means the classical CV
step itself isn't proposing regions for many real fractures, which caps your
final recall no matter how good the classifier is. That's worth reporting
honestly in your writeup as a known limitation (and ties directly into the
"classical CV vs deep learning" framing we discussed — this is exactly the
kind of ceiling a learned detector like YOLO doesn't have).

## No GPU needed

Nothing here runs on GPU — it's all CPU (skimage, opencv, sklearn). Both
`train_and_tune.py` and `evaluate_and_report.py` now parallelize candidate
detection across CPU cores automatically (`NUM_WORKERS = cpu_count() - 1`),
so more cores = proportionally faster. On a 2-core machine that's 1 worker;
on 4-core, 3 workers.

## Recommended before the full run

1. Test on a small subset (~50-100 images per class) first — confirms your
   `cocoJsonPath` and folder paths are correct before committing hours to
   the full ~4,014 image run.
2. Watch console output for `candidate_region_recall` — a low value means
   the classical CV step itself isn't proposing regions for many real
   fractures, which caps final recall regardless of classifier quality.
   Worth reporting honestly in your writeup if so.
3. Progress prints every 200 images (`train_and_tune.py`) / 100 images
   (`evaluate_and_report.py`) so you can gauge how far along a run is
   without babysitting it.
