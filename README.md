# Automated Bone Fracture Detection & Localization

A YOLOv8-based object detection pipeline that localizes fractures in X-ray images, built on the [FracAtlas](https://github.com/XLR8-07/FracAtlas) dataset. Rather than just classifying "fracture / no fracture," this model draws a bounding box around the fracture location.

## Overview

- **Task:** Fracture localization (object detection), not just classification
- **Model:** YOLOv8n, fine-tuned from COCO-pretrained weights
- **Dataset:** FracAtlas official detection split (717 annotated X-rays: 574 train / 82 val / 63 test)
- **Result:** 0.517 ± 0.041 mAP@0.5, 0.706 ± 0.058 precision, 0.449 ± 0.052 recall (mean ± std across 3 independently trained seeds)
- **Inference speed:** ~12ms per image (~82 FPS) on a Tesla T4 GPU

## Why localization, not just classification

Most public fracture-detection projects (including an earlier version of this one, built with MATLAB + Random Forest + handcrafted features) treat this as a binary classification problem: fracture present or not. That's useful, but it doesn't tell you *where* the fracture is. This version reframes the task as object detection, producing a bounding box a clinician could actually use.

## Methodology

### Dataset preparation
- Used FracAtlas's **official train/valid/test split** (not a random split), so results are comparable to published benchmarks on this dataset
- Only 717 of FracAtlas's ~4,000 images have bounding-box annotations (the rest are classification-only), so the detection task is trained on that smaller, correctly-annotated subset

### Class imbalance handling
- FracAtlas's detection split is naturally balanced 1:1 between fracture-positive and (boosted) negative images
- Added ~3,300 additional true-negative background X-rays (disjoint from val/test, to prevent leakage) to teach the model what a clean X-ray looks like and reduce false positives
- Applied 3x oversampling of fracture-positive images during training

### Small/hairline fracture mitigation
Over 90% of annotated fracture boxes occupy less than 2% of the image area. To preserve signal for these small objects:
- Trained at 1024px input resolution (up from YOLO's default 640px)
- Used `copy-paste` augmentation, which pastes fracture regions onto other training images, directly increasing exposure to small fracture patterns

### Statistical rigor: multi-seed validation
Every reported number is a **mean ± standard deviation across 3 independently trained models** (different random seeds, identical hyperparameters and data), not a single run. This distinguishes genuine performance differences from ordinary training-run noise — an early single-run comparison suggested CLAHE preprocessing *helped*; the 3-seed comparison showed the opposite (see below).

### CLAHE ablation
Ran a controlled ablation: trained the exact same model configuration on raw images vs. CLAHE + Non-local Means denoised images, with identical training composition and hyperparameters — isolating preprocessing as the only variable.

| Metric | Baseline | CLAHE + Denoising | Verdict |
|---|---|---|---|
| mAP@0.5 | 0.517 ± 0.041 | 0.457 ± 0.006 | **Real difference** — baseline is better |
| mAP@0.5:0.95 | 0.218 ± 0.020 | 0.180 ± 0.008 | **Real difference** — baseline is better |
| Precision | 0.706 ± 0.058 | 0.597 ± 0.066 | Within noise |
| Recall | 0.449 ± 0.052 | 0.423 ± 0.029 | Within noise |

**Finding:** CLAHE preprocessing did not help, and modestly hurt, detection performance for YOLOv8 on this dataset — contradicting the intuition carried over from the classical CV (Random Forest) version of this project, where CLAHE + denoising was a meaningful contributor. This suggests CNN-based detectors can already learn useful contrast-invariant features on their own, and manual contrast enhancement may introduce artifacts (denoising, edge softening) that work against a model already handling that internally.

## Results

**Best model:** Baseline (no CLAHE), selected by mean mAP@0.5 across seeds.

- **mAP@0.5:** 0.517 ± 0.041
- **mAP@0.5:0.95:** 0.218 ± 0.020
- **Precision:** 0.706 ± 0.058
- **Recall:** 0.449 ± 0.052
- **Inference latency:** 12.2ms ± 1.4ms (82.3 FPS) on Tesla T4

### Error analysis

At a fixed confidence threshold (0.25) and IoU match threshold (0.5) on the test set:
- **34 true positives, 20 false positives, 35 false negatives**
- **91% of missed detections (32 of 35) were small/hairline fractures** — the model's errors are concentrated almost entirely on the hardest, smallest fracture cases, not spread evenly across all fracture sizes

## Known limitations

- **Recall (~45%) is the main bottleneck.** The model misses roughly half of all fractures on the test set. Given that 91% of misses are small/hairline cases, this points to **limited annotated training data** (only 574 training images) as the primary constraint — not model architecture or preprocessing.
- **Small test set (63 images, 69 boxes).** Metrics have real variance at this scale; the multi-seed approach partially addresses this by showing the *direction* of effects is consistent, but absolute numbers should be read with that caveat.
- **Single-class detection.** The model detects "fracture present" as one class; it does not distinguish fracture types (e.g., hairline vs. displaced).

## Next steps

- More annotated training data is the highest-leverage improvement available — recall did not respond meaningfully to resolution/augmentation changes, suggesting the ceiling is data volume, not training technique
- Explore test-time augmentation (multi-scale inference) as a lower-cost way to improve recall without more labeled data
- Try a larger backbone (`yolov8s`/`yolov8m`) now that the pipeline and evaluation methodology are validated

## Tech stack

Python, PyTorch, Ultralytics YOLOv8, OpenCV, pandas, NumPy, Google Colab (T4 GPU)
