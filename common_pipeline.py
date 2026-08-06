"""
common_pipeline.py
===================
Shared building blocks used by both train_and_tune.py and evaluate_and_report.py.

This module ONLY does candidate-region detection + feature extraction.
It does NOT classify anything and it does NOT train anything -- that
separation is the whole point of the fix (see chat discussion): detection
of "candidate regions" is deterministic classical CV, classification of
"is this candidate actually a fracture" is a model that must be trained
ONCE on labeled data and reused, not re-fit per image.
"""

import os
import json
import warnings
import numpy as np
import cv2

from pathlib import Path
from skimage import io, color, exposure, filters, transform
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.filters import threshold_local, threshold_otsu
from skimage.morphology import (binary_opening, binary_closing, binary_dilation,
                                 remove_small_objects, skeletonize, disk)
from skimage.measure import label, regionprops
from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
from scipy.ndimage import binary_fill_holes

warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------
# GROUND TRUTH LOADING (COCO JSON)
# ----------------------------------------------------------------------

def load_coco_ground_truth(coco_json_path: str) -> dict:
    """
    Parse a COCO-format annotation file into:
        { image_filename: [ [x0, y0, x1, y1], ... ] }

    COCO stores boxes as [x, y, width, height] with (x, y) = top-left corner.
    We convert to [x0, y0, x1, y1] (top-left, bottom-right) to match the
    format used everywhere else in this project (skimage regionprops bbox
    convention is (min_row, min_col, max_row, max_col) = (y0, x0, y1, x1),
    so be careful with axis order when you compute IoU below -- we keep
    everything in (x0, y0, x1, y1) = (col0, row0, col1, row1) here and
    convert explicitly at the IoU call site).
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    gt = {fname: [] for fname in id_to_filename.values()}
    for ann in coco.get("annotations", []):
        fname = id_to_filename.get(ann["image_id"])
        if fname is None:
            continue
        x, y, w, h = ann["bbox"]
        gt[fname].append([x, y, x + w, y + h])  # -> [x0, y0, x1, y1]

    return gt


def iou_xyxy(box_a, box_b) -> float:
    """IoU between two boxes in [x0, y0, x1, y1] format."""
    xa0, ya0, xa1, ya1 = box_a
    xb0, yb0, xb1, yb1 = box_b

    inter_x0 = max(xa0, xb0)
    inter_y0 = max(ya0, yb0)
    inter_x1 = min(xa1, xb1)
    inter_y1 = min(ya1, yb1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa1 - xa0) * max(0.0, ya1 - ya0)
    area_b = max(0.0, xb1 - xb0) * max(0.0, yb1 - yb0)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def region_bbox_to_xyxy(skimage_bbox):
    """
    Convert skimage regionprops .bbox (min_row, min_col, max_row, max_col)
    to (x0, y0, x1, y1) = (min_col, min_row, max_col, max_row).
    """
    min_row, min_col, max_row, max_col = skimage_bbox
    return [min_col, min_row, max_col, max_row]


# ----------------------------------------------------------------------
# CANDIDATE REGION DETECTION (classical CV -- unchanged logic from your
# original script, just with classification stripped out)
# ----------------------------------------------------------------------

def detect_candidate_regions(image_path: str, config: dict) -> dict:
    """
    Runs preprocessing + segmentation + edge detection + region proposal.
    Returns candidate regions and their features, but does NOT decide
    which ones are fractures -- that's the trained classifier's job,
    done separately in train_and_tune.py / evaluate_and_report.py.
    """
    out = {
        "image_name": os.path.basename(image_path),
        "status": "ok",
        "rows": None, "cols": None,
        "props": [],           # skimage regionprops objects
        "features": None,      # np.array [n_regions, n_features]
        "feature_names": None,
        "labeled": None,       # labeled region image (for NMS / overlay later)
    }

    original_image = io.imread(image_path)
    if original_image is None or original_image.size == 0:
        out["status"] = "load_error"
        return out

    if original_image.ndim == 3:
        gray_uint8 = (color.rgb2gray(original_image) * 255).astype(np.uint8)
    else:
        gray_uint8 = original_image.astype(np.uint8)

    max_dim = config.get("maxImageDim", 0)
    scale = 1.0
    if max_dim and max(gray_uint8.shape) > max_dim:
        scale = max_dim / max(gray_uint8.shape)
        new_h = int(gray_uint8.shape[0] * scale)
        new_w = int(gray_uint8.shape[1] * scale)
        gray_uint8 = cv2.resize(gray_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out["scale"] = scale  # IMPORTANT: needed to rescale GT boxes to match the resized image
    rows, cols = gray_uint8.shape
    out["rows"], out["cols"] = rows, cols

    # CLAHE
    enhanced_image = exposure.equalize_adapthist(
        gray_uint8, clip_limit=0.02, kernel_size=max(8, gray_uint8.shape[0] // 8))
    enhanced_uint8 = (enhanced_image * 255).astype(np.uint8)

    # NLM denoise
    smoothed_uint8 = cv2.fastNlMeansDenoising(
        enhanced_uint8, h=6, searchWindowSize=25, templateWindowSize=7)

    # Bone segmentation
    thresh_local_val = threshold_local(smoothed_uint8, block_size=35, method='gaussian')
    bone_mask = smoothed_uint8 > (thresh_local_val * 0.5)
    bone_mask = binary_opening(bone_mask, disk(2))
    bone_mask = binary_closing(bone_mask, disk(5))
    bone_mask = remove_small_objects(bone_mask, min_size=500)
    filled = binary_fill_holes(bone_mask)
    marrow = remove_small_objects(~filled, min_size=5000)
    bone_mask = filled & ~marrow

    if np.sum(bone_mask) == 0:
        out["status"] = "no_bone"
        return out

    border_mask = np.zeros((rows, cols), dtype=bool)
    bw = 40
    border_mask[:bw, :] = True
    border_mask[-bw:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True
    bone_border_edge = filters.sobel(bone_mask.astype(np.float64)) > 0.05
    bone_border_edge = binary_dilation(bone_border_edge, disk(10))
    border_bone = bone_mask & (border_mask | bone_border_edge)

    smooth_f = smoothed_uint8.astype(np.float64)
    edge_log = filters.laplace(filters.gaussian(smooth_f, sigma=2)) > 0.006
    edge_canny = cv_canny_via_skimage(smooth_f)
    edge_combined = edge_log | edge_canny

    border_smooth = smooth_f.copy()
    border_smooth[~border_bone] = 0
    edge_border = (filters.laplace(filters.gaussian(border_smooth, sigma=2)) > 0.002) & border_bone

    fracture_candidate = (edge_combined & bone_mask) | edge_border
    fracture_candidate = remove_small_objects(fracture_candidate, min_size=15)
    fracture_thinned = skeletonize(fracture_candidate)
    fracture_thinned = remove_small_objects(fracture_thinned, min_size=20)

    from scipy.ndimage import distance_transform_edt
    distance_map = distance_transform_edt(bone_mask)
    thin_bone = distance_map < 3
    dark_threshold = threshold_otsu(smoothed_uint8[bone_mask]) * 0.5
    dark_regions = remove_small_objects((smoothed_uint8 < dark_threshold) & bone_mask, min_size=50)

    fracture_final = fracture_thinned | (thin_bone & edge_combined & bone_mask & dark_regions)
    fracture_final = remove_small_objects(fracture_final, min_size=25)

    labeled = label(fracture_final)
    props = regionprops(labeled, intensity_image=smoothed_uint8)

    if len(props) == 0:
        out["status"] = "no_candidates"
        return out

    features, names = extract_features(props, labeled, smooth_f, smoothed_uint8,
                                        edge_combined, border_bone, rows, cols,
                                        config.get("useAdvancedFeatures", True))

    out["props"] = props
    out["features"] = features
    out["feature_names"] = names
    out["labeled"] = labeled
    return out


def cv_canny_via_skimage(smooth_f):
    from skimage import feature as skfeature
    return skfeature.canny(smooth_f, low_threshold=0.10 * 255, high_threshold=0.20 * 255)


# ----------------------------------------------------------------------
# FEATURE EXTRACTION (original basic + gabor + HOG, PLUS new GLCM + LBP)
# ----------------------------------------------------------------------

def extract_features(props, labeled, smooth_f, smoothed_uint8, edge_combined,
                      border_bone, rows, cols, use_advanced=True):
    basic_list = []
    for p in props:
        area = p.area
        major = p.major_axis_length
        minor = p.minor_axis_length
        aspect = major / (minor + 1e-10)
        linearity = 1 - (minor / (major + 1e-10))
        perim_ar = p.perimeter / (np.sqrt(area) + 1e-10)
        cy, cx = int(p.centroid[0]), int(p.centroid[1])
        is_border = bool(border_bone[cy, cx]) if (0 <= cy < rows and 0 <= cx < cols) else False
        rm = labeled == p.label
        ed = np.sum(edge_combined[rm]) / (area + 1e-10)
        basic_list.append([area, p.eccentricity, aspect, p.solidity, linearity,
                            abs(np.degrees(p.orientation)), perim_ar,
                            p.mean_intensity, float(is_border), ed])
    basic_features = np.array(basic_list)
    basic_names = ["area", "eccentricity", "aspect", "solidity", "linearity",
                   "orientation_deg", "perim_area_ratio", "mean_intensity",
                   "is_border", "edge_density"]

    if not use_advanced:
        return basic_features, basic_names

    margin = 8
    gabor_list, hog_list, glcm_list, lbp_list = [], [], [], []

    for p in props:
        y0, x0, y1, x1 = p.bbox
        y0m, x0m = max(0, y0 - margin), max(0, x0 - margin)
        y1m, x1m = min(rows, y1 + margin), min(cols, x1 + margin)
        crop = smooth_f[y0m:y1m, x0m:x1m]
        local_mask = (labeled[y0m:y1m, x0m:x1m] == p.label)

        # --- Gabor (unchanged from original) ---
        feat = []
        for wl in range(4, 14, 2):
            for angle in [0, 45, 90, 135]:
                real, _ = filters.gabor(crop, frequency=1.0 / wl, theta=np.radians(angle))
                vals = real[local_mask]
                feat.append(float(np.mean(np.abs(vals))) if vals.size else 0.0)
        gabor_list.append(feat)

        # --- HOG (unchanged) ---
        bb = p.bbox
        reg = smoothed_uint8[bb[0]:bb[2], bb[1]:bb[3]]
        if reg.size == 0:
            reg = np.zeros((64, 64), dtype=np.uint8)
        reg64 = transform.resize(reg, (64, 64), anti_aliasing=True)
        hog_list.append(hog(reg64, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), block_norm='L2-Hys'))

        # --- NEW: GLCM texture features ---
        # Captures how gray-levels co-occur spatially -- good at picking up the
        # subtle textural disruption at a fracture line that plain edge
        # detection often misses (e.g. hairline fractures with weak gradients).
        reg_glcm = reg64 if reg64.dtype == np.uint8 else (reg64 * 255).astype(np.uint8)
        reg_glcm = (reg_glcm // 8).astype(np.uint8)  # quantize to 32 levels, standard practice
        glcm = graycomatrix(reg_glcm, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                             levels=32, symmetric=True, normed=True)
        glcm_feat = []
        for prop_name in ("contrast", "homogeneity", "energy", "correlation"):
            glcm_feat.extend(graycoprops(glcm, prop_name).flatten().tolist())
        glcm_list.append(glcm_feat)

        # --- NEW: LBP texture features (histogram of local binary patterns) ---
        # Captures fine local texture irregularity -- complements GLCM,
        # cheap to compute, historically strong for medical-image texture.
        lbp = local_binary_pattern(reg_glcm, P=8, R=1, method="uniform")
        n_bins = 10  # 8 uniform patterns + 2 (non-uniform, background)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        lbp_list.append(hist.tolist())

    gabor_features = np.array(gabor_list)
    hog_features = np.array(hog_list)
    glcm_features = np.array(glcm_list)
    lbp_features = np.array(lbp_list)

    all_features = np.hstack([basic_features, gabor_features, hog_features,
                               glcm_features, lbp_features])

    names = (basic_names
             + [f"gabor_{i}" for i in range(gabor_features.shape[1])]
             + [f"hog_{i}" for i in range(hog_features.shape[1])]
             + [f"glcm_{i}" for i in range(glcm_features.shape[1])]
             + [f"lbp_{i}" for i in range(lbp_features.shape[1])])

    return all_features, names


# ----------------------------------------------------------------------
# NMS (unchanged logic, factored out so both train/eval scripts share it)
# ----------------------------------------------------------------------

def apply_nms(centroids_xy, is_fracture_mask, confidence, nms_distance):
    """
    centroids_xy: array of shape [n_regions, 2] with (x, y) centroid of each region
                  (plain numpy array, NOT skimage region objects -- keeps this
                  function safe to use with results that crossed a process boundary).
    """
    frac_idx = np.where(is_fracture_mask)[0]
    if len(frac_idx) <= 1:
        return is_fracture_mask

    centroids = np.asarray(centroids_xy)[frac_idx]
    keep = np.ones(len(frac_idx), dtype=bool)
    for i in range(len(frac_idx)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(frac_idx)):
            if not keep[j]:
                continue
            if np.linalg.norm(centroids[i] - centroids[j]) < nms_distance:
                if confidence[frac_idx[i]] >= confidence[frac_idx[j]]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    kept_idx = frac_idx[keep]
    result = np.zeros(len(is_fracture_mask), dtype=bool)
    result[kept_idx] = True
    return result
