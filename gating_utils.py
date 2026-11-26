"""
Gating and verification utilities for urine microscopy object detection and classification.

This module provides post-classification morphological and biological verification (gating)
for candidate objects to filter out false positives and misclassifications:
- CO_DI: Square shape OR star mark (envelope pattern), verified with height, width, area, and aspect ratio.
- RBC: Circular shape (convex hull circularity) AND greater than minimum height, width, and area.
- small_sperm / Sperm: Distinct head (compact/rounded blob) and tail (slender flagellum).
- WBC: Circular shape, strictly rejecting degenerated cells (ruptured membranes, low solidity, ragged lysis borders).
- Yeast: 2 or more round structures (budding yeast, blastoconidia pairs, clusters, or chains).
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List


GATED_CLASSES = {
    "CO_DI",
    "RBC",
    "unstained_rbc",
    "WBC",
    "Yeast",
    "small_sperm",
    "Sperm",
    "sperm",
}


def is_gated_class(class_name: Optional[str]) -> bool:
    """Returns True if the class has gating verification rules."""
    if not class_name:
        return False
    return class_name in GATED_CLASSES or class_name.lower() in {"sperm", "small_sperm", "co_di", "rbc", "unstained_rbc", "wbc", "yeast"}


def extract_foreground_crop(img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Extracts the non-zero region from a patch (stripping border padding if present)
    and estimates the local slide background intensity.
    """
    if img is None or img.size == 0:
        return None, 0.0

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Check for zero/black letterbox padding
    mask = gray > 10
    if not np.any(mask):
        return None, 0.0

    y_id, x_id = np.where(mask)
    ymin, ymax = int(y_id.min()), int(y_id.max())
    xmin, xmax = int(x_id.min()), int(x_id.max())

    # If padded, crop to the active non-zero region
    if ymin > 0 or xmin > 0 or ymax < gray.shape[0] - 1 or xmax < gray.shape[1] - 1:
        crop = gray[ymin:ymax + 1, xmin:xmax + 1]
    else:
        crop = gray

    # Border median intensity as background estimate
    border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
    bg = float(np.median(border))
    return crop, bg


def get_center_contour(contours: List[np.ndarray], shape: Tuple[int, int], min_area: float = 15.0) -> Optional[np.ndarray]:
    """
    Selects the primary contour closest to or enclosing the crop center.
    """
    if not contours:
        return None
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        valid = contours

    inside = [c for c in valid if cv2.pointPolygonTest(c, (cx, cy), False) >= 0]
    if inside:
        return max(inside, key=cv2.contourArea)

    def dist_to_center(c):
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx_c = M["m10"] / M["m00"]
            cy_c = M["m01"] / M["m00"]
        else:
            bx, by, bw, bh = cv2.boundingRect(c)
            cx_c = bx + bw / 2.0
            cy_c = by + bh / 2.0
        return (cx_c - cx) ** 2 + (cy_c - cy) ** 2

    return min(valid, key=dist_to_center)


def verify_codi(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Gating logic for CO_DI (Calcium Oxalate Dihydrate):
    - Metadata check: aspect ratio square-like (0.60 - 1.60), area >= 250, w >= 15, h >= 15.
    - Morphology check: Square shape (side ratio >= 0.60) OR star mark / envelope pattern (diagonal gradient energy >= 0.30).
    """
    metrics: Dict[str, Any] = {}
    if meta is not None:
        ar = float(meta.get("aspect_ratio", 1.0))
        area = float(meta.get("area", 1000.0))
        w = float(meta.get("raw_w", meta.get("w", 35.0)))
        h = float(meta.get("raw_h", meta.get("h", 35.0)))
        metrics["meta_ar"] = ar
        metrics["meta_area"] = area
        metrics["meta_w"] = w
        metrics["meta_h"] = h

        if not (0.60 <= ar <= 1.60 and area >= 250 and w >= 15 and h >= 15):
            return False, "meta_bounds_rejected", metrics

    diff = np.abs(crop.astype(float) - bg)
    _, th = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "no_contours_found", metrics

    center_c = get_center_contour(cnts, crop.shape, min_area=20)
    if center_c is None:
        return False, "no_center_contour", metrics

    # 1. Square shape check via minimum area bounding rectangle
    rect = cv2.minAreaRect(center_c)
    rw, rh = rect[1]
    sq_ratio = min(rw, rh) / (max(rw, rh) + 1e-5)
    metrics["square_ratio"] = round(sq_ratio, 3)

    # 2. Star mark / envelope 'X' pattern check via diagonal gradient energy
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angles = np.arctan2(gy, gx) * 180 / np.pi % 180
    diag_mask = ((angles >= 25) & (angles <= 65)) | ((angles >= 115) & (angles <= 155))
    diag_energy = float(np.sum(mag[diag_mask]) / (np.sum(mag) + 1e-5))
    metrics["diagonal_energy"] = round(diag_energy, 3)

    is_square = sq_ratio >= 0.60
    has_star_mark = diag_energy >= 0.30

    if is_square or has_star_mark:
        status = "verified_square" if is_square else "verified_star_mark"
        return True, status, metrics

    return False, "not_square_no_star", metrics


def verify_rbc(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None,
               min_w: float = 18.0, min_h: float = 18.0, min_area: float = 400.0,
               min_hull_circ: float = 0.75, min_ar: float = 0.65, max_ar: float = 1.55) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Gating logic for RBC (Red Blood Cells):
    - Size check: greater than minimum width, height, or area (filters debris specks).
    - Aspect ratio: circular range (0.65 - 1.55).
    - Circular shape: Convex hull circularity >= 0.75.
    """
    metrics: Dict[str, Any] = {}
    if meta is not None:
        ar = float(meta.get("aspect_ratio", 1.0))
        area = float(meta.get("area", 1000.0))
        w = float(meta.get("raw_w", meta.get("w", 35.0)))
        h = float(meta.get("raw_h", meta.get("h", 35.0)))
        metrics["meta_ar"] = ar
        metrics["meta_area"] = area
        metrics["meta_w"] = w
        metrics["meta_h"] = h

        if w < min_w or h < min_h or area < min_area:
            return False, "size_too_small", metrics
        if ar < min_ar or ar > max_ar:
            return False, "aspect_ratio_out_of_bounds", metrics

    diff = np.abs(crop.astype(float) - bg)
    _, th = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.erode(th, kernel, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "no_contours_found", metrics

    center_c = get_center_contour(cnts, crop.shape, min_area=15)
    if center_c is None:
        return False, "no_center_contour", metrics

    hull = cv2.convexHull(center_c)
    h_area = cv2.contourArea(hull)
    h_peri = cv2.arcLength(hull, True)
    h_circ = (4 * np.pi * h_area) / (h_peri**2) if h_peri > 0 else 0.0
    h_circ = min(h_circ, 1.0)
    metrics["hull_circularity"] = round(h_circ, 3)

    if h_circ >= min_hull_circ:
        return True, "verified_circular", metrics

    return False, "non_circular", metrics


def verify_sperm(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Gating logic for small_sperm / Sperm:
    - Multi-scale / adaptive segmentation isolating compact head and slender flagellum tail.
    - Head detection: morphological opening with disk kernel retains round/oval head blob (area >= 20 px²).
    - Tail detection: residual slender structure with arc length >= 8 px or elongated body aspect ratio >= 2.2.
    """
    metrics: Dict[str, Any] = {}
    diff = np.abs(crop.astype(float) - bg)
    otsu_val, _ = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th_head = (diff >= max(12, otsu_val * 0.75)).astype(np.uint8) * 255
    th_all = (diff >= max(8, otsu_val * 0.45)).astype(np.uint8) * 255

    # Head isolation via morphological opening
    head_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    head_mask = cv2.morphologyEx(th_head, cv2.MORPH_OPEN, head_kernel)

    # Tail isolation in residual thin region
    head_dilated = cv2.dilate(head_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    tail_mask = cv2.subtract(th_all, head_dilated)

    head_cnts, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tail_cnts, _ = cv2.findContours(tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_heads = [c for c in head_cnts if cv2.contourArea(c) >= 20]
    valid_tails = [c for c in tail_cnts if cv2.arcLength(c, False) >= 8 or cv2.contourArea(c) >= 15]

    has_head = len(valid_heads) > 0
    has_tail = len(valid_tails) > 0
    metrics["has_head"] = has_head
    metrics["has_tail"] = has_tail

    if has_head and has_tail:
        return True, "verified_head_and_tail", metrics

    # Fallback: single continuous elongated sperm flagellum body
    cnts_all, _ = cv2.findContours(th_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_all:
        main_c = max(cnts_all, key=cv2.contourArea)
        if cv2.contourArea(main_c) >= 100:
            rect = cv2.minAreaRect(main_c)
            rw, rh = rect[1]
            aspect = max(rw, rh) / (min(rw, rh) + 1e-5)
            metrics["elongation_aspect"] = round(aspect, 2)
            if aspect >= 2.2:
                return True, "verified_elongated_sperm", metrics

    return False, "no_head_or_tail", metrics


def verify_wbc(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None,
               min_hull_circ: float = 0.75, min_solidity: float = 0.75, min_circ: float = 0.35) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Gating logic for WBC (White Blood Cells):
    - Must be circular in shape: hull_circularity >= 0.75, aspect ratio 0.60 - 1.65.
    - Degenerated cells are strictly rejected:
      - Low solidity (< 0.75) indicates ruptured / vacuolated / notched membranes.
      - Low raw circularity (< 0.35) indicates ragged, degenerated, fuzzy cell borders.
    - Size criteria: larger than typical RBC (area >= 500 px², w >= 20 px, h >= 20 px).
    """
    metrics: Dict[str, Any] = {}
    if meta is not None:
        ar = float(meta.get("aspect_ratio", 1.0))
        area = float(meta.get("area", 1500.0))
        w = float(meta.get("raw_w", meta.get("w", 45.0)))
        h = float(meta.get("raw_h", meta.get("h", 45.0)))
        metrics["meta_ar"] = ar
        metrics["meta_area"] = area
        metrics["meta_w"] = w
        metrics["meta_h"] = h

        if not (0.60 <= ar <= 1.65 and area >= 500 and w >= 20 and h >= 20):
            return False, "degenerated_or_meta_bounds", metrics

    diff = np.abs(crop.astype(float) - bg)
    _, th = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Hole filling for cellular interior
    ch, cw = crop.shape
    th_fill = th.copy()
    mf = np.zeros((ch + 2, cw + 2), np.uint8)
    cv2.floodFill(th_fill, mf, (0, 0), 255)
    filled = th | cv2.bitwise_not(th_fill)

    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "no_contours_found", metrics

    center_c = get_center_contour(cnts, crop.shape, min_area=30)
    if center_c is None:
        return False, "no_center_contour", metrics

    area = cv2.contourArea(center_c)
    peri = cv2.arcLength(center_c, True)
    raw_circ = (4 * np.pi * area) / (peri**2) if peri > 0 else 0.0

    hull = cv2.convexHull(center_c)
    h_area = cv2.contourArea(hull)
    h_peri = cv2.arcLength(hull, True)
    h_circ = (4 * np.pi * h_area) / (h_peri**2) if h_peri > 0 else 0.0
    h_circ = min(h_circ, 1.0)

    solidity = area / (h_area + 1e-5)
    metrics["hull_circularity"] = round(h_circ, 3)
    metrics["raw_circularity"] = round(raw_circ, 3)
    metrics["solidity"] = round(solidity, 3)

    if h_circ < min_hull_circ:
        return False, "degenerated_non_circular", metrics
    if solidity < min_solidity:
        return False, "degenerated_low_solidity", metrics
    if raw_circ < min_circ:
        return False, "degenerated_ragged_boundary", metrics

    return True, "verified_intact_wbc", metrics


def verify_yeast(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Gating logic for Yeast:
    - Must possess 2 or more round structures (budding yeast pairs, clusters, or chains).
    - Solitary single round cells (e.g. unbudded yeast, solitary RBCs, round debris) are strictly rejected.
    - Verified via:
      1. Multiple distinct round contours (len(round_cells) >= 2):
         - Each cell contour must have cellular size (area >= 25 px²), high convex hull circularity (>= 0.68),
           solidity (>= 0.72), and bounded aspect ratio (<= 1.85).
         - At least one primary cell (area >= 50 px²), and the pair must be in close proximity (centroid distance <= 50 px).
      2. Conjoined budding yeast (snowman / dumbbell / figure-8 mother + bud):
         - Must be elongated (oriented minAreaRect aspect ratio >= 1.25) to avoid false-triggering on solitary round cells.
         - Verified constriction neck between lobes via:
           a) Morphological neck erosion (k in [3, 5, 7, 9]): splits body into >= 2 round lobes (each lobe area >= 12 px², circularity >= 0.65).
           b) Euclidean Distance Transform dual peaks: >= 2 distinct peak components (inscribed radius >= 3.5 px) separated by >= 8 px,
              with a prominent saddle drop (valley <= 0.82 * min(R1, R2)) along the line connecting the peaks.
    """
    metrics: Dict[str, Any] = {}
    if crop is None or crop.shape[0] < 8 or crop.shape[1] < 8:
        return False, "crop_too_small", metrics

    if meta is not None:
        metrics["meta_ar"] = float(meta.get("aspect_ratio", 1.0))
        metrics["meta_area"] = float(meta.get("area", 0.0))

    diff = np.abs(crop.astype(float) - bg)
    _, th = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 1. Distinct discrete round structures (separated or lightly touching cells)
    cnts_raw, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    round_cells = []
    centroids = []
    for c in cnts_raw:
        area = cv2.contourArea(c)
        if area >= 25:
            hull = cv2.convexHull(c)
            h_area = cv2.contourArea(hull)
            h_peri = cv2.arcLength(hull, True)
            h_circ = (4 * np.pi * h_area) / (h_peri**2) if h_peri > 0 else 0.0
            solidity = area / (h_area + 1e-5)
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            ar = max(rw, rh) / (min(rw, rh) + 1e-5)
            if h_circ >= 0.68 and solidity >= 0.72 and ar <= 1.85:
                round_cells.append(c)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

    if len(round_cells) >= 2:
        areas = [cv2.contourArea(c) for c in round_cells]
        if max(areas) >= 50:
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    d = np.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                    if d <= 50.0:
                        metrics["round_cells"] = len(round_cells)
                        metrics["min_cell_dist"] = round(d, 1)
                        return True, "multiple_discrete_round_cells", metrics

    # 2. Conjoined budding yeast (snowman / dumbbell / figure-8)
    if cnts_raw:
        c = max(cnts_raw, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area >= 60:
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            body_ar = max(rw, rh) / (min(rw, rh) + 1e-5)
            metrics["body_aspect_ratio"] = round(body_ar, 2)

            if body_ar >= 1.25:
                # 2A. Morphological neck erosion into >= 2 round lobes
                for k in [3, 5, 7, 9]:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    eroded = cv2.erode(th, kernel, iterations=1)
                    sub_cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_lobes = []
                    for sc in sub_cnts:
                        s_area = cv2.contourArea(sc)
                        if s_area >= 12:
                            s_hull = cv2.convexHull(sc)
                            s_h_area = cv2.contourArea(s_hull)
                            s_h_peri = cv2.arcLength(s_hull, True)
                            s_circ = (4 * np.pi * s_h_area) / (s_h_peri**2) if s_h_peri > 0 else 0.0
                            if s_circ >= 0.65:
                                valid_lobes.append(sc)
                    if len(valid_lobes) >= 2:
                        metrics["eroded_lobes"] = len(valid_lobes)
                        return True, "budding_eroded_round_lobes", metrics

                # 2B. Euclidean Distance Transform Dual Peaks with Saddle Drop
                dist = cv2.distanceTransform(th, cv2.DIST_L2, 5)
                for min_r in [3.5, 4.0, 4.5]:
                    peak_mask = (dist >= min_r).astype(np.uint8)
                    num_p_labels, p_labels, p_stats, p_centroids = cv2.connectedComponentsWithStats(peak_mask)
                    if num_p_labels >= 3:
                        pts = p_centroids[1:]
                        p_dists = [dist[int(pt[1]), int(pt[0])] for pt in pts]
                        max_idx1 = np.argmax(p_dists)
                        r1 = p_dists[max_idx1]
                        pt1 = pts[max_idx1]
                        for idx, (pt, r) in enumerate(zip(pts, p_dists)):
                            if idx == max_idx1:
                                continue
                            d = np.hypot(pt[0] - pt1[0], pt[1] - pt1[1])
                            if d >= 8.0 and r >= min_r:
                                num_samples = max(5, int(d))
                                xs = np.clip(np.linspace(pt1[0], pt[0], num_samples).astype(int), 0, crop.shape[1] - 1)
                                ys = np.clip(np.linspace(pt1[1], pt[1], num_samples).astype(int), 0, crop.shape[0] - 1)
                                profile = dist[ys, xs]
                                saddle_min = np.min(profile)
                                min_peak_r = min(r1, r)
                                if saddle_min <= 0.82 * min_peak_r:
                                    metrics["saddle_ratio"] = round(saddle_min / (min_peak_r + 1e-5), 2)
                                    metrics["peak_sep"] = round(d, 1)
                                    return True, "budding_distance_dual_peaks", metrics

    return False, "solitary_cell_or_no_bud", metrics


def verify_gated_object(
    patch_or_crop: Optional[np.ndarray],
    predicted_class: str,
    meta: Optional[Dict[str, Any]] = None,
    rejection_class: str = "Debris"
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    Main gating verification entrypoint.

    Args:
        patch_or_crop: Patch image (80x80 / 224x224) or tight crop as a numpy array.
        predicted_class: Class name produced by detector or classifier.
        meta: Optional metadata dict with keys like 'raw_w', 'raw_h', 'area', 'aspect_ratio'.
        rejection_class: Class to reassign if verification fails (default: 'Debris').

    Returns:
        is_verified (bool): True if verified, False if rejected by gating.
        final_class (str): predicted_class if verified, rejection_class if rejected.
        reason (str): Short description of why the decision was made.
        metrics (dict): Numeric features extracted during verification.
    """
    # Non-gated classes pass through untouched
    if not is_gated_class(predicted_class):
        return True, predicted_class, "class_not_gated", {}

    if patch_or_crop is None or patch_or_crop.size == 0:
        return False, rejection_class, "empty_patch_or_crop", {}

    crop, bg = extract_foreground_crop(patch_or_crop)
    if crop is None or crop.size == 0:
        return False, rejection_class, "empty_foreground", {}

    cls_lower = predicted_class.lower()

    if cls_lower == "co_di":
        ok, reason, metrics = verify_codi(crop, bg, meta)
    elif cls_lower in ("rbc", "unstained_rbc"):
        ok, reason, metrics = verify_rbc(crop, bg, meta)
    elif cls_lower in ("small_sperm", "sperm"):
        ok, reason, metrics = verify_sperm(crop, bg, meta)
    elif cls_lower == "wbc":
        ok, reason, metrics = verify_wbc(crop, bg, meta)
    elif cls_lower == "yeast":
        ok, reason, metrics = verify_yeast(crop, bg, meta)
    else:
        return True, predicted_class, "class_not_gated", {}

    final_class = predicted_class if ok else rejection_class
    return ok, final_class, reason, metrics
