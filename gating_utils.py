"""
Gating and verification utilities for urine microscopy object detection and classification.

This module provides post-classification morphological and biological verification (gating)
for candidate objects to filter out false positives and misclassifications:
- CO_DI: Square shape OR star mark (envelope pattern), verified with height, width, area, and aspect ratio.
- RBC: Circular shape (convex hull circularity) AND greater than minimum height, width, and area.
- small_sperm / Sperm: Distinct head (compact/rounded blob) and tail (slender flagellum).
- WBC: Circular shape, strictly rejecting degenerated cells (ruptured membranes, low solidity, ragged lysis borders).
- Yeast: 2 or more round structures (budding yeast, blastoconidia pairs, clusters, or chains).
- EP_SQ: Squamous epithelial cell - very large, solid element (box area + side length + element solidity/fill).
- EP_TR: Transitional epithelial cell - larger than WBC/RBC AND round/oval (circularity on a smoothed outline).
- Struvite / small_struvite morphology verifiers are retained below but currently disabled in gating.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List


GATED_CLASSES = {
    "CO_DI",
    "RBC",
    "WBC",
    "Yeast",
    "small_sperm",
    "Sperm",
    "sperm",
    "EP_SQ",
    "EP_TR",
}

_GATED_LOWER = {
    "sperm", "small_sperm", "co_di", "rbc", "wbc", "yeast",
    "ep_sq", "ep_tr",
}



def is_gated_class(class_name: Optional[str]) -> bool:
    """Returns True if the class has gating verification rules."""
    if not class_name:
        return False
    return class_name in GATED_CLASSES or class_name.lower() in _GATED_LOWER


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


def _blob_shape(c: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return area, circularity, solidity, aspect ratio, and enclosing-circle fill."""
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    circularity = min(4.0 * np.pi * area / (perimeter ** 2), 1.0) if perimeter > 0 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(c)))
    solidity = area / (hull_area + 1e-5)
    rw, rh = cv2.minAreaRect(c)[1]
    aspect = max(rw, rh) / (min(rw, rh) + 1e-5)
    (_, _), radius = cv2.minEnclosingCircle(c)
    circle_fill = area / (np.pi * radius * radius + 1e-5)
    return area, circularity, solidity, aspect, circle_fill


def _is_round_blob(c: np.ndarray, min_circ: float, min_solidity: float,
                   max_aspect: float, min_circle_fill: float) -> bool:
    _, circularity, solidity, aspect, circle_fill = _blob_shape(c)
    return (circularity >= min_circ and solidity >= min_solidity
            and aspect <= max_aspect and circle_fill >= min_circle_fill)


def _split_touching_blobs(body_mask: np.ndarray) -> List[np.ndarray]:
    """Split a merged body into candidate lobes using distance-transform seeds and watershed."""
    distance = cv2.distanceTransform(body_mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance < 3.0:
        return []

    for fraction in (0.80, 0.70, 0.60):
        seeds = (distance >= fraction * max_distance).astype(np.uint8)
        count, labels = cv2.connectedComponents(seeds)
        if count - 1 < 2:
            continue

        markers = np.zeros(body_mask.shape, np.int32)
        markers[body_mask == 0] = 1
        markers[labels > 0] = labels[labels > 0] + 1
        watershed_image = cv2.cvtColor(body_mask, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_image, markers)

        lobes = []
        for label in range(2, count + 1):
            component = (markers == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                lobes.append(max(contours, key=cv2.contourArea))
        if len(lobes) >= 2:
            return lobes
    return []


def verify_yeast(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None,
                 min_round: int = 2, min_blob_area: float = 30.0, min_circ: float = 0.80,
                 min_solidity: float = 0.88, max_aspect: float = 1.5,
                 min_circle_fill: float = 0.70) -> Tuple[bool, str, Dict[str, Any]]:
    """Accept Yeast only for multiple round blobs or a merged body split into round lobes."""
    metrics: Dict[str, Any] = {}
    filled = _prep_mask(crop, bg, close_k=3)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_blob_area]
    metrics["distinct_contours"] = len(contours)
    if not contours:
        return False, "no_contours_found", metrics

    # Separate blobs must each have round, compact geometry.
    round_contours = [
        c for c in contours
        if _is_round_blob(c, min_circ, min_solidity, max_aspect, min_circle_fill)
    ]
    metrics["round_structures"] = len(round_contours)
    if len(round_contours) >= min_round:
        return True, "multiple_round_structures", metrics

    # Touching buds must split into round lobes and have a visible waist.
    body = max(contours, key=cv2.contourArea)
    body_mask = np.zeros(filled.shape, np.uint8)
    cv2.drawContours(body_mask, [body], -1, 255, -1)
    lobes = _split_touching_blobs(body_mask)
    metrics["lobes"] = len(lobes)
    if len(lobes) >= max(2, min_round):
        _, _, body_solidity, _, _ = _blob_shape(body)
        metrics["body_solidity"] = round(body_solidity, 3)
        lobe_shapes = [_blob_shape(c) for c in lobes]
        if (body_solidity < 0.95
                and all(area >= 12 and circularity >= min_circ - 0.05
                        and aspect <= max_aspect and circle_fill >= min_circle_fill - 0.10
                        for area, circularity, _, aspect, circle_fill in lobe_shapes)):
            return True, "budding_round_lobes", metrics

    _, body_circularity, body_solidity, body_aspect, body_circle_fill = _blob_shape(body)
    metrics.update({
        "body_circularity": round(body_circularity, 3),
        "body_solidity": round(body_solidity, 3),
        "body_aspect": round(body_aspect, 2),
        "body_circle_fill": round(body_circle_fill, 3),
    })
    return False, ("single_round_structure" if len(round_contours) == 1 else "not_round"), metrics


def _prep_mask(crop: np.ndarray, bg: float, smooth_sigma: float = 0.0, close_k: int = 5) -> np.ndarray:
    """
    Shared segmentation for the large-element classes (EP_SQ / EP_TR / Struvite):
    trims the blended 2 px rim left by patch resizing, pads with background so objects touching the
    crop edge still close, Otsu-thresholds |crop - bg|, closes gaps and fills holes (pale interior).
    """
    t = 2
    if crop.shape[0] > 4 * t + 8 and crop.shape[1] > 4 * t + 8:
        crop = crop[t:-t, t:-t]
    crop = cv2.copyMakeBorder(crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=int(bg))
    diff = cv2.GaussianBlur(np.abs(crop.astype(np.float32) - bg), (0, 0), 1.0)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    if int(diff_u8.max()) < 12:                      # no contrast against background
        return np.zeros(crop.shape[:2], np.uint8)
    _, th = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))

    # Hole filling for cellular / crystal interior
    ch, cw = th.shape
    th_fill = th.copy()
    mf = np.zeros((ch + 2, cw + 2), np.uint8)
    cv2.floodFill(th_fill, mf, (0, 0), 255)
    filled = th | cv2.bitwise_not(th_fill)

    if smooth_sigma > 0:                             # fair perimeter for circularity
        sm = cv2.GaussianBlur(filled, (0, 0), smooth_sigma)
        filled = ((sm > 127) * 255).astype(np.uint8)
    return filled


def _meta_dims(meta: Optional[Dict[str, Any]], metrics: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Reads (ar, area, w, h) from meta (original-image pixels) and records them in metrics."""
    if meta is None:
        return None
    ar = float(meta.get("aspect_ratio", 1.0))
    area = float(meta.get("area", 1000.0))
    w = float(meta.get("raw_w", meta.get("w", 35.0)))
    h = float(meta.get("raw_h", meta.get("h", 35.0)))
    metrics["meta_ar"] = ar
    metrics["meta_area"] = area
    metrics["meta_w"] = w
    metrics["meta_h"] = h
    return ar, area, w, h


def verify_ep_sq(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None,
                 min_area: float = 3000.0, min_side: float = 35.0,
                 min_solidity: float = 0.55, min_fill: float = 0.30) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Gating logic for EP_SQ (Squamous epithelial cell):
    - Size check (main rule): box area >= 3000 px² and shortest side >= 35 px (squamous cells are far
      bigger than WBC / RBC / yeast / bacteria / debris; rejects thin strands).
    - Element check: the largest hole-filled element must be solid (area / hull area >= 0.55,
      rejects stringy / fragmented clumps) and cover >= 30% of the crop (rejects a speck in a big box).
    """
    metrics: Dict[str, Any] = {}
    dims = _meta_dims(meta, metrics)
    if dims is not None:
        _, area, w, h = dims
        if area < min_area:
            return False, "too_small", metrics
        if min(w, h) < min_side:
            return False, "too_thin", metrics

    filled = _prep_mask(crop, bg)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "no_contours_found", metrics

    c = max(cnts, key=cv2.contourArea)
    el_area = cv2.contourArea(c)
    h_area = cv2.contourArea(cv2.convexHull(c))
    solidity = el_area / (h_area + 1e-5)
    fill = el_area / float(max(crop.shape[0] * crop.shape[1], 1))
    metrics["element_area"] = round(float(el_area), 1)
    metrics["solidity"] = round(float(solidity), 3)
    metrics["fill_fraction"] = round(float(fill), 3)

    if dims is None and el_area < min_area:
        return False, "too_small", metrics
    if fill < min_fill:
        return False, "element_too_sparse", metrics
    if solidity < min_solidity:
        return False, "not_solid", metrics

    return True, "verified_large_squamous", metrics


def verify_ep_tr(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None,
                 min_area: float = 1000.0, min_circ: float = 0.50,
                 max_box_ar: float = 2.2, min_fill: float = 0.30,
                 smooth_sigma: float = 2.0) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Gating logic for EP_TR (Transitional / urothelial epithelial cell):
    - Size check: box area >= 1000 px² (bigger than WBC / RBC / yeast / debris); no upper limit.
    - Aspect ratio: box within 1/2.2 - 2.2 (round / oval, not a long strand).
    - Round shape: circularity 4*pi*A/P² >= 0.50 on a smoothed outline (perfect circle = 1.0,
      normal oval ~0.8, irregular / stringy clump much lower); element must cover >= 30% of the crop.
    """
    metrics: Dict[str, Any] = {}
    dims = _meta_dims(meta, metrics)
    if dims is not None:
        ar, area, w, h = dims
        if area < min_area:
            return False, "too_small", metrics
        if ar > max_box_ar or ar < 1.0 / max_box_ar:
            return False, "aspect_ratio_out_of_bounds", metrics

    filled = _prep_mask(crop, bg, smooth_sigma=smooth_sigma)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 60]
    if not cnts:
        return False, "no_contours_found", metrics

    crop_area = float(max(crop.shape[0] * crop.shape[1], 1))

    def circ_of(c):
        peri = max(float(cv2.arcLength(c, True)), 1.0)
        return min(4.0 * np.pi * cv2.contourArea(c) / (peri ** 2), 1.0)

    # roundest element that is big enough relative to the crop (same pick rule as the screening script)
    cands = [c for c in cnts if cv2.contourArea(c) / crop_area >= min_fill] or [max(cnts, key=cv2.contourArea)]
    c = max(cands, key=circ_of)
    el_area = cv2.contourArea(c)
    circ = circ_of(c)
    fill = el_area / crop_area
    metrics["element_area"] = round(float(el_area), 1)
    metrics["circularity"] = round(float(circ), 3)
    metrics["fill_fraction"] = round(float(fill), 3)

    if dims is None and el_area < min_area:
        return False, "too_small", metrics
    if fill < min_fill:
        return False, "element_too_sparse", metrics
    if circ < min_circ:
        return False, "not_round", metrics

    return True, "verified_round_transitional", metrics


# Struvite, small_struvite, and Debris-rescue filters are disabled for now.
# def verify_struvite(crop: np.ndarray, bg: float, meta: Optional[Dict[str, Any]] = None,
#                     min_max_side: float = 30.0, min_area: float = 600.0,
#                     min_rectangularity: float = 0.70, min_solidity: float = 0.85,
#                     min_verts: int = 4, max_verts: int = 8, max_aspect: float = 5.0,
#                     max_circ: float = 0.85, cluster_min_area: float = 1500.0,
#                     cluster_min_lines: int = 4, eps_frac: float = 0.02,
#                     min_contrast: float = 12.0) -> Tuple[bool, str, Dict[str, Any]]:
#     """
#     Gating logic for Struvite (triple phosphate, "coffin-lid" crystals):
#     - Size check: longest box side >= 30 px and box area >= 600 px².
#     - Prism shape: largest hole-filled element is rectangular - min-area-rectangle fill >= 0.70,
#       solidity >= 0.85, 4-8 polygon vertices, long/short side ratio <= 5, and not round (circularity < 0.85).
#     - Cluster fallback (overlapping / aggregated crystals): large element (>= 1500 px²) with >= 4 long
#       straight edge segments (Canny + Hough) pointing in at least two different directions.
#     """
#     metrics: Dict[str, Any] = {}
#     dims = _meta_dims(meta, metrics)
#     if dims is not None:
#         _, area, w, h = dims
#         if max(w, h) < min_max_side or area < min_area:
#             return False, "too_small", metrics
#
#     contrast = float(np.percentile(np.abs(crop.astype(np.float32) - bg), 99))
#     metrics["contrast"] = round(contrast, 1)
#     if contrast < min_contrast:
#         return False, "low_contrast", metrics
#
#     filled = _prep_mask(crop, bg)
#     cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = [c for c in cnts if cv2.contourArea(c) >= 40]
#     if not cnts:
#         return False, "no_contours_found", metrics
#
#     c = max(cnts, key=cv2.contourArea)
#     el_area = cv2.contourArea(c)
#     peri = cv2.arcLength(c, True)
#     h_area = cv2.contourArea(cv2.convexHull(c))
#     solidity = el_area / (h_area + 1e-5)
#     circ = min(4.0 * np.pi * el_area / (peri ** 2), 1.0) if peri > 0 else 0.0
#
#     rw, rh = cv2.minAreaRect(c)[1]
#     rectangularity = min(el_area / max(float(rw * rh), 1.0), 1.0)
#     aspect = max(rw, rh) / (min(rw, rh) + 1e-5)
#     verts = len(cv2.approxPolyDP(c, eps_frac * peri, True))
#     metrics.update({
#         "element_area": round(float(el_area), 1), "rectangularity": round(float(rectangularity), 3),
#         "solidity": round(float(solidity), 3), "vertices": int(verts),
#         "rect_aspect": round(float(aspect), 2), "circularity": round(float(circ), 3),
#     })
#
#     if dims is None and (el_area < min_area or max(rw, rh) < min_max_side):
#         return False, "too_small", metrics
#     if circ >= max_circ:
#         return False, "round_not_prism", metrics
#
#     if (rectangularity >= min_rectangularity and solidity >= min_solidity
#             and min_verts <= verts <= max_verts and aspect <= max_aspect):
#         return True, "verified_rectangular_prism", metrics
#
#     # Cluster fallback: several straight edges in more than one direction
#     if el_area >= cluster_min_area:
#         edges = cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 40, 120)
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=3)
#         angs = []
#         if lines is not None:
#             # OpenCV normally returns (N, 1, 4), but some builds return
#             # (N, 4) or a single flattened line. Normalize the shape first.
#             line_data = np.asarray(lines)
#             if line_data.size % 4 == 0:
#                 line_data = line_data.reshape(-1, 4)
#             else:
#                 line_data = np.empty((0, 4), dtype=np.int32)
#             for x1, y1, x2, y2 in line_data:
#                 angs.append(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0))
#         metrics["straight_edges"] = len(angs)
#         if len(angs) >= cluster_min_lines:
#             def adiff(a):
#                 d = abs(a - angs[0]) % 180.0
#                 return min(d, 180.0 - d)
#             if any(adiff(a) > 20.0 for a in angs):
#                 return True, "verified_crystal_cluster", metrics
#
#     return False, "not_rectangular_prism", metrics
#
#
# def verify_small_struvite(crop: np.ndarray, bg: float,
#                           meta: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
#     """Apply the Struvite prism check with small-crystal size limits."""
#     return verify_struvite(
#         crop, bg, meta,
#         min_max_side=12.0, min_area=120.0,
#         min_rectangularity=0.65, min_solidity=0.80,
#         min_verts=4, max_verts=8, max_aspect=5.0, max_circ=0.85,
#         cluster_min_area=400.0, cluster_min_lines=4,
#         eps_frac=0.04, min_contrast=12.0,
#     )
#
#
# def rescue_struvite_from_debris(patch: Optional[np.ndarray],
#                                 meta: Optional[Dict[str, Any]] = None,
#                                 min_side: float = 30.0) -> Tuple[bool, str, Dict[str, Any]]:
#     """Use a strict rectangular-prism test to identify Struvite among Debris predictions."""
#     if patch is None or patch.size == 0:
#         return False, "empty_patch", {}
#     crop, bg = extract_foreground_crop(patch)
#     if crop is None or crop.size == 0:
#         return False, "empty_foreground", {}
#     return verify_struvite(
#         crop, bg, meta,
#         min_max_side=min_side, min_area=600.0,
#         min_rectangularity=0.80, min_solidity=0.90,
#         min_verts=4, max_verts=6, max_aspect=4.5, max_circ=0.80,
#         cluster_min_area=1e12, cluster_min_lines=10 ** 6,
#         eps_frac=0.02, min_contrast=30.0,
#     )
#
#

def verify_gated_object(
    patch_or_crop: Optional[np.ndarray],
    predicted_class: str,
    meta: Optional[Dict[str, Any]] = None,
    rejection_class: Optional[str] = None,
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    Main gating verification entrypoint.

    Args:
        patch_or_crop: Patch image (80x80 / 224x224) or tight crop as a numpy array.
        predicted_class: Class name produced by detector or classifier.
        meta: Optional metadata dict with keys like 'raw_w', 'raw_h', 'area', 'aspect_ratio'.
        rejection_class: Class to reassign if verification fails (default: '<predicted_class>_rejected').

    Returns:
        is_verified (bool): True if verified, False if rejected by gating.
        final_class (str): predicted_class if verified, rejection_class if rejected.
        reason (str): Short description of why the decision was made.
        metrics (dict): Numeric features extracted during verification.
    """
    # Non-gated classes pass through untouched
    if not is_gated_class(predicted_class):
        return True, predicted_class, "class_not_gated", {}

    if rejection_class is None:
        rejection_class = f"{predicted_class}_rejected"

    if patch_or_crop is None or patch_or_crop.size == 0:
        return False, rejection_class, "empty_patch_or_crop", {}

    cls_lower = predicted_class.lower()

    crop, bg = extract_foreground_crop(patch_or_crop)
    if crop is None or crop.size == 0:
        return False, rejection_class, "empty_foreground", {}

    if cls_lower == "co_di":
        ok, reason, metrics = verify_codi(crop, bg, meta)
    elif cls_lower == "rbc":
        ok, reason, metrics = verify_rbc(crop, bg, meta)
    elif cls_lower in ("small_sperm", "sperm"):
        ok, reason, metrics = verify_sperm(crop, bg, meta)
    elif cls_lower == "wbc":
        ok, reason, metrics = verify_wbc(crop, bg, meta)
    elif cls_lower == "yeast":
        ok, reason, metrics = verify_yeast(crop, bg, meta)
    elif cls_lower == "ep_sq":
        ok, reason, metrics = verify_ep_sq(crop, bg, meta)
    elif cls_lower == "ep_tr":
        ok, reason, metrics = verify_ep_tr(crop, bg, meta)
    # Struvite morphology filtering is temporarily disabled for both labels.
    # elif cls_lower == "struvite":
    #     ok, reason, metrics = verify_struvite(crop, bg, meta)
    # elif cls_lower == "small_struvite":
    #     ok, reason, metrics = verify_small_struvite(crop, bg, meta)
    else:
        return True, predicted_class, "class_not_gated", {}

    final_class = predicted_class if ok else rejection_class
    return ok, final_class, reason, metrics
