"""
run_sahi_inference.py
---------------------
Unified SAHI detection and patch classification pipeline.

Detection Model:
    /home/sigtuple/Downloads/urine_iter_4_5_only_object.onnx
    Classes: Object, Amorphous, Bacteria, Debris

Classification Model:
    /home/sigtuple/DS/SAHI_Wrapper_yolov26/SAHI_Wrapper_yolov26/urine_mobilenetv4_gem_msfm (2).onnx
    Classes (13):
        - "CO_DI"
        - "CO_DI_CLUMP"
        - "Debris"
        - "EP_SQ"
        - "EP_TR"
        - "fungal_hyphae"
        - "Granular_cast"
        - "RBC"
        - "Sperm"
        - "Struvite"
        - "unknown_crystals"
        - "WBC"
        - "Yeast"

Patch Process:
    extract_isolated_patch_black: Extracts target element at 1:1 scale centered on a black canvas (224x224).
"""

import argparse
import os
import time
import json
import pickle
import shutil
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sahi.predict import get_sliced_prediction

from sahi_yolo_wrapper import YOLOSAHIWrapper
from classification_inferencing import FasterNetClassifier, FASTERNET_CLASSES, CLASSIFIER_CLASSES, WBC_Classifier

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DET_MODEL = os.path.join(_DIR, "urine_iter_4_5_only_object.onnx")
if not os.path.exists(DEFAULT_DET_MODEL):
    DEFAULT_DET_MODEL = "/home/sigtuple/Downloads/urine_iter_4_5_only_object.onnx"



DETECTION_CLASS_MAPPING = {
    "0": "Object",
    "1": "Amorphous",
    "2": "Bacteria",
    "3": "Debris",
}

YOLOV8_CLASS_MAPPING = DETECTION_CLASS_MAPPING
YOLOX_CLASS_MAPPING = DETECTION_CLASS_MAPPING
YOLOV26_CLASS_MAPPING = DETECTION_CLASS_MAPPING

CLASS_MAPPING_LOOKUP = {
    "yolox":  YOLOX_CLASS_MAPPING,
    "yolov8": YOLOV8_CLASS_MAPPING,
    "yolov26": YOLOV26_CLASS_MAPPING,
}

COLOUR_BY_NAME = {
    # Detection classes
    "Object": (255, 255, 255),
    "Amorphous": (210, 180, 140),
    "Bacteria": (0, 255, 255),
    "Debris": (255, 165, 0),
    # Classification classes
    "Bilirubin": (0, 215, 255),
    "CO_DI": (255, 105, 180),
    "CO_DI_CLUMP": (199, 21, 133),
    "CO_MO": (238, 130, 238),
    "EP_SQ": (60, 179, 113),
    "EP_TR": (32, 178, 170),
    "Fat": (0, 140, 255),
    "Granular_cast": (128, 0, 128),
    "RBC": (0, 0, 255),
    "Sperm": (255, 255, 0),
    "Struvite": (205, 90, 106),
    "UNFOCUSED_CO_DI": (180, 105, 255),
    "WBC": (255, 0, 0),
    "Yeast": (0, 255, 127),
    "fungal_hyphae": (139, 69, 19),
    "unknown_crystals": (147, 20, 255),
    "unstained_rbc": (50, 50, 205),
}

STANDARD_DETECTION_CLASSES = ["Object", "Amorphous", "Bacteria", "Debris"]

CLASSIFIER_CLASSES = [
    "CO_DI",
    "CO_DI_CLUMP",
    "Debris",
    "EP_SQ",
    "EP_TR",
    "fungal_hyphae",
    "Granular_cast",
    "RBC",
    "Sperm",
    "Struvite",
    "unknown_crystals",
    "WBC",
    "Yeast",
]

STANDARD_CLASSIFIED_CLASSES = [
    "Bacteria",
    "RBC",
    "WBC",
    "EP_SQ",
    "EP_TR",
    "Yeast",
    "Sperm",
    "Granular_cast",
    "Struvite",
    "CO_DI",
    "CO_DI_CLUMP",
    "fungal_hyphae",
    "unknown_crystals",
    "Amorphous",
    "Debris",
]


def clean_col_name(name, suffix="count"):
    clean = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return f"{clean}_{suffix}" if suffix else clean


def parse_args():
    p = argparse.ArgumentParser(description="Unified SAHI inference with Classification")
    p.add_argument("--model_path", required=False,
                   default=DEFAULT_DET_MODEL,
                   help="Path to the ONNX detection model")
    p.add_argument("--model_type", required=False, default='yolox', choices=["yolox", "yolov8", "yolov26"],
                   help="Model type backend for detection")
    p.add_argument("--cls_model_path", required=False,
                   default='/home/sigtuple/DS/SAHI_Wrapper_yolov26/SAHI_Wrapper_yolov26/urine_mobilenetv4_gem_msfm_CLAHE_rbc_combined.onnx',
                   help="Path to the ONNX classification model")
    p.add_argument("--patch_size", type=int, default=224,
                   help="Patch size for classification model (default: 224)")
    p.add_argument("--batch_dir", default=None,
                   help="Root directory containing tray_ids/order_ids for batch processing (supports e.g. test_dataset/device_1/ or tray or .../urine/Recon_data/rgb_image/)")
    p.add_argument("--batch_output_dir", default=None,
                   help="Optional root output directory for batch mode (defaults to <order_path>/results/)")
    p.add_argument("--skip_existing", action="store_true", default=False,
                   help="Skip orders whose timing CSV already exists in results folder")
    p.add_argument("--max_orders", type=int, default=None,
                   help="Maximum number of orders to process in batch mode (useful for testing)")
    p.add_argument("--max_fovs", type=int, default=None,
                   help="Maximum number of FOVs to process per order (useful for quick testing)")
    p.add_argument("--input_dir", required=False,
                   default='/home/sigtuple/DS/urine_validation/test_dataset/device_1/8b478c5865324c97/c2c1c9a3-5321-4f22-9578-4e5c46882aca/urine/Recon_data/rgb_image/',
                   help="Directory with input images (single-order mode)")
    p.add_argument("--output_dir", default=None,
                   help="Directory for annotated outputs (single-order mode, default: <order_dir>/results/)")
    p.add_argument("--timing_csv", default="inference_time.csv",
                   help="Filename for the inference timing and detection statistics CSV output (default: inference_time.csv)")
    p.add_argument("--fov_csv", default="fov_counts.csv",
                   help="Filename for per-FOV detection and classification counts CSV (default: fov_counts.csv)")
    p.add_argument("--overall_csv", default="overall_counts.csv",
                   help="Filename for overall order element counts CSV (default: overall_counts.csv)")
    p.add_argument("--mean_csv", default="mean_results.csv",
                   help="Filename for final mean of elements CSV (default: mean_results.csv)")
    p.add_argument("--tray_id", default="",
                   help="Optional tray ID (single-order mode)")
    p.add_argument("--order_id", default="",
                   help="Optional order ID (single-order mode)")
    p.add_argument("--csv_only", action="store_true", default=False,
                   help="Generate only CSV reports. Disables saving patch images, annotated bbox images, and COCO JSON for maximum speed.")
    p.add_argument("--save_patches", dest="save_patches", action="store_true", default=True,
                   help="Save cropped patches class-wise in classification_result folder (default: True unless --csv_only)")
    p.add_argument("--no_save_patches", dest="save_patches", action="store_false",
                   help="Do not save cropped patches")
    p.add_argument("--save_annotated", dest="save_annotated", action="store_true", default=True,
                   help="Save full FOV images with bounding boxes drawn (default: True unless --csv_only)")
    p.add_argument("--no_save_annotated", dest="save_annotated", action="store_false",
                   help="Do not save full FOV images with bounding boxes")
    p.add_argument("--save_coco", dest="save_coco", action="store_true", default=True,
                   help="Save predictions_coco.json file (default: True unless --csv_only)")
    p.add_argument("--no_save_coco", dest="save_coco", action="store_false",
                   help="Do not save predictions_coco.json file")
    p.add_argument("--cls_batch_size", type=int, default=64,
                   help="Batch size for patch classification (default: 64)")
    p.add_argument("--save_patch_classes", default="all", choices=["all", "classified_only"],
                   help="Which classes to save patches for: 'all' (all detections) or 'classified_only' (only Object classified by classifier)")
    p.add_argument("--patch_dir_name", default="classification_result",
                   help="Subdirectory name inside output_dir to save class-wise patches (default: classification_result)")
    p.add_argument("--img_size", type=int, default=416, help="Slice / model input size")
    p.add_argument("--confidence", type=float, default=0.1, help="Confidence threshold")
    p.add_argument("--overlap", type=float, default=0.0, help="Slice overlap ratio")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Pre-scale factor applied to the image before slicing (e.g. 0.5 to halve resolution)")
    p.add_argument("--iou_threshold", type=float, default=0.4,
                   help="IOU threshold for duplicate box suppression")
    p.add_argument("--containment_threshold", type=float, default=0.5,
                   help="Containment threshold for suppressing small boxes inside bigger ones")

    args = p.parse_args()
    if args.csv_only:
        args.save_patches = False
        args.save_annotated = False
        args.save_coco = False

    return args


# ──────────────────────────────────────────────
# Duplicate box removal (IOU + containment suppression)
# ──────────────────────────────────────────────
def compute_iou_containment(box_a, box_b):
    """
    box format: [x1, y1, x2, y2]
    Returns: (iou, containment)
    containment = fraction of the smaller box covered by the larger box
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0

    smaller_area = min(area_a, area_b)
    containment = inter_area / smaller_area if smaller_area > 0 else 0.0

    return iou, containment


def deduplicate_predictions(predictions, iou_threshold=0.4, containment_threshold=0.5):
    """
    predictions: list of SAHI ObjectPrediction objects
    Removes duplicate / contained boxes. Keeps the higher-confidence box.
    Returns a filtered list of ObjectPrediction objects.
    """
    if not predictions:
        return predictions

    preds_sorted = sorted(predictions, key=lambda p: p.score.value, reverse=True)

    kept = []
    suppressed = [False] * len(preds_sorted)

    for i, pred_i in enumerate(preds_sorted):
        if suppressed[i]:
            continue

        kept.append(pred_i)
        box_i = [pred_i.bbox.minx, pred_i.bbox.miny, pred_i.bbox.maxx, pred_i.bbox.maxy]

        for j in range(i + 1, len(preds_sorted)):
            if suppressed[j]:
                continue

            pred_j = preds_sorted[j]
            box_j = [pred_j.bbox.minx, pred_j.bbox.miny, pred_j.bbox.maxx, pred_j.bbox.maxy]

            iou, containment = compute_iou_containment(box_i, box_j)

            if iou >= iou_threshold:
                suppressed[j] = True
                continue

            if containment >= containment_threshold:
                suppressed[j] = True

    return kept


def get_fov_number(img_name):
    """
    Extract FOV number from image filename.
    Examples:
        0057_010.jpg -> "0057"
        0000_001_BestFocusedWLImage.jpg -> "0000"
        0000_001_BestFocusedWLImage_patch_0_0.png -> "0000_001_BestFocusedWLImage_patch_0_0"
    """
    base = os.path.splitext(img_name)[0]
    parts = base.split('_')
    if len(parts) >= 1 and parts[0].isdigit():
        if "_patch_" in base:
            return base
        return parts[0]
    return base


def extract_isolated_patch_black(img, bbox, patch_size=224, pad_ratio=0.15):
    """
    Extracts only the target element at 1:1 scale centered on a black canvas.

    Args:
        img: Source image (numpy array, HxWxC)
        bbox: [x, y, w, h]
        patch_size: Output square dimension (default: 224)
        pad_ratio: Margin around the cell (default: 0.15 = 15%)

    Returns:
        patch: (patch_size, patch_size, 3) numpy array with black background
    """
    ih, iw = img.shape[:2]
    x, y, w, h = bbox

    # 1. Expand bbox by margin to capture full cell boundary
    px = int(round(w * pad_ratio))
    py = int(round(h * pad_ratio))
    x1 = max(0, int(round(x - px)))
    y1 = max(0, int(round(y - py)))
    x2 = min(iw, int(round(x + w + px)))
    y2 = min(ih, int(round(y + h + py)))

    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

    # 2. Scale down only if the object exceeds patch size (rare fallback)
    max_dim = max(cw, ch)
    if max_dim > patch_size:
        scale = patch_size / float(max_dim)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ch, cw = crop.shape[:2]

    # 3. Create black canvas and center the crop at 1:1 scale
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    top = (patch_size - ch) // 2
    left = (patch_size - cw) // 2
    canvas[top:top + ch, left:left + cw] = crop

    return canvas


def get_patch(img, x1, y1, x2, y2, patch_size=224):
    """
    Compatibility wrapper around extract_isolated_patch_black.
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    return extract_isolated_patch_black(img, [x1, y1, bw, bh], patch_size=patch_size)


def classify_patches_batched(classifier, patches, batch_size=64):
    """
    Classifies a list of patch numpy arrays in batches using FasterNetClassifier.
    Returns list of (cls_name, conf) tuples matching input order.
    """
    if not patches:
        return []
    results = []
    for i in range(0, len(patches), batch_size):
        chunk = patches[i : i + batch_size]
        stacked = np.stack(chunk, axis=0)
        res = classifier.predict(stacked)
        if isinstance(res, list):
            results.extend(res)
        else:
            results.append(res)
    return results


def has_pkl_files(d):
    """Checks if directory contains valid Recon PKL files."""
    if not d or not os.path.isdir(d):
        return False
    try:
        return any(
            f.lower().endswith(".pkl") and not f.lower().startswith(("metadata", "micro_qc", "macro"))
            for f in os.listdir(d)
        )
    except Exception:
        return False


def has_rgb_images(d):
    """Checks if directory contains PNG or JPG image files."""
    if not d or not os.path.isdir(d):
        return False
    try:
        return any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(d))
    except Exception:
        return False


def find_recon_pkl_dir(path):
    """
    Given an order folder, a urine folder, a Recon_data folder, or an rgb_image folder,
    finds the directory containing the actual .pkl files.
    Walks up the directory tree if path does not exist yet (e.g. uncreated rgb_image).
    """
    if not path:
        return None
    curr = os.path.abspath(path)
    # If path does not exist (e.g. uncreated rgb_image subfolder), walk up to existing parent
    while curr and not os.path.exists(curr) and curr != os.path.dirname(curr):
        curr = os.path.dirname(curr)
    if not os.path.exists(curr):
        return None
    if os.path.isfile(curr):
        curr = os.path.dirname(curr)

    # Check curr and its parents up to 4 levels
    check_dirs = [curr]
    p = curr
    for _ in range(4):
        p = os.path.dirname(p)
        if p and p not in check_dirs:
            check_dirs.append(p)

    for d in check_dirs:
        if has_pkl_files(d):
            return d
        for sub in [
            os.path.join("urine", "Recon_data"),
            os.path.join("urine", "recon_data"),
            "Recon_data",
            "recon_data",
        ]:
            cand = os.path.join(d, sub)
            if has_pkl_files(cand):
                return cand
    return None


def ensure_rgb_images_from_pkl(recon_dir):
    """
    Checks if recon_dir (or its PKL location) contains .pkl files.
    If so, extracts BestFocusedWLImage to recon_dir/rgb_image/ as .png files.
    Returns path to rgb_image directory, or None if no pkls found.
    """
    if not recon_dir:
        return None

    actual_pkl_dir = find_recon_pkl_dir(recon_dir)
    if not actual_pkl_dir:
        return None

    pkl_files = [
        f for f in sorted(os.listdir(actual_pkl_dir))
        if f.lower().endswith(".pkl") and not f.lower().startswith(("metadata", "micro_qc", "macro"))
    ]
    if not pkl_files:
        return None

    rgb_dir = os.path.join(actual_pkl_dir, "rgb_image")
    try:
        os.makedirs(rgb_dir, exist_ok=True)
    except Exception as e:
        print(f"[Warning] Could not create {rgb_dir}: {e}")
        return None

    # Check which PKLs need conversion
    to_convert = []
    for f in pkl_files:
        png_name = os.path.splitext(f)[0] + ".png"
        png_path = os.path.join(rgb_dir, png_name)
        if not (os.path.exists(png_path) and os.path.getsize(png_path) > 0):
            to_convert.append((f, png_path))

    if to_convert:
        parent_label = os.path.basename(os.path.dirname(os.path.dirname(actual_pkl_dir))) or os.path.basename(actual_pkl_dir)
        print(f"\n[Auto-Convert] Found {len(pkl_files)} PKL files in {actual_pkl_dir}")
        print(f"               Converting {len(to_convert)} PKLs to RGB PNG in {rgb_dir}...")
        for pkl_name, png_path in tqdm(to_convert, desc=f"PKL->RGB [{parent_label}]", leave=False):
            pkl_path = os.path.join(actual_pkl_dir, pkl_name)
            try:
                with open(pkl_path, "rb") as pf:
                    data = pickle.load(pf)
                if isinstance(data, dict) and "BestFocusedWLImage" in data:
                    img = data["BestFocusedWLImage"]
                    if img is not None and getattr(img, "size", 0) > 0:
                        cv2.imwrite(png_path, img)
            except Exception as e:
                print(f"[Warning] Failed to convert {pkl_path}: {e}")

    has_img = has_rgb_images(rgb_dir)
    return rgb_dir if has_img else None


def get_images_dir(folder, auto_convert=True):
    """
    Checks common image subfolder locations within an order directory:
    - urine/Recon_data/rgb_image (and case variants)
    - Recon_data/rgb_image
    - rgb_image
    - merged_fov
    - images
    - folder itself (flat images)
    If no images are found, checks for PKL files via find_recon_pkl_dir(folder)
    and auto-converts BestFocusedWLImage to an rgb_image/ subfolder when auto_convert=True.
    If auto_convert=False, returns candidate path if PKLs exist without converting yet.
    Returns path to the folder containing images, or None.
    """
    if not folder:
        return None

    curr = os.path.abspath(folder)

    # 1. Existing image folders check
    if has_rgb_images(curr):
        return curr

    sub_candidates = [
        os.path.join(curr, "urine", "Recon_data", "rgb_image"),
        os.path.join(curr, "urine", "recon_data", "rgb_image"),
        os.path.join(curr, "Recon_data", "rgb_image"),
        os.path.join(curr, "recon_data", "rgb_image"),
        os.path.join(curr, "rgb_image"),
        os.path.join(curr, "merged_fov"),
        os.path.join(curr, "images"),
    ]
    for c in sub_candidates:
        if has_rgb_images(c):
            return c

    # 2. Check for PKL files in Recon_data
    recon_pkl_dir = find_recon_pkl_dir(folder)
    if recon_pkl_dir:
        rgb_dir = os.path.join(recon_pkl_dir, "rgb_image")
        if has_rgb_images(rgb_dir):
            return rgb_dir
        if auto_convert:
            return ensure_rgb_images_from_pkl(recon_pkl_dir)
        else:
            return rgb_dir

    return None


def find_batch_orders(batch_dir):
    """
    Finds all (tray_id, order_id, order_path, images_dir) inside batch_dir.
    Supports:
        - Direct rgb_image folder (e.g. .../<tray>/<order>/urine/Recon_data/rgb_image/)
        - Direct order folder (e.g. .../<tray>/<order>/)
        - Tray directory (e.g. .../<tray>/ containing orders)
        - Device root (e.g. .../device_1/ containing <tray>/<order>/...)
    """
    batch_dir = os.path.abspath(batch_dir.rstrip("/" + os.sep))
    orders = []

    # Case A: batch_dir itself contains images directly or in urine/Recon_data/rgb_image
    img_dir = get_images_dir(batch_dir, auto_convert=False)
    if img_dir:
        parts = batch_dir.split(os.sep)
        if parts[-1] == "rgb_image" and len(parts) >= 4 and parts[-2].lower() == "recon_data" and parts[-3].lower() == "urine":
            order_path = os.path.abspath(os.path.join(batch_dir, "..", "..", ".."))
            base_parts = order_path.split(os.sep)
            order_id = base_parts[-1]
            tray_id = base_parts[-2] if len(base_parts) >= 2 else ""
        elif parts[-1].lower() == "recon_data" and len(parts) >= 3 and parts[-2].lower() == "urine":
            order_path = os.path.abspath(os.path.join(batch_dir, "..", ".."))
            base_parts = order_path.split(os.sep)
            order_id = base_parts[-1]
            tray_id = base_parts[-2] if len(base_parts) >= 2 else ""
        elif parts[-1] in ("merged_fov", "images", "rgb", "rgb_image"):
            order_path = os.path.abspath(os.path.join(batch_dir, ".."))
            base_parts = order_path.split(os.sep)
            order_id = base_parts[-1]
            tray_id = base_parts[-2] if len(base_parts) >= 2 else ""
        else:
            order_path = batch_dir
            order_id = parts[-1]
            tray_id = parts[-2] if len(parts) >= 2 else ""
        orders.append((tray_id, order_id, order_path, img_dir))
        return orders

    # Case B: Standard hierarchy search (supports device / tray / order OR tray / order)
    for root, dirs, files in os.walk(batch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("results", "__pycache__")]

        sub_img_dir = get_images_dir(root, auto_convert=False)
        if sub_img_dir and sub_img_dir != root:
            # root is an order folder containing urine/Recon_data/rgb_image
            rel = os.path.relpath(root, batch_dir).split(os.sep)
            if len(rel) == 1:
                tray_id = os.path.basename(batch_dir)
                order_id = rel[0]
            elif len(rel) >= 2:
                tray_id = rel[-2]
                order_id = rel[-1]
            else:
                tray_id, order_id = "", rel[0]
            orders.append((tray_id, order_id, root, sub_img_dir))
            dirs.clear()
        elif sub_img_dir == root:
            # flat image folder directly under tray/order
            if not any(p.lower() in ("urine", "recon_data") for p in root.split(os.sep)):
                rel = os.path.relpath(root, batch_dir).split(os.sep)
                if len(rel) >= 2:
                    tray_id = rel[-2]
                    order_id = rel[-1]
                    orders.append((tray_id, order_id, root, root))
    orders.sort(key=lambda x: (x[0], x[1]))
    return orders


def process_single_order(input_dir, output_dir, detection_model, classifier, args, tray_id="", order_id=""):
    """
    Processes all images in input_dir for a single order.
    Saves annotations, patch images, COCO JSON, class counts CSV, timing CSV,
    FOV counts CSV, overall counts CSV, and mean results CSV in output_dir.
    Returns (timing_records, class_counts, fov_records, overall_rec, mean_rec).
    """
    os.makedirs(output_dir, exist_ok=True)

    classification_result_dir = os.path.join(output_dir, args.patch_dir_name)
    if args.save_patches:
        if os.path.exists(classification_result_dir):
            print(f"Clearing previous classification results for order {order_id} at: {classification_result_dir}")
            shutil.rmtree(classification_result_dir)
        os.makedirs(classification_result_dir, exist_ok=True)

    class_counts = defaultdict(int)
    class_mapping = CLASS_MAPPING_LOOKUP[args.model_type]

    image_paths = sorted(
        glob(os.path.join(input_dir, "*.jpg"))
        + glob(os.path.join(input_dir, "*.jpeg"))
        + glob(os.path.join(input_dir, "*.png"))
    )

    if not image_paths:
        # Fallback: attempt just-in-time PKL conversion if Recon_data PKLs exist
        converted_dir = get_images_dir(input_dir, auto_convert=True)
        if converted_dir and has_rgb_images(converted_dir):
            input_dir = converted_dir
            image_paths = sorted(
                glob(os.path.join(input_dir, "*.jpg"))
                + glob(os.path.join(input_dir, "*.jpeg"))
                + glob(os.path.join(input_dir, "*.png"))
            )

    if not image_paths:
        print(f"No images found in {input_dir}, skipping.")
        return [], {}, [], {}, {}

    if getattr(args, "max_fovs", None) and args.max_fovs > 0:
        image_paths = image_paths[:args.max_fovs]

    # Standard classes for raw detection and final classification
    all_det_classes = list(dict.fromkeys(STANDARD_DETECTION_CLASSES + list(class_mapping.values())))
    all_final_classes = list(dict.fromkeys(STANDARD_CLASSIFIED_CLASSES + CLASSIFIER_CLASSES))

    # Unified COCO categories (detection + classification classes)
    all_categories = list(dict.fromkeys(list(class_mapping.values()) + CLASSIFIER_CLASSES))
    category_to_id = {name: i for i, name in enumerate(all_categories)}

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name}
            for name, i in category_to_id.items()
        ]
    }

    ann_id = 1
    timing_records = []
    fov_records = []
    desc = f"Inference [{order_id or os.path.basename(input_dir)}]"

    for image_id, img_path in enumerate(tqdm(image_paths, desc=desc), start=1):
        t_img_start = time.perf_counter()
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Optional pre-scale
        if args.scale != 1.0:
            img_input = cv2.resize(img, (0, 0), fx=args.scale, fy=args.scale)
        else:
            img_input = img.copy()

        h, w = img.shape[:2]

        if args.save_coco:
            coco["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": w,
                "height": h
            })

        # ── SAHI sliced prediction (Detection Timing) ──
        t_det_start = time.perf_counter()
        result = get_sliced_prediction(
            img_input[:, :, ::-1],          # SAHI expects RGB
            detection_model,
            slice_height=args.img_size,
            slice_width=args.img_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            perform_standard_pred=False,
            verbose=0,
        )
        det_time = time.perf_counter() - t_det_start

        vis = img.copy() if args.save_annotated else None
        patch_count = 0
        cls_time = 0.0
        patch_save_time = 0.0
        total_detections = len(result.object_prediction_list)

        fov_raw_det_counts = defaultdict(int)
        fov_final_counts = defaultdict(int)

        # Collect patches for object detections to classify in batches
        obj_indices = []
        obj_patches = []
        for pred_i, pred in enumerate(result.object_prediction_list):
            det_cls_name = pred.category.name
            fov_raw_det_counts[det_cls_name] += 1
            if det_cls_name not in all_det_classes:
                all_det_classes.append(det_cls_name)

            if det_cls_name.lower() == "object":
                x1 = int(round(pred.bbox.minx / args.scale))
                y1 = int(round(pred.bbox.miny / args.scale))
                x2 = int(round(pred.bbox.maxx / args.scale))
                y2 = int(round(pred.bbox.maxy / args.scale))
                bbox = [x1, y1, x2 - x1, y2 - y1]
                patch = extract_isolated_patch_black(img, bbox, patch_size=args.patch_size)
                obj_indices.append(pred_i)
                obj_patches.append(patch)

        # Batch classify all object patches in this FOV
        t_cls_start = time.perf_counter()
        if obj_patches:
            classified_results = classify_patches_batched(classifier, obj_patches, batch_size=args.cls_batch_size)
            patch_count = len(obj_patches)
        else:
            classified_results = []
        cls_time = time.perf_counter() - t_cls_start

        obj_map = {idx: (res, patch) for idx, res, patch in zip(obj_indices, classified_results, obj_patches)}

        for pred_i, pred in enumerate(result.object_prediction_list):
            x1 = int(round(pred.bbox.minx / args.scale))
            y1 = int(round(pred.bbox.miny / args.scale))
            x2 = int(round(pred.bbox.maxx / args.scale))
            y2 = int(round(pred.bbox.maxy / args.scale))
            det_cls_name = pred.category.name
            det_conf = pred.score.value

            cls_name_classify = None
            cls_conf = None
            patch = None

            if pred_i in obj_map:
                (cls_name_classify, cls_conf), patch = obj_map[pred_i]
                final_cls = cls_name_classify
                final_conf = cls_conf
            else:
                final_cls = det_cls_name
                final_conf = det_conf
                if args.save_patches and args.save_patch_classes == "all":
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    patch = extract_isolated_patch_black(img, bbox, patch_size=args.patch_size)

            # Filter sperm if low confidence
            if final_cls == "Sperm" and final_conf < 0.5:
                continue

            class_counts[final_cls] += 1
            fov_final_counts[final_cls] += 1
            if final_cls not in all_final_classes:
                all_final_classes.append(final_cls)

            # Save patch class-wise inside classification_result
            if args.save_patches and patch is not None:
                should_save = (args.save_patch_classes == "all") or (det_cls_name.lower() == "object")
                if should_save:
                    t_save_start = time.perf_counter()
                    cls_dir = os.path.join(classification_result_dir, final_cls)
                    os.makedirs(cls_dir, exist_ok=True)
                    base_name = os.path.splitext(img_name)[0]
                    patch_filename = f"{base_name}_det_{ann_id}_{final_cls}_{final_conf:.2f}.jpg"
                    patch_save_path = os.path.join(cls_dir, patch_filename)
                    cv2.imwrite(patch_save_path, patch)
                    patch_save_time += (time.perf_counter() - t_save_start)

            # Draw bounding box only if save_annotated is True
            if args.save_annotated and vis is not None:
                colour = COLOUR_BY_NAME.get(final_cls, (255, 255, 255))
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
                label = f"{final_cls} {final_conf:.2f}"

                # Put text label for non-bacteria
                if final_cls != "Bacteria" and det_cls_name != "Bacteria":
                    cv2.putText(
                        vis,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        1,
                        cv2.LINE_AA
                    )

            # Add to COCO annotations only if save_coco is True
            if args.save_coco:
                bw = x2 - x1
                bh = y2 - y1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_to_id[final_cls],
                    "category_name": final_cls,
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "score": float(final_conf),
                    "detection_class": det_cls_name,
                    "detection_score": float(det_conf),
                    "classification_class": cls_name_classify,
                    "classification_score": float(cls_conf) if cls_conf is not None else None,
                })
            ann_id += 1

        if args.save_annotated and vis is not None:
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, vis)
        overall_time = time.perf_counter() - t_img_start

        # Record FOV timing and detection data
        fov_num = get_fov_number(img_name)
        rec = {}
        if tray_id:
            rec["tray_id"] = tray_id
        if order_id:
            rec["order_id"] = order_id
        rec.update({
            "image_name": img_name,
            "fov_number": fov_num,
            "complete_detection_numbers": total_detections,
            "patches_for_classification": patch_count,
            "overall_time": round(overall_time, 5),
            "detection_time": round(det_time, 5),
            "classification_time": round(cls_time, 5),
            "time_to_save_patches": round(patch_save_time, 5),
        })
        timing_records.append(rec)

        # Determine element_present: 1 if any target element (excluding Debris) is detected, else 0
        has_element = any(
            cnt > 0 for cname, cnt in fov_final_counts.items()
            if cname.lower() not in ["debris", "debries"]
        )
        element_present = 1 if has_element else 0

        # Build FOV record for fov_counts.csv
        fov_rec = {
            "tray_id": tray_id,
            "order_id": order_id,
            "image_name": img_name,
            "fov_number": fov_num,
            "element_present": element_present,
            "total_detections": total_detections,
        }
        for det_cls in all_det_classes:
            fov_rec[f"det_{clean_col_name(det_cls, 'count')}"] = fov_raw_det_counts.get(det_cls, 0)
        for cls_name in all_final_classes:
            fov_rec[clean_col_name(cls_name, "count")] = fov_final_counts.get(cls_name, 0)

        fov_records.append(fov_rec)

    # Calculate overall counts and mean
    total_fovs = len(image_paths)
    element_present_fovs = sum(r["element_present"] for r in fov_records)

    # 1. Overall counts record (for overall_counts.csv)
    overall_rec = {
        "tray_id": tray_id,
        "order_id": order_id,
        "total_fov_count": total_fovs,
        "element_present_fov_count": element_present_fovs,
    }
    for det_cls in all_det_classes:
        col = f"det_{clean_col_name(det_cls, 'count')}"
        overall_rec[col] = sum(r.get(col, 0) for r in fov_records)

    for cls_name in all_final_classes:
        col = clean_col_name(cls_name, "count")
        overall_rec[col] = sum(r.get(col, 0) for r in fov_records)

    # 2. Mean results record (for mean_results.csv)
    # mean of element = count of the element / fov count where element present
    mean_rec = {
        "tray_id": tray_id,
        "order_id": order_id,
        "total_fov_count": total_fovs,
        "element_present_fov_count": element_present_fovs,
    }
    for det_cls in all_det_classes:
        count_col = f"det_{clean_col_name(det_cls, 'count')}"
        mean_col = f"det_{clean_col_name(det_cls, 'mean')}"
        tot_cnt = overall_rec.get(count_col, 0)
        mean_rec[mean_col] = round(tot_cnt / element_present_fovs, 6) if element_present_fovs > 0 else 0.0

    for cls_name in all_final_classes:
        count_col = clean_col_name(cls_name, "count")
        mean_col = clean_col_name(cls_name, "mean")
        tot_cnt = overall_rec.get(count_col, 0)
        mean_rec[mean_col] = round(tot_cnt / element_present_fovs, 6) if element_present_fovs > 0 else 0.0

    # Save COCO predictions if enabled
    if args.save_coco:
        json_path = os.path.join(output_dir, "predictions_coco.json")
        with open(json_path, "w") as f:
            json.dump(coco, f, indent=2)

    # Save class counts CSV
    csv_path = os.path.join(output_dir, "class_counts.csv")
    count_df = pd.DataFrame(
        [{"class": cls, "count": count} for cls, count in sorted(class_counts.items())]
    )
    count_df.to_csv(csv_path, index=False)

    # Save inference timing CSV
    timing_csv_path = os.path.join(output_dir, args.timing_csv)
    timing_df = pd.DataFrame(timing_records)
    timing_df.to_csv(timing_csv_path, index=False)

    # Save CSV 1: fov_counts.csv
    fov_csv_path = os.path.join(output_dir, args.fov_csv)
    pd.DataFrame(fov_records).to_csv(fov_csv_path, index=False)

    # Save CSV 2: overall_counts.csv
    overall_csv_path = os.path.join(output_dir, args.overall_csv)
    pd.DataFrame([overall_rec]).to_csv(overall_csv_path, index=False)

    # Save CSV 3: mean_results.csv
    mean_csv_path = os.path.join(output_dir, args.mean_csv)
    pd.DataFrame([mean_rec]).to_csv(mean_csv_path, index=False)

    return timing_records, class_counts, fov_records, overall_rec, mean_rec


def main():
    args = parse_args()
    class_mapping = CLASS_MAPPING_LOOKUP[args.model_type]

    # ── Build the unified SAHI detection model (loaded once) ──
    print(f"Loading Detection Model       : {args.model_type} ({args.model_path})")
    detection_model = YOLOSAHIWrapper(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        category_mapping=class_mapping,
        model_type=args.model_type,
        image_size=args.img_size,
    )

    # ── Build the FasterNet classification model (loaded once) ──
    print(f"Loading Classification Model  : {args.cls_model_path}")
    classifier = FasterNetClassifier(model_path=args.cls_model_path, classes=CLASSIFIER_CLASSES)
    classifier.load_model(device="CPU")
    print(f"Classification Classes        : {len(CLASSIFIER_CLASSES)} classes")
    print(f"Slice                         : {args.img_size}×{args.img_size}  overlap={args.overlap}")
    print(f"Dedup                         : iou_threshold={args.iou_threshold}  containment_threshold={args.containment_threshold}")
    print(f"Mode                          : {'CSV ONLY (zero images/patches written)' if args.csv_only else 'Standard'}")
    print(f"Save Patches                  : {args.save_patches}")
    print(f"Save Annotated Bbox Images    : {args.save_annotated}")
    print(f"Save COCO JSON                : {args.save_coco}")
    print(f"Classification Batch Size     : {args.cls_batch_size}")
    print()

    # ── Execution Mode: Batch vs Single Order ──
    if args.batch_dir:
        orders = find_batch_orders(args.batch_dir)
        if args.max_orders:
            orders = orders[:args.max_orders]

        print(f"Found {len(orders)} order(s) across trays in {args.batch_dir}")
        if not orders:
            print("No orders containing images were found to process.")
            return

        all_batch_records = []
        all_batch_fov_records = []
        all_batch_overall_records = []
        all_batch_mean_records = []
        overall_t0 = time.time()

        for idx, (tray_id, order_id, order_path, images_dir) in enumerate(orders, 1):
            if args.batch_output_dir:
                order_out = os.path.join(args.batch_output_dir, tray_id, order_id, "results")
            else:
                order_out = os.path.join(order_path, "results")

            timing_file = os.path.join(order_out, args.timing_csv)
            fov_file = os.path.join(order_out, args.fov_csv)
            overall_file = os.path.join(order_out, args.overall_csv)
            mean_file = os.path.join(order_out, args.mean_csv)

            if (
                args.skip_existing
                and os.path.exists(timing_file)
                and os.path.exists(fov_file)
                and os.path.exists(overall_file)
                and os.path.exists(mean_file)
            ):
                print(f"[{idx}/{len(orders)}] Skipping {tray_id}/{order_id} (already completed all CSVs)")
                try:
                    all_batch_records.extend(pd.read_csv(timing_file).to_dict(orient="records"))
                except Exception:
                    pass
                if os.path.exists(fov_file):
                    try:
                        all_batch_fov_records.extend(pd.read_csv(fov_file).to_dict(orient="records"))
                    except Exception:
                        pass
                if os.path.exists(overall_file):
                    try:
                        all_batch_overall_records.extend(pd.read_csv(overall_file).to_dict(orient="records"))
                    except Exception:
                        pass
                if os.path.exists(mean_file):
                    try:
                        all_batch_mean_records.extend(pd.read_csv(mean_file).to_dict(orient="records"))
                    except Exception:
                        pass
                continue

            print(f"\n[{idx}/{len(orders)}] Processing Tray: {tray_id} | Order: {order_id}")
            resolved_images_dir = (
                get_images_dir(images_dir, auto_convert=True)
                or get_images_dir(order_path, auto_convert=True)
                or images_dir
            )
            records, counts, fov_recs, overall_rec, mean_rec = process_single_order(
                input_dir=resolved_images_dir,
                output_dir=order_out,
                detection_model=detection_model,
                classifier=classifier,
                args=args,
                tray_id=tray_id,
                order_id=order_id,
            )
            all_batch_records.extend(records)
            all_batch_fov_records.extend(fov_recs)
            if overall_rec:
                all_batch_overall_records.append(overall_rec)
            if mean_rec:
                all_batch_mean_records.append(mean_rec)
            print(f"Completed {order_id}: {len(records)} FOVs processed. Saved to {order_out}/")

        # Save consolidated master batch CSVs
        batch_out_root = args.batch_output_dir or args.batch_dir
        if batch_out_root:
            b_parts = os.path.abspath(batch_out_root).rstrip(os.sep).split(os.sep)
            if len(b_parts) >= 4 and b_parts[-1] == "rgb_image" and b_parts[-2].lower() == "recon_data" and b_parts[-3].lower() == "urine":
                batch_out_root = os.path.abspath(os.path.join(batch_out_root, "..", "..", ".."))
            elif len(b_parts) >= 3 and b_parts[-1].lower() == "recon_data" and b_parts[-2].lower() == "urine":
                batch_out_root = os.path.abspath(os.path.join(batch_out_root, "..", ".."))
        if all_batch_records:
            master_csv_path = os.path.join(batch_out_root, f"batch_{args.timing_csv}")
            pd.DataFrame(all_batch_records).to_csv(master_csv_path, index=False)

        if all_batch_fov_records:
            master_fov_path = os.path.join(batch_out_root, f"batch_{args.fov_csv}")
            pd.DataFrame(all_batch_fov_records).to_csv(master_fov_path, index=False)

        if all_batch_overall_records:
            master_overall_path = os.path.join(batch_out_root, f"batch_{args.overall_csv}")
            pd.DataFrame(all_batch_overall_records).to_csv(master_overall_path, index=False)

        if all_batch_mean_records:
            master_mean_path = os.path.join(batch_out_root, f"batch_{args.mean_csv}")
            pd.DataFrame(all_batch_mean_records).to_csv(master_mean_path, index=False)

        total_elapsed = time.time() - overall_t0
        print(f"\n==========================================")
        print(f"Batch Processing Complete!")
        print(f"Total Orders Processed : {len(orders)}")
        print(f"Total FOVs Processed   : {len(all_batch_records)}")
        print(f"Total Batch Time       : {total_elapsed:.2f}s")
        if all_batch_records:
            print(f"Master Timing CSV      : {os.path.join(batch_out_root, f'batch_{args.timing_csv}')}")
        if all_batch_fov_records:
            print(f"Master FOV CSV         : {os.path.join(batch_out_root, f'batch_{args.fov_csv}')}")
        if all_batch_overall_records:
            print(f"Master Overall CSV     : {os.path.join(batch_out_root, f'batch_{args.overall_csv}')}")
        if all_batch_mean_records:
            print(f"Master Mean CSV        : {os.path.join(batch_out_root, f'batch_{args.mean_csv}')}")
        print(f"==========================================")

    else:
        # Single order mode
        raw_parts = os.path.abspath(args.input_dir).rstrip(os.sep).split(os.sep)
        if len(raw_parts) >= 4 and raw_parts[-1] == "rgb_image" and raw_parts[-2].lower() == "recon_data" and raw_parts[-3].lower() == "urine":
            order_path = os.path.abspath(os.path.join(args.input_dir, "..", "..", ".."))
        elif len(raw_parts) >= 3 and raw_parts[-1].lower() == "recon_data" and raw_parts[-2].lower() == "urine":
            order_path = os.path.abspath(os.path.join(args.input_dir, "..", ".."))
        elif raw_parts[-1] in ("merged_fov", "rgb", "images", "rgb_image", "results"):
            order_path = os.path.abspath(os.path.join(args.input_dir, ".."))
        else:
            order_path = os.path.abspath(args.input_dir)

        resolved_img_dir = (
            get_images_dir(args.input_dir, auto_convert=True)
            or get_images_dir(order_path, auto_convert=True)
            or args.input_dir
        )

        if args.output_dir is None:
            args.output_dir = os.path.join(order_path, "results")

        tray_id = args.tray_id
        order_id = args.order_id
        if not tray_id or not order_id:
            norm_parts = order_path.split(os.sep)
            if len(norm_parts) >= 2:
                inferred_order = norm_parts[-1]
                inferred_tray = norm_parts[-2]
                if not order_id:
                    order_id = inferred_order
                if not tray_id:
                    tray_id = inferred_tray

        t0 = time.time()
        records, counts, fov_records, overall_rec, mean_rec = process_single_order(
            input_dir=resolved_img_dir,
            output_dir=args.output_dir,
            detection_model=detection_model,
            classifier=classifier,
            args=args,
            tray_id=tray_id,
            order_id=order_id,
        )
        elapsed = time.time() - t0
        print(f"\nDone — {len(records)} images in {elapsed:.2f}s  "
              f"({elapsed / max(len(records), 1):.2f}s/img)")
        if records:
            avg_overall = np.mean([r["overall_time"] for r in records])
            avg_det = np.mean([r["detection_time"] for r in records])
            avg_cls = np.mean([r["classification_time"] for r in records])
            avg_save = np.mean([r["time_to_save_patches"] for r in records])
            total_dets = sum([r["complete_detection_numbers"] for r in records])
            total_cls_patches = sum([r["patches_for_classification"] for r in records])

            print(f"Total Detections            : {total_dets}")
            print(f"Total Classification Patches: {total_cls_patches}")
            print(f"Avg Overall Time            : {avg_overall:.5f}s / FOV")
            print(f"Avg Detection Time          : {avg_det:.5f}s / FOV")
            print(f"Avg Classification Time     : {avg_cls:.5f}s / FOV")
            print(f"Avg Time to Save Patches    : {avg_save:.5f}s / FOV")
        print(f"Results saved to            : {args.output_dir}/")
        print(f"  -> FOV counts CSV         : {os.path.join(args.output_dir, args.fov_csv)}")
        print(f"  -> Overall counts CSV     : {os.path.join(args.output_dir, args.overall_csv)}")
        print(f"  -> Mean results CSV       : {os.path.join(args.output_dir, args.mean_csv)}")


if __name__ == "__main__":
    main()