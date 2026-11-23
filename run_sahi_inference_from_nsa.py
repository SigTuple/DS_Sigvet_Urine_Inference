"""
run_sahi_inference_from_nsa.py
--------------------------------
SAHI detection and patch classification pipeline from NAS.

Workflow:
  1. Accepts a set of tray_run_ids (via CLI arguments or a .txt/.csv file).
  2. Resolves NAS mount path (e.g. smb://192.168.0.253/sigvet_data/sigvet_data/ -> GVFS / CIFS mount).
  3. Locates each tray_run_id (.zip file or directory) across NAS device folders.
  4. Inspects inside the zip / folder for urine/recon_data/*.pkl files, grouping by order_id.
  5. Extracts BestFocusedWLImage from PKLs and converts them to RGB PNG images.
  6. Runs unified SAHI detection + patch classification on the converted images.
  7. Generates individual CSVs for each order_id:
     - fov_counts.csv
     - overall_counts.csv
     - mean_results.csv
     - inference_time.csv
     - class_counts.csv
     - predictions_coco.json
  8. Generates consolidated combined CSVs for all processed orders:
     - combined_fov_counts.csv (and batch_fov_counts.csv)
     - combined_overall_counts.csv (and batch_overall_counts.csv)
     - combined_mean_results.csv (and batch_mean_results.csv)
     - combined_inference_time.csv (and batch_inference_time.csv)
     - combined_order_summary.csv
"""

import argparse
import os
import sys
import time
import json
import zipfile
import pickle
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

DEFAULT_DETECTION_MODEL = os.path.join(_DIR, "urine_iter_4_5_only_object.onnx")
if not os.path.exists(DEFAULT_DETECTION_MODEL):
    DEFAULT_DETECTION_MODEL = "/home/sigtuple/Downloads/urine_iter_4_5_only_object.onnx"

DEFAULT_CLASSIFIER_MODEL = os.path.join(_DIR, "urine_mobilenetv4_gem_msfm (1).onnx")
if not os.path.exists(DEFAULT_CLASSIFIER_MODEL):
    alt_cls = os.path.join(_DIR, "urine_mobilenetv4_gem_msfm.onnx")
    if os.path.exists(alt_cls):
        DEFAULT_CLASSIFIER_MODEL = alt_cls
    else:
        DEFAULT_CLASSIFIER_MODEL = "/home/sigtuple/DS/models/urine_convnext_v2.onnx"


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
    "unstained_rbc",
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
    "Fat",
    "UNFOCUSED_CO_DI",
    "fungal_hyphae",
    "unknown_crystals",
    "unstained_rbc",
    "Amorphous",
    "Debris",
]


def clean_col_name(name, suffix="count"):
    clean = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return f"{clean}_{suffix}" if suffix else clean


def resolve_nas_path(nas_input=None):
    """
    Resolves SMB / local mount path for sigvet_data NAS.
    Handles URI format: smb://192.168.0.253/sigvet_data/sigvet_data/
    """
    if nas_input and os.path.isdir(nas_input):
        return os.path.abspath(nas_input)

    uid = os.getuid()
    candidates = [
        f"/run/user/{uid}/gvfs/smb-share:server=192.168.0.253,share=sigvet_data/sigvet_data",
        "/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=sigvet_data/sigvet_data",
        "/mnt/nas/sigvet_data/sigvet_data",
        "/mnt/nas",
        "/Volumes/sigvet_data/sigvet_data",
        "/Volumes/sigvet_data",
        f"/run/user/{uid}/gvfs/smb-share:server=192.168.0.253,share=sigvet_data",
        "/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=sigvet_data",
    ]

    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)

    # Glob search for any GVFS mount matching 192.168.0.253 or sigvet_data
    gvfs_dirs = (
        glob(f"/run/user/{uid}/gvfs/*192.168.0.253*")
        + glob(f"/run/user/{uid}/gvfs/*sigvet_data*")
        + glob("/run/user/*/gvfs/*192.168.0.253*")
        + glob("/run/user/*/gvfs/*sigvet_data*")
    )
    for g in gvfs_dirs:
        sub = os.path.join(g, "sigvet_data")
        if os.path.isdir(sub):
            return os.path.abspath(sub)
        if os.path.isdir(g):
            return os.path.abspath(g)

    return nas_input


def find_tray_on_nas(tray_id, nas_root, device_folder=None):
    """
    Searches for tray_id (.zip file or directory) across NAS device subfolders.
    Returns (tray_path, is_zip).
    """
    clean_id = tray_id.strip()
    if clean_id.lower().endswith(".zip"):
        clean_id = clean_id[:-4]

    # 1. Direct device folder check
    if device_folder:
        dev_dir = os.path.join(nas_root, device_folder)
        if os.path.isdir(dev_dir):
            for candidate in [os.path.join(dev_dir, f"{clean_id}.zip"), os.path.join(dev_dir, clean_id)]:
                if os.path.isfile(candidate):
                    return candidate, True
                elif os.path.isdir(candidate):
                    return candidate, False

    # 2. Check directly in nas_root
    for candidate in [os.path.join(nas_root, f"{clean_id}.zip"), os.path.join(nas_root, clean_id)]:
        if os.path.isfile(candidate):
            return candidate, True
        elif os.path.isdir(candidate):
            return candidate, False

    # 3. Check all subdirectories (device folders) in nas_root
    try:
        entries = sorted(os.listdir(nas_root))
    except Exception as e:
        print(f"[Error] Failed to read NAS root {nas_root}: {e}")
        return None, False

    for entry in entries:
        if entry.startswith("."):
            continue
        sub_dir = os.path.join(nas_root, entry)
        if not os.path.isdir(sub_dir):
            continue

        candidate_zip = os.path.join(sub_dir, f"{clean_id}.zip")
        if os.path.isfile(candidate_zip):
            return candidate_zip, True

        candidate_dir = os.path.join(sub_dir, clean_id)
        if os.path.isdir(candidate_dir):
            return candidate_dir, False

    return None, False


def get_order_pkls_from_zip(zip_path):
    """
    Scans a tray ZIP archive for urine/recon_data *.pkl files.
    Returns dict: {order_id: [internal_pkl_paths]}
    """
    orders = defaultdict(list)
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".pkl"):
                continue
            base = os.path.basename(name).lower()
            if base in ("metadata.pkl", "micro_qc_data.pkl") or base.startswith("micro_qc") or base.startswith("macro"):
                continue

            parts = name.strip("/").split("/")
            lower_parts = [p.lower() for p in parts]
            if "urine" in lower_parts:
                u_idx = lower_parts.index("urine")
                if u_idx > 0 and any("recon" in p for p in lower_parts[u_idx:]):
                    order_id = parts[u_idx - 1]
                    orders[order_id].append(name)
    return dict(orders)


def get_order_pkls_from_dir(tray_dir):
    """
    Scans a tray directory on disk for urine/recon_data *.pkl files.
    Returns dict: {order_id: [full_pkl_paths]}
    """
    orders = defaultdict(list)
    for root, _, files in os.walk(tray_dir):
        lower_parts = [p.lower() for p in root.split(os.sep)]
        if "urine" in lower_parts and any("recon" in p for p in lower_parts):
            u_idx = lower_parts.index("urine")
            if u_idx > 0:
                parts = root.split(os.sep)
                order_id = parts[u_idx - 1]
                for f in sorted(files):
                    if f.lower().endswith(".pkl") and not f.lower().startswith(("metadata", "micro_qc", "macro")):
                        orders[order_id].append(os.path.join(root, f))
    return dict(orders)


def convert_pkls_to_png(pkl_list, is_zip, zip_path_or_none, dest_dir, skip_existing=True, max_fovs=None):
    """
    Extracts BestFocusedWLImage from PKLs and saves them as PNG images in dest_dir.
    Returns sorted list of saved PNG file paths.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if max_fovs and max_fovs > 0:
        pkl_list = pkl_list[:max_fovs]

    saved_images = []
    zip_obj = None
    if is_zip and zip_path_or_none:
        zip_obj = zipfile.ZipFile(zip_path_or_none, "r")

    try:
        desc = f"Converting PKL->RGB ({len(pkl_list)} FOVs)"
        for pkl_item in tqdm(pkl_list, desc=desc, leave=False):
            base_name = os.path.splitext(os.path.basename(pkl_item))[0] + ".png"
            dest_png = os.path.join(dest_dir, base_name)

            if skip_existing and os.path.isfile(dest_png) and os.path.getsize(dest_png) > 0:
                saved_images.append(dest_png)
                continue

            try:
                if is_zip:
                    with zip_obj.open(pkl_item) as f:
                        data = pickle.load(f)
                else:
                    with open(pkl_item, "rb") as f:
                        data = pickle.load(f)

                if isinstance(data, dict) and "BestFocusedWLImage" in data:
                    img = data["BestFocusedWLImage"]
                    if img is not None and getattr(img, "size", 0) > 0:
                        success = cv2.imwrite(dest_png, img)
                        if success:
                            saved_images.append(dest_png)
                        else:
                            print(f"[Warning] Failed to write image: {dest_png}")
            except Exception as e:
                print(f"[Warning] Error converting {pkl_item}: {e}")
    finally:
        if zip_obj is not None:
            zip_obj.close()

    saved_images.sort()
    return saved_images


def load_tray_ids(args):
    """
    Loads list of tray_run_ids from CLI arguments or file (.txt or .csv).
    """
    tray_ids = []
    if args.tray_ids:
        for t in args.tray_ids:
            for sub_t in t.split(","):
                sub_t = sub_t.strip()
                if sub_t:
                    tray_ids.append(sub_t)

    if args.tray_ids_file:
        file_path = args.tray_ids_file
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Tray IDs file not found: {file_path}")

        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
            matched_col = None
            for col in df.columns:
                c_clean = str(col).strip().lower().replace("-", "_").replace(" ", "_")
                if c_clean in ("tray_run_id", "tray_run_ids", "tray_id", "tray_ids", "tray", "run_id"):
                    matched_col = col
                    break
            if matched_col is None:
                matched_col = df.columns[0]
            ids = df[matched_col].dropna().astype(str).str.strip().tolist()
            tray_ids.extend([i for i in ids if i])
        else:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        tray_ids.append(line)

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for tid in tray_ids:
        clean = tid.strip()
        if clean.lower().endswith(".zip"):
            clean = clean[:-4]
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


# ──────────────────────────────────────────────
# Duplicate box removal (IOU + containment suppression)
# ──────────────────────────────────────────────
def compute_iou_containment(box_a, box_b):
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
            if iou >= iou_threshold or containment >= containment_threshold:
                suppressed[j] = True

    return kept


def get_fov_number(img_name):
    base = os.path.splitext(img_name)[0]
    parts = base.split("_")
    if len(parts) >= 1 and parts[0].isdigit():
        if "_patch_" in base:
            return base
        return parts[0]
    return base


def extract_isolated_patch_black(img, bbox, patch_size=224, pad_ratio=0.15):
    x, y, w, h = [int(round(v)) for v in bbox]
    pad_w = int(round(w * pad_ratio))
    pad_h = int(round(h * pad_ratio))

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)

    crop = img[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    max_dim = max(cw, ch)
    if max_dim > patch_size:
        scale = patch_size / float(max_dim)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ch, cw = crop.shape[:2]

    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    top = (patch_size - ch) // 2
    left = (patch_size - cw) // 2
    canvas[top:top + ch, left:left + cw] = crop
    return canvas


def process_single_order(
    images_dir,
    output_dir,
    detection_model,
    classifier,
    args,
    tray_id="",
    order_id="",
):
    """
    Processes all RGB images in images_dir for a single order.
    Saves:
        - predictions_coco.json
        - class_counts.csv
        - inference_time.csv
        - fov_counts.csv
        - overall_counts.csv
        - mean_results.csv
        - classification_result/ (patches)
    Returns: (timing_records, class_counts, fov_records, overall_rec, mean_rec)
    """
    os.makedirs(output_dir, exist_ok=True)

    classification_result_dir = os.path.join(output_dir, args.patch_dir_name)
    if args.save_patches:
        os.makedirs(classification_result_dir, exist_ok=True)

    class_counts = defaultdict(int)
    class_mapping = CLASS_MAPPING_LOOKUP[args.model_type]

    image_paths = sorted(
        glob(os.path.join(images_dir, "*.jpg"))
        + glob(os.path.join(images_dir, "*.jpeg"))
        + glob(os.path.join(images_dir, "*.png"))
    )

    if not image_paths:
        print(f"No images found in {images_dir}, skipping.")
        return [], {}, [], {}, {}

    if args.max_fovs and args.max_fovs > 0:
        image_paths = image_paths[:args.max_fovs]

    all_det_classes = list(dict.fromkeys(STANDARD_DETECTION_CLASSES + list(class_mapping.values())))
    all_final_classes = list(dict.fromkeys(STANDARD_CLASSIFIED_CLASSES + CLASSIFIER_CLASSES))

    all_categories = list(dict.fromkeys(list(class_mapping.values()) + CLASSIFIER_CLASSES))
    category_to_id = {name: i for i, name in enumerate(all_categories)}

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for name, i in category_to_id.items()],
    }

    ann_id = 1
    timing_records = []
    fov_records = []
    desc = f"Inference [{tray_id}/{order_id}]"

    for image_id, img_path in enumerate(tqdm(image_paths, desc=desc), start=1):
        t_img_start = time.perf_counter()
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            continue

        if args.scale != 1.0:
            img_input = cv2.resize(img, (0, 0), fx=args.scale, fy=args.scale)
        else:
            img_input = img.copy()

        h, w = img.shape[:2]
        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": w,
            "height": h,
        })

        t_det_start = time.perf_counter()
        result = get_sliced_prediction(
            img_input[:, :, ::-1],  # SAHI expects RGB
            detection_model,
            slice_height=args.img_size,
            slice_width=args.img_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            perform_standard_pred=False,
            verbose=0,
        )
        det_time = time.perf_counter() - t_det_start

        vis = img.copy()
        patch_count = 0
        cls_time = 0.0
        patch_save_time = 0.0
        total_detections = len(result.object_prediction_list)

        fov_raw_det_counts = defaultdict(int)
        fov_final_counts = defaultdict(int)

        # De-duplicate detections if requested
        deduped_preds = deduplicate_predictions(
            result.object_prediction_list,
            iou_threshold=args.iou_threshold,
            containment_threshold=args.containment_threshold,
        )

        for pred in deduped_preds:
            x1 = int(round(pred.bbox.minx / args.scale))
            y1 = int(round(pred.bbox.miny / args.scale))
            x2 = int(round(pred.bbox.maxx / args.scale))
            y2 = int(round(pred.bbox.maxy / args.scale))
            det_cls_name = pred.category.name
            det_conf = pred.score.value
            fov_raw_det_counts[det_cls_name] += 1
            if det_cls_name not in all_det_classes:
                all_det_classes.append(det_cls_name)

            cls_name_classify = None
            cls_conf = None
            patch = None

            # Detections of 'Object' are passed to classifier
            if det_cls_name.lower() == "object":
                t_cls_start = time.perf_counter()
                bbox = [x1, y1, x2 - x1, y2 - y1]
                patch = extract_isolated_patch_black(img, bbox, patch_size=args.patch_size)
                if patch is not None:
                    cls_name_classify, cls_conf = classifier.predict(patch)
                    cls_time += (time.perf_counter() - t_cls_start)
                    patch_count += 1
                    final_cls = cls_name_classify
                    final_conf = cls_conf
                else:
                    cls_time += (time.perf_counter() - t_cls_start)
                    final_cls = det_cls_name
                    final_conf = det_conf
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

            # Save patch class-wise
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

            # Draw bounding box
            colour = COLOUR_BY_NAME.get(final_cls, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            label = f"{final_cls} {final_conf:.2f}"

            if final_cls != "Bacteria" and det_cls_name != "Bacteria":
                cv2.putText(
                    vis,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

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

        out_annotated_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_annotated_path, vis)
        overall_time = time.perf_counter() - t_img_start

        fov_num = get_fov_number(img_name)
        rec = {
            "tray_id": tray_id,
            "order_id": order_id,
            "image_name": img_name,
            "fov_number": fov_num,
            "complete_detection_numbers": total_detections,
            "patches_for_classification": patch_count,
            "overall_time": round(overall_time, 5),
            "detection_time": round(det_time, 5),
            "classification_time": round(cls_time, 5),
            "time_to_save_patches": round(patch_save_time, 5),
        }
        timing_records.append(rec)

        has_element = any(
            cnt > 0 for cname, cnt in fov_final_counts.items()
            if cname.lower() not in ["debris", "debries"]
        )
        element_present = 1 if has_element else 0

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

    total_fovs = len(image_paths)
    element_present_fovs = sum(r["element_present"] for r in fov_records)

    # 1. Overall counts record
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

    # 2. Mean results record
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

    # Save order files
    with open(os.path.join(output_dir, "predictions_coco.json"), "w") as f:
        json.dump(coco, f, indent=2)

    pd.DataFrame(
        [{"class": cls, "count": count} for cls, count in sorted(class_counts.items())]
    ).to_csv(os.path.join(output_dir, "class_counts.csv"), index=False)

    pd.DataFrame(timing_records).to_csv(os.path.join(output_dir, args.timing_csv), index=False)
    pd.DataFrame(fov_records).to_csv(os.path.join(output_dir, args.fov_csv), index=False)
    pd.DataFrame([overall_rec]).to_csv(os.path.join(output_dir, args.overall_csv), index=False)
    pd.DataFrame([mean_rec]).to_csv(os.path.join(output_dir, args.mean_csv), index=False)

    return timing_records, class_counts, fov_records, overall_rec, mean_rec


def parse_args():
    p = argparse.ArgumentParser(description="Run SAHI Inference on Tray IDs from NAS (smb://192.168.0.253/sigvet_data/sigvet_data/)")

    # Tray selection
    p.add_argument("--tray_ids", nargs="*", default=None,
                   help="One or more tray_run_ids (space-separated or comma-separated)")
    p.add_argument("--tray_ids_file", default=None,
                   help="Path to .txt (one ID per line) or .csv containing tray_run_ids")

    # NAS options
    p.add_argument("--nas_path", default="smb://192.168.0.253/sigvet_data/sigvet_data/",
                   help="Path or URI to NAS share (default: smb://192.168.0.253/sigvet_data/sigvet_data/)")
    p.add_argument("--device_folder", default=None,
                   help="Optional specific device folder on NAS (e.g. SIGALP-N002-SV-0086, SIG-SIGVET-DC-04)")

    # Output options
    p.add_argument("--output_dir", default="./nas_inference_output",
                   help="Root directory for converted images, results, and combined CSVs")
    p.add_argument("--keep_converted_images", action="store_true", default=True,
                   help="Retain extracted RGB PNG images on disk (default: True)")
    p.add_argument("--no_keep_converted_images", dest="keep_converted_images", action="store_false")

    # Model options
    p.add_argument("--model_path", default=DEFAULT_DETECTION_MODEL,
                   help=f"Path to ONNX detection model (default: {DEFAULT_DETECTION_MODEL})")
    p.add_argument("--model_type", default="yolox", choices=["yolox", "yolov8", "yolov26"],
                   help="Detection model backend type (default: yolox)")
    p.add_argument("--cls_model_path", default=DEFAULT_CLASSIFIER_MODEL,
                   help=f"Path to ONNX classification model (default: {DEFAULT_CLASSIFIER_MODEL})")
    p.add_argument("--device", default="auto", choices=["auto", "CPU", "GPU"],
                   help="Execution provider for classifier (default: auto)")

    # Inference parameters
    p.add_argument("--patch_size", type=int, default=224, help="Classification patch size (default: 224)")
    p.add_argument("--img_size", type=int, default=416, help="Slice / detection model input size (default: 416)")
    p.add_argument("--confidence", type=float, default=0.1, help="Detection confidence threshold")
    p.add_argument("--overlap", type=float, default=0.0, help="Slice overlap ratio")
    p.add_argument("--scale", type=float, default=1.0, help="Pre-scale factor before slicing")
    p.add_argument("--iou_threshold", type=float, default=0.4, help="IOU threshold for duplicate box suppression")
    p.add_argument("--containment_threshold", type=float, default=0.5, help="Containment threshold for duplicate box suppression")

    # CSV and patch saving
    p.add_argument("--timing_csv", default="inference_time.csv", help="Order timing CSV filename")
    p.add_argument("--fov_csv", default="fov_counts.csv", help="Order FOV counts CSV filename")
    p.add_argument("--overall_csv", default="overall_counts.csv", help="Order overall counts CSV filename")
    p.add_argument("--mean_csv", default="mean_results.csv", help="Order mean results CSV filename")
    p.add_argument("--save_patches", action="store_true", default=True, help="Save cropped patches class-wise")
    p.add_argument("--no_save_patches", dest="save_patches", action="store_false")
    p.add_argument("--save_patch_classes", default="all", choices=["all", "classified_only"],
                   help="Classes to save patches for: 'all' or 'classified_only'")
    p.add_argument("--patch_dir_name", default="classification_result", help="Subdirectory for cropped patches")

    # Execution controls
    p.add_argument("--skip_existing", action="store_true", default=False,
                   help="Skip orders whose results are already completed")
    p.add_argument("--skip_existing_conversion", action="store_true", default=True,
                   help="Skip re-converting PKLs to PNG if the PNG already exists")
    p.add_argument("--max_orders", type=int, default=None, help="Maximum number of orders to process")
    p.add_argument("--max_fovs", type=int, default=None, help="Maximum number of FOVs to process per order (for testing)")

    return p.parse_args()


def main():
    args = parse_args()

    # 1. Resolve Tray IDs
    tray_ids = load_tray_ids(args)
    if not tray_ids:
        print("[Error] No tray_run_ids provided. Specify via --tray_ids or --tray_ids_file.")
        sys.exit(1)

    # 2. Resolve NAS Root
    nas_root = resolve_nas_path(args.nas_path)
    if not nas_root or not os.path.isdir(nas_root):
        print(f"[Error] Could not access NAS path: {args.nas_path}")
        print("Please ensure the SMB share is mounted at GVFS (/run/user/1000/gvfs/...) or /mnt/nas.")
        sys.exit(1)

    print(f"============================================================")
    print(f"SAHI Inference from NAS Pipeline")
    print(f"============================================================")
    print(f"NAS Root                : {nas_root}")
    print(f"Total Tray IDs Requested: {len(tray_ids)}")
    print(f"Output Directory        : {os.path.abspath(args.output_dir)}")
    print(f"Detection Model         : {args.model_type} ({args.model_path})")
    print(f"Classification Model    : {args.cls_model_path}")
    print(f"Slice Config            : {args.img_size}×{args.img_size}  overlap={args.overlap}")
    print(f"============================================================\n")

    # 3. Load Models Once
    class_mapping = CLASS_MAPPING_LOOKUP[args.model_type]
    print("Loading Detection Model...")
    detection_model = YOLOSAHIWrapper(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        category_mapping=class_mapping,
        model_type=args.model_type,
        image_size=args.img_size,
    )

    print("Loading Classification Model...")
    cls_device = "CPU"
    if args.device == "GPU":
        cls_device = "GPU"
    elif args.device == "auto":
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                cls_device = "GPU"
        except Exception:
            cls_device = "CPU"

    classifier = FasterNetClassifier(model_path=args.cls_model_path, classes=CLASSIFIER_CLASSES)
    classifier.load_model(device=cls_device)
    print(f"Classifier Device       : {cls_device}")
    print()

    # 4. Search and Process Tray IDs
    all_batch_records = []
    all_batch_fov_records = []
    all_batch_overall_records = []
    all_batch_mean_records = []
    processed_orders_count = 0
    overall_t0 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    for t_idx, tray_id in enumerate(tray_ids, start=1):
        print(f"\n[{t_idx}/{len(tray_ids)}] Locating Tray Run ID: {tray_id}")
        tray_path, is_zip = find_tray_on_nas(tray_id, nas_root, device_folder=args.device_folder)

        if not tray_path:
            print(f"  [Warning] Tray ID '{tray_id}' not found on NAS under {nas_root}.")
            continue

        print(f"  Found on NAS: {tray_path} ({'ZIP' if is_zip else 'Directory'})")

        # Discover orders and PKLs
        if is_zip:
            order_pkl_map = get_order_pkls_from_zip(tray_path)
        else:
            order_pkl_map = get_order_pkls_from_dir(tray_path)

        if not order_pkl_map:
            print(f"  [Warning] No urine/recon_data PKL files found in {tray_id}.")
            continue

        print(f"  Found {len(order_pkl_map)} order(s) inside tray:")
        for o_id, p_list in order_pkl_map.items():
            print(f"    - Order: {o_id} ({len(p_list)} PKL files)")

        # Process each order
        for order_id, pkl_files in order_pkl_map.items():
            if args.max_orders and processed_orders_count >= args.max_orders:
                print(f"\nReached max_orders limit ({args.max_orders}). Stopping batch.")
                break

            order_out_dir = os.path.join(args.output_dir, tray_id, order_id, "results")
            rgb_images_dir = os.path.join(args.output_dir, tray_id, order_id, "rgb_images")

            timing_file = os.path.join(order_out_dir, args.timing_csv)
            fov_file = os.path.join(order_out_dir, args.fov_csv)
            overall_file = os.path.join(order_out_dir, args.overall_csv)
            mean_file = os.path.join(order_out_dir, args.mean_csv)

            # Check if existing results should be skipped
            if (
                args.skip_existing
                and os.path.isfile(timing_file)
                and os.path.isfile(fov_file)
                and os.path.isfile(overall_file)
                and os.path.isfile(mean_file)
            ):
                print(f"  [Skipping] {tray_id}/{order_id} (already completed all CSVs)")
                try:
                    all_batch_records.extend(pd.read_csv(timing_file).to_dict(orient="records"))
                    all_batch_fov_records.extend(pd.read_csv(fov_file).to_dict(orient="records"))
                    all_batch_overall_records.extend(pd.read_csv(overall_file).to_dict(orient="records"))
                    all_batch_mean_records.extend(pd.read_csv(mean_file).to_dict(orient="records"))
                    processed_orders_count += 1
                except Exception as e:
                    print(f"  [Notice] Could not read existing CSVs: {e}")
                continue

            # Convert PKL files to RGB PNG
            print(f"\n  Processing Order: {order_id} (Tray: {tray_id})")
            saved_images = convert_pkls_to_png(
                pkl_files,
                is_zip=is_zip,
                zip_path_or_none=tray_path if is_zip else None,
                dest_dir=rgb_images_dir,
                skip_existing=args.skip_existing_conversion,
                max_fovs=args.max_fovs,
            )

            if not saved_images:
                print(f"  [Warning] No valid RGB images extracted for order {order_id}. Skipping inference.")
                continue

            print(f"  Converted/verified {len(saved_images)} RGB images in {rgb_images_dir}")

            # Run inference on RGB images
            records, counts, fov_recs, overall_rec, mean_rec = process_single_order(
                images_dir=rgb_images_dir,
                output_dir=order_out_dir,
                detection_model=detection_model,
                classifier=classifier,
                args=args,
                tray_id=tray_id,
                order_id=order_id,
            )

            if records:
                all_batch_records.extend(records)
                all_batch_fov_records.extend(fov_recs)
                if overall_rec:
                    all_batch_overall_records.append(overall_rec)
                if mean_rec:
                    all_batch_mean_records.append(mean_rec)
                processed_orders_count += 1

            # Save / update combined CSVs incrementally after each order
            if all_batch_records:
                for prefix in ("combined_", "batch_"):
                    pd.DataFrame(all_batch_records).to_csv(
                        os.path.join(args.output_dir, f"{prefix}{args.timing_csv}"), index=False
                    )
            if all_batch_fov_records:
                for prefix in ("combined_", "batch_"):
                    pd.DataFrame(all_batch_fov_records).to_csv(
                        os.path.join(args.output_dir, f"{prefix}{args.fov_csv}"), index=False
                    )
            if all_batch_overall_records:
                for prefix in ("combined_", "batch_"):
                    pd.DataFrame(all_batch_overall_records).to_csv(
                        os.path.join(args.output_dir, f"{prefix}{args.overall_csv}"), index=False
                    )
            if all_batch_mean_records:
                for prefix in ("combined_", "batch_"):
                    pd.DataFrame(all_batch_mean_records).to_csv(
                        os.path.join(args.output_dir, f"{prefix}{args.mean_csv}"), index=False
                    )

            # Combined Summary CSV (one row per order with counts and means)
            if all_batch_overall_records and all_batch_mean_records:
                df_overall = pd.DataFrame(all_batch_overall_records)
                df_mean = pd.DataFrame(all_batch_mean_records)
                summary_df = pd.merge(
                    df_overall,
                    df_mean,
                    on=["tray_id", "order_id", "total_fov_count", "element_present_fov_count"],
                    how="outer",
                )
                summary_df.to_csv(os.path.join(args.output_dir, "combined_order_summary.csv"), index=False)

            print(f"  Completed Order {order_id}: {len(records)} FOVs processed.")
            print(f"  Results saved to: {order_out_dir}/")

        if args.max_orders and processed_orders_count >= args.max_orders:
            break

    total_elapsed = time.time() - overall_t0
    print(f"\n============================================================")
    print(f"Batch Processing from NAS Complete!")
    print(f"Total Orders Processed : {processed_orders_count}")
    print(f"Total FOVs Processed   : {len(all_batch_records)}")
    print(f"Total Elapsed Time     : {total_elapsed:.2f}s")
    print(f"============================================================")
    if all_batch_records:
        print(f"Combined Timing CSV    : {os.path.join(args.output_dir, f'combined_{args.timing_csv}')}")
    if all_batch_fov_records:
        print(f"Combined FOV CSV       : {os.path.join(args.output_dir, f'combined_{args.fov_csv}')}")
    if all_batch_overall_records:
        print(f"Combined Overall CSV   : {os.path.join(args.output_dir, f'combined_{args.overall_csv}')}")
    if all_batch_mean_records:
        print(f"Combined Mean CSV      : {os.path.join(args.output_dir, f'combined_{args.mean_csv}')}")
        print(f"Combined Summary CSV   : {os.path.join(args.output_dir, 'combined_order_summary.csv')}")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
