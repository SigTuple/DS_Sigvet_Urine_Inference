import os
import cv2
import json
import numpy as np
import onnxruntime as ort

# =====================================================
# CONFIG
# =====================================================

ONNX_PATH = "/home/sigtuple/Downloads/urine_iter_4_yolov11_all_data.onnx"

INPUT_DIR = "/Volumes/SigVet_2/june_3_416/images/train2017_416"

OUTPUT_DIR = "/Volumes/SigVet_2/june_3_416/images/inference_results"

IMG_SIZE = 416
CONF_THRES = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# CLASS-WISE CONFIDENCE THRESHOLDS
# =====================================================
class_confidence_thresholds = {
    # Add class IDs and their thresholds as needed
}

# ─────────────────────────────────────────────────────────────────────────────
# NMS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# IoU threshold: boxes with IoU > this value are considered overlapping.
# Lower  → more aggressive suppression (removes more boxes)
# Higher → more permissive (keeps more boxes)
NMS_IOU_THRESHOLD = 0.45

# Containment threshold: if the smaller box's area that overlaps with a larger
# box exceeds this fraction of the smaller box's own area, it is suppressed.
# 0.80 means "80 % of the small box is inside the big box → remove it".
CONTAINMENT_THRESHOLD = 0.90
# ─────────────────────────────────────────────────────────────────────────────

# =====================================================
# CLASS NAMES
# =====================================================

class_names = {
    0: "Ammonium_Biurate",
    1: "Amorphous",
    2: "Bacteria",
    3: "Bilirubin",
    4: "Ca_C",
    5: "CO_DI",
    6: "CO_DI_CLUMP",
    7: "CO_MO",
    8: "Cys",
    9: "Debris",
    10: "doubtful_rbc",
    11: "doubtful_wbc",
    12: "doubtful_yeast",
    13: "EP_SQ",
    14: "EP_TR",
    15: "Fat",
    16: "fungal_hyphae",
    17: "Granular_cast",
    18: "RBC",
    19: "Sperm",
    20: "Struvite",
    21: "UNFOCUSED_CO_DI",
    22: "unknown",
    23: "unstained_rbc",
    24: "WBC",
    25: "WBC_Clump",
    26: "WBC_cast",
    27: "Yeast",
}
# =====================================================
# COLORS
# =====================================================

color_mapping = {
    i: tuple(np.random.randint(0, 255, 3).tolist())
    for i in range(len(class_names))
}

# =====================================================
# LETTERBOX
# =====================================================

def letterbox(img, new_shape=416, color=(114, 114, 114)):

    shape = img.shape[:2]

    r = min(
        new_shape / shape[0],
        new_shape / shape[1]
    )

    new_unpad = (
        int(round(shape[1] * r)),
        int(round(shape[0] * r))
    )

    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]

    dw /= 2
    dh /= 2

    img = cv2.resize(
        img,
        new_unpad,
        interpolation=cv2.INTER_LINEAR
    )

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img, r, (dw, dh)

# =====================================================
# PREPROCESS
# =====================================================

def preprocess(img):

    img, ratio, dwdh = letterbox(
        img,
        IMG_SIZE
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # img = img.astype(
    #     np.float32
    # ) / 255.0

    # img = np.transpose(
    #     img,
    #     (2, 0, 1)
    # )

    # img = np.expand_dims(
    #     img,
    #     axis=0
    # )

    return img, ratio, dwdh

# =====================================================
# SCALE BOXES
# =====================================================

def scale_boxes(boxes, ratio, dwdh, shape):

    dw, dh = dwdh

    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh

    boxes /= ratio

    h, w = shape

    boxes[:, 0] = np.clip(
        boxes[:, 0], 0, w
    )

    boxes[:, 2] = np.clip(
        boxes[:, 2], 0, w
    )

    boxes[:, 1] = np.clip(
        boxes[:, 1], 0, h
    )

    boxes[:, 3] = np.clip(
        boxes[:, 3], 0, h
    )

    return boxes

# =====================================================
# ION & CONTAINMENT COMPUTATION
# =====================================================

def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
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

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_containment(box_small, box_large):
    """
    Return the fraction of box_small's area that is covered by box_large.
    Both boxes in [x1, y1, x2, y2] format.
    """
    xs1, ys1, xs2, ys2 = box_small
    xl1, yl1, xl2, yl2 = box_large

    inter_x1 = max(xs1, xl1)
    inter_y1 = max(ys1, yl1)
    inter_x2 = min(xs2, xl2)
    inter_y2 = min(ys2, yl2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    small_area = max(0, xs2 - xs1) * max(0, ys2 - ys1)
    if small_area <= 0:
        return 0.0
    return inter_area / small_area

# =====================================================
# NMS & CONTAINMENT SUPPRESSION
# =====================================================

def apply_nms_and_containment_suppression(predictions):
    """
    Given a list of predictions - each a dict with keys:
        'bbox_xyxy'  : [x1, y1, x2, y2]
        'score'      : float
        'class_id'   : int

    Returns the filtered list after:
      1. Cross-class IoU-based NMS  (lower-score box removed on high overlap)
      2. Containment suppression    (smaller box removed when mostly inside bigger box)
    """
    if not predictions:
        return predictions

    # Sort descending by score so higher-confidence boxes are processed first
    predictions = sorted(predictions, key=lambda p: p['score'], reverse=True)

    suppressed = [False] * len(predictions)

    for i in range(len(predictions)):
        if suppressed[i]:
            continue
        box_i = predictions[i]['bbox_xyxy']
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])

        for j in range(i + 1, len(predictions)):
            if suppressed[j]:
                continue
            box_j = predictions[j]['bbox_xyxy']
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

            # ── 1. IoU-based NMS (cross-class) ──────────────────────────────
            iou = compute_iou(box_i, box_j)
            if iou > NMS_IOU_THRESHOLD:
                # box_j has lower score (list is sorted) → suppress it
                suppressed[j] = True
                continue  # No need to check containment if already suppressed

            # ── 2. Containment suppression ───────────────────────────────────
            if area_i >= area_j:
                # box_i is larger; check how much of box_j is inside box_i
                containment = compute_containment(box_j, box_i)
                if containment > CONTAINMENT_THRESHOLD:
                    suppressed[j] = True
            else:
                # box_j is larger; check how much of box_i is inside box_j
                containment = compute_containment(box_i, box_j)
                if containment > CONTAINMENT_THRESHOLD:
                    suppressed[i] = True
                    break  # box_i is suppressed; move to next i

    return [p for idx, p in enumerate(predictions) if not suppressed[idx]]

# =====================================================
# COCO SAVE
# =====================================================

def save_prediction_to_coco_json(
    image_data,
    image_dir,
    output_json_path
):

    coco_json = {
        "info": {
            "description": "YOLO26 Predictions"
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    for class_id, class_name in class_names.items():

        coco_json["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "urine"
        })

    annotation_id = 1
    image_id = 1

    for img_name, data in image_data.items():

        img_path = os.path.join(
            image_dir,
            img_name
        )

        img = cv2.imread(img_path)

        if img is None:
            continue

        height, width = img.shape[:2]

        coco_json["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        for box in data["boxes"]:

            x, y, w, h, category_id = box

            coco_json["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [
                    float(x),
                    float(y),
                    float(w),
                    float(h)
                ],
                "area": float(w * h),
                "segmentation": [],
                "iscrowd": 0
            })

            annotation_id += 1

        image_id += 1

    with open(output_json_path, "w") as f:

        json.dump(
            coco_json,
            f,
            indent=4
        )

    print(
        f"Saved COCO JSON to {output_json_path}"
    )

    print(
        f"Images: {len(coco_json['images'])}"
    )

    print(
        f"Annotations: {len(coco_json['annotations'])}"
    )

# =====================================================
# LOAD MODEL
# =====================================================

# session = ort.InferenceSession(
#     ONNX_PATH,
#     providers=["CPUExecutionProvider"]
# )

net = cv2.dnn.readNet(ONNX_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# print(
#     "Input Shape:",
#     session.get_inputs()[0].shape
# )

# =====================================================
# INFERENCE
# =====================================================

image_data = {}

for image_name in os.listdir(INPUT_DIR):

    if not image_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image_path = os.path.join(
        INPUT_DIR,
        image_name
    )

    image = cv2.imread(
        image_path
    )

    if image is None:
        continue

    original = image.copy()

    inp, ratio, dwdh = preprocess(
        image
    )

    print(inp.shape)

    blog = cv2.dnn.blobFromImage(inp,1/255.0,(416,416),swapRB=False,crop=False)
    net.setInput(blog)
    outputs = net.forward()
    print(outputs)
    break

    preds = outputs[0][0]

    if len(preds) == 0:
        continue

    boxes = preds[:, :4]
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # ────────────────────────────────────────────────────────────────────────
    # APPLY CLASS-WISE CONFIDENCE THRESHOLDS
    # ────────────────────────────────────────────────────────────────────────
    mask = np.ones(len(scores), dtype=bool)
    for idx, class_id in enumerate(class_ids):
        class_threshold = class_confidence_thresholds.get(class_id, CONF_THRES)
        if scores[idx] < class_threshold:
            mask[idx] = False

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes):

        boxes = scale_boxes(
            boxes,
            ratio,
            dwdh,
            original.shape[:2]
        )

        # ────────────────────────────────────────────────────────────────────
        # COLLECT RAW PREDICTIONS FOR NMS & CONTAINMENT SUPPRESSION
        # ────────────────────────────────────────────────────────────────────
        raw_predictions = []
        for idx in range(len(boxes)):
            raw_predictions.append({
                'bbox_xyxy': boxes[idx],
                'score': scores[idx],
                'class_id': int(class_ids[idx])
            })

        # ────────────────────────────────────────────────────────────────────
        # APPLY NMS & CONTAINMENT SUPPRESSION
        # ────────────────────────────────────────────────────────────────────
        filtered_predictions = apply_nms_and_containment_suppression(raw_predictions)

        print(
            f"{image_name}: {len(raw_predictions)} raw → {len(filtered_predictions)} after suppression"
        )

        # Update boxes, scores, and class_ids with filtered predictions
        if filtered_predictions:
            boxes = np.array([p['bbox_xyxy'] for p in filtered_predictions])
            scores = np.array([p['score'] for p in filtered_predictions])
            class_ids = np.array([p['class_id'] for p in filtered_predictions])
        else:
            boxes = np.array([])
            scores = np.array([])
            class_ids = np.array([])

    image_data[image_name] = {
        "boxes": []
    }

    for box, score, cls in zip(
        boxes,
        scores,
        class_ids
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        class_name = class_names.get(
            int(cls),
            str(cls)
        )

        color = color_mapping.get(
            int(cls),
            (255, 255, 255)
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            f"{class_name} {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

        image_data[image_name]["boxes"].append([
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            int(cls)
        ])

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            image_name
        ),
        image
    )

# =====================================================
# SAVE COCO JSON
# =====================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "predictions_coco.json"
)

save_prediction_to_coco_json(
    image_data=image_data,
    image_dir=INPUT_DIR,
    output_json_path=json_path
)

print("Done.")