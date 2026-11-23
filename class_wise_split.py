import os
import cv2
import json
from collections import defaultdict

image_path = "/home/sigtuple/DS/YOLOX/datasets/iter_2_4_urine/train2017"
json_path = "/home/sigtuple/DS/YOLOX/datasets/iter_2_4_urine/annotation/train2017.json"

output_dir = "/home/sigtuple/DS/YOLOX/datasets/iter_2_4_urine/classification_model"
os.makedirs(output_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

# image_id -> file_name
image_id_to_file = {
    img["id"]: img["file_name"]
    for img in data["images"]
}

# category_id -> class_name
category_id_to_name = {
    cat["id"]: cat["name"]
    for cat in data["categories"]
}

class_counter = defaultdict(int)

# To make bbox width +10 and height +10
# Add 20 total width and 20 total height
pad_x = 10
pad_y = 10

for ann in data["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]

    file_name = image_id_to_file[image_id]
    class_name = category_id_to_name[category_id]

    img_file = os.path.join(image_path, file_name)

    if not os.path.exists(img_file):
        continue

    img = cv2.imread(img_file)

    if img is None:
        print(f"Could not read image: {img_file}")
        continue

    img_h, img_w = img.shape[:2]

    # COCO bbox format: [x, y, width, height]
    x, y, w, h = ann["bbox"]

    x1 = int(max(0, x - pad_x))
    y1 = int(max(0, y - pad_y))
    x2 = int(min(img_w, x + w + pad_x))
    y2 = int(min(img_h, y + h + pad_y))

    patch = img[y1:y2, x1:x2]

    if patch.size == 0:
        continue

    ph, pw = patch.shape[:2]

    # Resize if larger than 120 in either dimension
    if ph > 120 or pw > 120:
        scale = min(120 / ph, 120 / pw)
        new_w = int(pw * scale)
        new_h = int(ph * scale)

        patch = cv2.resize(
            patch,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        ph, pw = patch.shape[:2]

    # Pad to 120x120
    pad_top = (120 - ph) // 2
    pad_bottom = 120 - ph - pad_top

    pad_left = (120 - pw) // 2
    pad_right = 120 - pw - pad_left

    patch = cv2.copyMakeBorder(
        patch,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # Final safety check
    patch = cv2.resize(patch, (120, 120))

    class_folder = os.path.join(output_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

    class_counter[class_name] += 1

    base_name = os.path.splitext(file_name)[0]
    patch_name = f"{base_name}_ann_{ann['id']}.jpg"

    save_path = os.path.join(class_folder, patch_name)
    cv2.imwrite(save_path, patch)