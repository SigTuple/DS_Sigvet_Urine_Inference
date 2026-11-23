import os
import glob
import pickle
import cv2

input_dir = "/media/sigtuple/SigVet_DS/left_out/f7c475bf-ffd9-48f7-8a21-1c29e73d2a08/d29a6181-b2eb-4750-8f66-301d963e9b01/urine/Recon_data/"

output_dir = "/media/sigtuple/SigVet_DS/left_out/f7c475bf-ffd9-48f7-8a21-1c29e73d2a08/d29a6181-b2eb-4750-8f66-301d963e9b01/"

os.makedirs(output_dir, exist_ok=True)

pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

for path in pkl_files:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        image = data["BestFocusedWLImage"]

        image_name = os.path.basename(path).replace(".pkl", ".png")
        output_path = os.path.join(output_dir, image_name)

        success = cv2.imwrite(output_path, image)

        if success:
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to save: {path}")

    except Exception as e:
        print(f"Error processing {path}: {e}")

print(f"\nTotal PKL files found: {len(pkl_files)}")