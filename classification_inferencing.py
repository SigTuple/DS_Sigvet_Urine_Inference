import numpy as np
import cv2
import torch
from collections import Counter
from onnx_model_loader import *
from scipy.special import softmax




import os

_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_DIR, "efficientnet_v2_urine_classification.onnx")
class_names =  ['CO_DI', 'Fat', 'RBC', 'WBC','Yeast','Uric acid']

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

MACRO_CLASSES = [
    "Debris",
    "EP_SQ",
    "EP_TR",
    "fungal_hyphae",
    "Sperm",
    "Struvite",
]

MICRO_CLASSES = [
    "CO_DI",
    "CO_DI_CLUMP",
    "Debris",
    "EP_TR",
    "RBC",
    "WBC",
    "Yeast",
    "small_sperm",
    "small_struvite",
    "unknown_crystals",
]

MICRO_METADATA_MEAN = np.array([39.7208, 40.0455, 1.0283, 1695.2015], dtype=np.float32)
MICRO_METADATA_STD  = np.array([11.5513, 11.7379, 0.2207, 901.9366], dtype=np.float32)
MICRO_METADATA_EPS  = 1e-7


def normalize_box_metadata(height, width):
    """
    Normalizes metadata features [h, w, aspect_ratio, area] using:
        x_norm = (x - mean) / (std + eps)
    where:
        mean = [39.7208, 40.0455, 1.0283, 1695.2015]
        std  = [11.5513, 11.7379, 0.2207, 901.9366]
        eps  = 1e-7
        aspect_ratio = w / max(h, 1e-6)
        area = w * h
    """
    h = float(height)
    w = float(width)
    ar = w / max(h, 1e-6)
    area = w * h
    raw = np.array([h, w, ar, area], dtype=np.float32)
    norm = (raw - MICRO_METADATA_MEAN) / (MICRO_METADATA_STD + MICRO_METADATA_EPS)
    return norm


FASTERNET_MODEL_PATH = "/home/sigtuple/Downloads/urine_fasternet_t0.onnx"
FASTERNET_CLASSES = CLASSIFIER_CLASSES


class FasterNetClassifier:
    def __init__(self, model_path=FASTERNET_MODEL_PATH, classes=None) -> None:
        self.model = ONNX_Model_Loader()
        self.model_path = model_path
        self.classes = classes if classes is not None else FASTERNET_CLASSES

    def load_model(self, model_path=None, device='CPU'):
        if model_path is None:
            model_path = self.model_path
        self.model.load_model(model_path, device=device)

    def pre_process(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if image.ndim == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            chw = np.transpose(rgb.astype(np.float32), (2, 0, 1))
            return np.expand_dims(chw, axis=0)
        elif image.ndim == 4:
            batch = []
            for img in image:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch.append(np.transpose(rgb.astype(np.float32), (2, 0, 1)))
            return np.ascontiguousarray(batch, dtype=np.float32)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

    def predict(self, image, metadata=None):
        tensor = self.pre_process(image)
        input_names = self.model.get_inputs()
        inputs = {input_names[0]: tensor}
        if len(input_names) > 1:
            if metadata is not None:
                if not isinstance(metadata, np.ndarray):
                    metadata = np.array(metadata, dtype=np.float32)
                if metadata.ndim == 1:
                    metadata = np.expand_dims(metadata, axis=0)
                inputs[input_names[1]] = metadata.astype(np.float32)
            else:
                batch_size = tensor.shape[0]
                inputs[input_names[1]] = np.zeros((batch_size, 4), dtype=np.float32)

        outputs = self.model.run_inference(inputs)
        probs = softmax(outputs[0], axis=1)
        top_indices = np.argmax(probs, axis=1)
        top_confs = np.max(probs, axis=1)

        results = [(self.classes[idx], float(conf)) for idx, conf in zip(top_indices, top_confs)]
        if isinstance(image, list):
            return results
        if isinstance(image, np.ndarray) and image.ndim == 4:
            return results
        return results[0]

    def run(self, image, metadata=None):
        tensor = self.pre_process(image)
        input_names = self.model.get_inputs()
        inputs = {input_names[0]: tensor}
        if len(input_names) > 1:
            if metadata is not None:
                if not isinstance(metadata, np.ndarray):
                    metadata = np.array(metadata, dtype=np.float32)
                if metadata.ndim == 1:
                    metadata = np.expand_dims(metadata, axis=0)
                inputs[input_names[1]] = metadata.astype(np.float32)
            else:
                batch_size = tensor.shape[0]
                inputs[input_names[1]] = np.zeros((batch_size, 4), dtype=np.float32)

        outputs = self.model.run_inference(inputs)
        probs = softmax(outputs[0], axis=1)
        classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return classes, confidences

    def terminate(self):
        self.unload_model()

    def unload_model(self):
        if self.model:
            self.model.unload_model()




class WBC_Classifier():

    def __init__(self) -> None:
        self.model = ONNX_Model_Loader()
        self.model_path = model_path
        self.classes = class_names

    def load_model(self, model_path=None, device='CPU'):
        if model_path is None:
            model_path = self.model_path
        self.model.load_model(model_path, device=device)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def pre_process(self, image):
       
        # image = image / 255.0
        torch_tensor = torch.from_numpy(np.array(image)).to(torch.float32)
        torch_tensor = torch_tensor.permute(2, 0, 1)  # Rearrange dimensions (C, H, W)
        torch_tensor = torch_tensor.unsqueeze(0)  # Add batch dimension

        return torch_tensor

    def run(self, image):
        torch_tensor = self.pre_process(image)
        input_names = self.model.get_inputs()
        inputs = {input_names[0]: self.to_numpy(torch_tensor)}
        predictions = self.model.run_inference(inputs)[0]
        # predictions = np.squeeze(predictions)
        probs = softmax(predictions, axis=1)
        classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        # classes = np.argmax(predictions, axis=1)
        # confidences = np.max(predictions, axis=1)
        return classes , confidences

    def terminate(self):
        self.model = None
        self.model.unload_model()

    def unload_model(self):
        self.model.unload_model()