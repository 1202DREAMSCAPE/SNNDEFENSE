import cv2
import numpy as np
import os


IMG_SHAPE = (155, 220, 1)

def preprocess_signature(image_path, preprocessing_type="clahe"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    if preprocessing_type == "clahe":
        # Ensure the image is in uint8 and 0–255 range
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(image)
        normalized = processed / 255.0
    elif preprocessing_type == "minmax":
        image = image.astype(np.float32)
        min_val, max_val = np.min(image), np.max(image)
        if max_val - min_val == 0:
            processed = np.zeros_like(image, dtype=np.uint8)
        else:
            processed = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        normalized = processed / 255.0
    else:
        raise ValueError(f"Unknown preprocessing_type: {preprocessing_type}")

    resized = cv2.resize(normalized, (220, 155))  # (W, H)
    final = np.expand_dims(resized, axis=-1)

    edge_count = np.count_nonzero(cv2.Canny(processed, 50, 150))
    return final, edge_count

