import cv2
import numpy as np
import os

IMG_SHAPE = (155, 220, 1)

def preprocess_signature(image_path, preprocessing_type="clahe", save_processed=False, save_name=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    if preprocessing_type == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(image)
    elif preprocessing_type == "minmax":
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val == 0:
            processed = np.zeros_like(image, dtype=np.uint8)
        else:
            processed = (image - min_val) / (max_val - min_val)
            processed = (processed * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown preprocessing_type: {preprocessing_type}")

    # Save for frontend preview if needed
    if save_processed and save_name:
        os.makedirs("static/temp", exist_ok=True)
        cv2.imwrite(os.path.join("static/temp", save_name), processed)

    resized = cv2.resize(processed, (IMG_SHAPE[1], IMG_SHAPE[0]))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=-1)
