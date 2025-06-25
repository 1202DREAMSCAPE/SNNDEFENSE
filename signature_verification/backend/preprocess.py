import cv2
import numpy as np

IMG_SHAPE = (155, 220, 1)

def preprocess_signature(image_path, preprocessing_type="clahe"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    if preprocessing_type == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    elif preprocessing_type == "minmax":
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val == 0:
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            image = (image - min_val) / (max_val - min_val)
            image = (image * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown preprocessing_type: {preprocessing_type}")

    image = cv2.resize(image, (IMG_SHAPE[1], IMG_SHAPE[0]))
    image = image / 255.0
    return np.expand_dims(image, axis=-1)
