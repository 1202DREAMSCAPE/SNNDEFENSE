import cv2
import numpy as np
import os

IMG_SHAPE = (155, 220, 1)

def compute_cnr(image: np.ndarray, debug=False) -> float:
    """
    Compute the Contrast-to-Noise Ratio (CNR) of a preprocessed grayscale image.
    Uses adaptive percentile thresholding with a fixed fallback for white backgrounds.
    """

    # Normalize to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0

    # Squeeze in case image has channel dimension
    if image.ndim > 2:
        image = image.squeeze()

    # Step 1: Adaptive threshold
    thresh_val = np.percentile(image, 30)
    signal_mask = image <= thresh_val
    background_mask = ~signal_mask

    # Step 2: Fallback if signal pixels are too few
    if np.sum(signal_mask) < 50:
        if debug: print("[DEBUG] Signal pixels too few, applying fallback threshold 0.75")
        signal_mask = image <= 0.75
        background_mask = image > 0.75

    signal_pixels = image[signal_mask]
    background_pixels = image[background_mask]

    # Step 3: Validate masks
    if len(signal_pixels) == 0 or len(background_pixels) == 0:
        if debug: print("[DEBUG] Empty signal or background pixel set.")
        return 0.0

    sigma_background = np.std(background_pixels)
    min_sigma = 0.01  # prevent division by very small values
    if sigma_background < min_sigma or np.isnan(sigma_background):
        if debug: print(f"[DEBUG] Background std too low: {sigma_background}, applying min_sigma={min_sigma}")
        sigma_background = min_sigma

    mu_signal = np.mean(signal_pixels)
    mu_background = np.mean(background_pixels)

    cnr = abs(mu_signal - mu_background) / sigma_background
    cnr = min(cnr, 100.0)  # optional: cap CNR to 100 for sanity
    return round(float(cnr), 4)



def preprocess_signature(image_path, preprocessing_type="clahe"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    if preprocessing_type == "clahe":
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

    # Replace edge count with CNR
    cnr_value = compute_cnr(resized)
    return final, cnr_value
