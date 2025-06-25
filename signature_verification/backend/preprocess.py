import cv2
import numpy as np

IMG_SHAPE = (155, 220, 1)

def preprocess_signature(image_path, preprocessing_type="clahe"):
    """
    Preprocess the signature image based on the specified preprocessing type.
    - preprocessing_type: "clahe" for CLAHE preprocessing (enhanced model),
                          "minmax" for MinMax normalization (base model).
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if preprocessing_type == "clahe":
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    elif preprocessing_type == "minmax":
        # Apply MinMax normalization
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)  # Normalize to [0, 1]
        image = (image * 255).astype(np.uint8)  # Scale back to [0, 255] for consistency

    # Resize to match the input shape
    image = cv2.resize(image, (IMG_SHAPE[1], IMG_SHAPE[0]))
    
    # Normalize the image (final normalization to [0, 1])
    image = image / 255.0
    
    # Expand dimensions for model input
    return np.expand_dims(image, axis=-1)