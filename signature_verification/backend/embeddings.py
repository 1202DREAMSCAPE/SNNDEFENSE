import pickle
import numpy as np
from preprocess import preprocess_signature
import tensorflow as tf

def generate_reference_embeddings(reference_signatures, model, output_path, model_type="base"):
    """
    Generate embeddings for reference signatures using the specified model.
    
    Args:
        reference_signatures (dict): Dictionary mapping writer IDs to signature file paths.
        model (tf.keras.Model): The model to use for generating embeddings.
        output_path (str): Path to save the generated embeddings.
        model_type (str): Type of model ("base" or "enhanced").
    """
    reference_embeddings = {}
    for writer_id, signature_path in reference_signatures.items():
        signature = preprocess_signature(signature_path)

        if model_type == "base":
            # Base model expects a single input tensor
            reference_embeddings[writer_id] = model.predict(np.expand_dims(signature, axis=0))
        elif model_type == "enhanced":
            # Enhanced model expects three input tensors: anchor, positive, and negative
            anchor = np.expand_dims(signature, axis=0)  # Example: Use the same signature for all inputs
            positive = np.expand_dims(signature, axis=0)  # Replace with actual positive sample
            negative = np.expand_dims(signature, axis=0)  # Replace with actual negative sample
            inputs = [anchor, positive, negative]
            reference_embeddings[writer_id] = model.predict(inputs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # Save embeddings to file
    with open(output_path, "wb") as f:
        pickle.dump(reference_embeddings, f)