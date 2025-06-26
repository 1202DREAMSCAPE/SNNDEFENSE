import os
import csv
import pickle
import numpy as np
from preprocess import preprocess_signature
import tensorflow as tf

def generate_reference_embeddings(reference_signatures, model, output_path, model_type="base"):
    """
    Generate and save reference embeddings for each writer.

    Args:
        reference_signatures (dict): Mapping of writer_id -> list of signature image paths
        model (tf.keras.Model): Trained model (either base or enhanced)
        output_path (str): Path to save the pickle file
        model_type (str): 'base' or 'enhanced'
    """
    reference_embeddings = {}

    for writer_id, signature_paths in reference_signatures.items():
        for signature_path in signature_paths:
            try:
                signature = preprocess_signature(
                    signature_path,
                    preprocessing_type="minmax" if model_type == "base" else "clahe"
                )
                if model_type == "base":
                    embedding = model.predict(np.expand_dims(signature, axis=0), verbose=0)[0].flatten()
                elif model_type == "enhanced":
                    anchor = np.expand_dims(signature, axis=0)
                    embedding = model.predict([anchor, anchor, anchor], verbose=0)[0][0]
                else:
                    raise ValueError(f"[ERROR] Unsupported model_type: {model_type}")

                reference_embeddings.setdefault(writer_id, []).append({
                    "embedding": embedding,
                    "path": signature_path
                })

            except Exception as e:
                print(f"[WARN] Failed to process {signature_path}: {e}")

    # --- Save as Pickle ---
    with open(output_path, "wb") as f:
        pickle.dump(reference_embeddings, f)
    print(f"[✓] Saved reference embeddings to: {output_path}")

    # --- Log Writer Embedding Counts as CSV ---
    csv_log = output_path.replace(".pkl", "_log.csv")
    with open(csv_log, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["writer_id", "num_embeddings"])
        for writer_id, entries in reference_embeddings.items():
            writer.writerow([writer_id, len(entries)])
    print(f"[✓] Logged writer counts to: {csv_log}")
