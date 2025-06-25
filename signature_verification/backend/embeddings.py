import pickle
import numpy as np
from preprocess import preprocess_signature
import tensorflow as tf
import os
import csv

def generate_reference_embeddings(reference_signatures, model, output_path, model_type="base"):
    """
    Generate and save reference embeddings for each writer.

    Args:
        reference_signatures (dict): Mapping of writer_id -> list of signature_paths
        model (tf.keras.Model): Trained model (base or triplet-based)
        output_path (str): File path to save the embeddings (pkl)
        model_type (str): Either 'base' or 'enhanced'
    """
    reference_embeddings = {}

    for writer_id, signature_paths in reference_signatures.items():
        # print(f"[INFO] Processing Writer {writer_id} with {len(signature_paths)} signatures...")
        for signature_path in signature_paths:
            # print(f"[DEBUG] - {signature_path}")
            signature = preprocess_signature(signature_path)

            if model_type == "base":
                embedding = model.predict(np.expand_dims(signature, axis=0), verbose=0)[0].flatten()
            elif model_type == "enhanced":
                anchor = np.expand_dims(signature, axis=0)
                triplet_out = model.predict([anchor, anchor, anchor], verbose=0)
                embedding = triplet_out[0][0]
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            reference_embeddings.setdefault(writer_id, []).append({
                "embedding": embedding,
                "path": signature_path
            })

    # Save as pickle
    with open(output_path, "wb") as f:
        pickle.dump(reference_embeddings, f)
    print(f"[✓] Saved reference embeddings to: {output_path}")

    # Save writer summary to CSV
    csv_log = output_path.replace(".pkl", "_log.csv")
    with open(csv_log, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["writer_id", "num_embeddings"])
        for writer_id, entries in reference_embeddings.items():
            writer.writerow([writer_id, len(entries)])
    print(f"[✓] Logged writer counts to: {csv_log}")
