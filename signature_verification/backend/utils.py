from sklearn.metrics import roc_curve, f1_score
import numpy as np
from preprocess import preprocess_signature


def calculate_youden_j_threshold(distances, labels):
    """
    Calculate the optimal threshold using Youden's J statistic.
    """
    fpr, tpr, thresholds = roc_curve(labels, distances, pos_label=1)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]

def calculate_f1_threshold(distances, labels):
    """
    Calculate the optimal threshold using F1-thresholding.
    """
    thresholds = np.linspace(0, 1, 2001)
    best_f1, best_thr = 0.0, 0.0

    for thr in thresholds:
        preds = (distances <= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr

def verify_signature(claimed_writer_id, uploaded_signature_path, reference_embeddings, model, threshold=0.5):
    """
    Verify the authenticity of a signature using the specified threshold.
    Handles both base (single embedding) and enhanced (list of dicts) models.
    """
    # Preprocess the uploaded signature
    uploaded_signature = preprocess_signature(uploaded_signature_path)

    # Generate embedding for uploaded signature
    uploaded_emb = model.predict(np.expand_dims(uploaded_signature, axis=0), verbose=0)[0].flatten()

    # Get all reference embeddings for the claimed writer
    reference_objs = reference_embeddings.get(claimed_writer_id)
    if reference_objs is None:
        return {"error": f"Writer ID {claimed_writer_id} not found."}

    # Determine if it's a base model (single np.array) or enhanced (list of dicts)
    if isinstance(reference_objs, np.ndarray):
        # Base model: single reference embedding
        distance = np.linalg.norm(reference_objs.flatten() - uploaded_emb)
    else:
        # Enhanced model: list of embeddings
        min_dist = float("inf")
        for ref in reference_objs:
            dist = np.linalg.norm(ref["embedding"] - uploaded_emb)
            if dist < min_dist:
                min_dist = dist
        distance = min_dist

    # Decision
    is_authentic = distance <= threshold

    return {
        "result": "Genuine" if is_authentic else "Forged",
        "distance": float(distance),
        "threshold": float(threshold),
        "confidence": float(1 - (distance / threshold)) if is_authentic else float(distance / threshold)
    }

