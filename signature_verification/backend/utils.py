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

def verify_signature(
    claimed_writer_id,
    reference_embeddings,
    model,
    model_type="enhanced",
    uploaded_signature_path=None,
    uploaded_emb=None,
    threshold=None
):
    """
    Verifies a signature with global writer comparison and rejection classification.
    Returns result, distance, threshold, closest_writer, and rejection_type.
    """

    # Set threshold if not provided
    if threshold is None:
        threshold = 0.827 if model_type == "enhanced" else 0.4982339

    # Step 1: Generate embedding if not provided
    if uploaded_emb is None:
        if uploaded_signature_path is None:
            raise ValueError("Must provide uploaded_signature_path or uploaded_emb.")
        img, _ = preprocess_signature(
            uploaded_signature_path,
            preprocessing_type="clahe" if model_type == "enhanced" else "minmax"
        )
        raw_emb = model.predict(np.expand_dims(img, axis=0), verbose=0)[0].flatten()
        uploaded_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-10)

    # Step 2: Find closest writer (across all)
    min_global_dist = float("inf")
    closest_writer = None

    for writer, refs in reference_embeddings.items():
        for ref in refs:
            ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
            dist = np.linalg.norm(uploaded_emb - ref_emb)
            if dist < min_global_dist:
                min_global_dist = dist
                closest_writer = writer

    # Step 3: Find min distance to claimed_writer_id
    min_claimed_dist = float("inf")
    if claimed_writer_id in reference_embeddings:
        for ref in reference_embeddings[claimed_writer_id]:
            ref_emb = ref["embedding"] / (np.linalg.norm(ref["embedding"]) + 1e-10)
            dist = np.linalg.norm(uploaded_emb - ref_emb)
            if dist < min_claimed_dist:
                min_claimed_dist = dist

    else:
        # If claimed writer has no references
        min_claimed_dist = float("inf")

    # Step 4: Apply threshold logic
    distance = min_claimed_dist
    is_authentic = (closest_writer == claimed_writer_id) and (distance <= threshold)

    # Step 5: Determine rejection classification
    if closest_writer == claimed_writer_id:
        rejection_type = "true_accept" if distance <= threshold else "false_rejection"
    else:
        rejection_type = "false_acceptance" if distance <= threshold else "true_reject"

    return {
        "result": "Genuine" if is_authentic else "Forged",
        "distance": float(round(distance, 4)),
        "threshold": float(threshold),
        "closest_writer": closest_writer,
        "rejection_type": rejection_type
    }
