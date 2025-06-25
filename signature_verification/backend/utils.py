from sklearn.metrics import roc_curve, f1_score
import numpy as np

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

def verify_signature(claimed_writer_id, uploaded_signature_path, reference_embeddings, model, threshold_type="youden", threshold=0.5):
    """
    Verify the authenticity of a signature using the specified thresholding method.
    - threshold_type: "youden" for Youden's J statistic (base model), "f1" for F1-thresholding (enhanced model).
    """
    # Preprocess the uploaded signature
    uploaded_signature = preprocess_signature(uploaded_signature_path)
    
    # Retrieve the reference embedding for the claimed writer
    reference_emb = reference_embeddings.get(claimed_writer_id)
    if reference_emb is None:
        return {"error": f"Writer ID {claimed_writer_id} not found."}
    
    # Generate embedding for the uploaded signature
    uploaded_emb = model.predict(np.expand_dims(uploaded_signature, axis=0))
    
    # Calculate the distance between embeddings
    distance = np.linalg.norm(reference_emb - uploaded_emb)
    
    # Dynamically calculate the threshold if needed
    if threshold_type == "youden":
        threshold = calculate_youden_j_threshold(distances, labels)  # Replace `distances` and `labels` with actual data
    elif threshold_type == "f1":
        threshold = calculate_f1_threshold(distances, labels)  # Replace `distances` and `labels` with actual data
    
    # Compare distance against threshold
    is_authentic = distance <= threshold
    
    return {
        "result": "Genuine" if is_authentic else "Forged",
        "distance": distance,
        "threshold": threshold,
        "confidence": 1 - (distance / threshold) if is_authentic else distance / threshold
    }