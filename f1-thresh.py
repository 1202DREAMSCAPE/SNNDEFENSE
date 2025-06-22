import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.utils import register_keras_serializable
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
import seaborn as sns
import umap
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from SignatureDataGenerator import SignatureDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import csv
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
from tensorflow.keras.layers import Lambda
import sys
run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

np.random.seed(1337)
random.seed(1337)
tf.random.set_seed(1337)

@register_keras_serializable()
def create_base_network(input_shape):
    """
    Base CNN to extract 128-D feature embeddings from input images.
    Mirrors a deep architecture (inspired by SigNet/AlexNet).
    """
    model = Sequential([
        Input(shape=input_shape),
        layers.Conv2D(96, (11,11), activation='relu', strides=(4,4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        layers.ZeroPadding2D((2,2)),
        layers.Conv2D(256, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        layers.Dropout(0.3),

        layers.ZeroPadding2D((1,1)),
        layers.Conv2D(384, (3,3), activation='relu'),
        layers.ZeroPadding2D((1,1)),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0005))  # Final embedding
    ], name="base_network")
    return model

def build_siamese_network(input_shape):
    """
    Siamese network using no explicit distance metric.
    Combines embeddings through dense layers and uses softmax for classification.
    Aligned with the 2024 study by Nasr et al.
    """
    #call the signet architecture to create the base CNN
    base_network = create_base_network(input_shape)

    # Two inputs: image A and image B
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Step 2a + 2b: Pass both inputs through the same CNN base
    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    # Use Lambda layer for TensorFlow operations
    distance = Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_a, encoded_b])
    
    # Modified classifier head
    x = layers.Dense(64, activation='relu')(distance)
    output = layers.Dense(2, activation='softmax')(x)

    # Full Siamese model
    model = Model(inputs=[input_a, input_b], outputs=output)
    return model

def evaluate_classification_metrics(
    y_true,
    y_pred_probs,
    dataset_name=None,
    output_dir="outputs/f1threshold"
):
    scores = y_pred_probs[:, 1]     
    best_f1, best_thr = 0.0, 0.0
    thresholds = np.linspace(0, 1, 2001)
    f1_curve = []

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        f1_curve.append(f1)

        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"📌 Optimal F1 threshold = {best_thr:.4f}  |  Best F1 = {best_f1:.4f}")

    y_pred = (scores >= best_thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)

    #metric calculations
    acc  = accuracy_score(y_true, y_pred)
    try:
        auc  = roc_auc_score(y_true, scores)
    except ValueError:
        auc  = None

    # ------------------------------------------------------------------
    # Plot F1-curve and FAR / FRR bar
    # ------------------------------------------------------------------
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # -- F1 vs threshold curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_curve, color="darkorange")
    plt.axvline(best_thr, color="green", ls="--",
                label=f"Best F1 = {best_f1:.4f} @ {best_thr:.4f}")
    plt.title("F1-score vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_run{run_id}_f1_curve.png"))
    plt.close()

    # -- FAR / FRR bar chart
    plt.figure(figsize=(4, 5))
    bars = plt.bar(["FAR", "FRR"], [far, frr], color=["red", "blue"])
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{bar.get_height():.4f}", ha="center", va="bottom")
    plt.ylim(0, 1); plt.ylabel("Rate"); plt.title("FAR & FRR @ F1-Optimal Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_run{run_id}_far_frr.png"))
    plt.close()

    # -- Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Genuine", "Forged"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix @ Threshold {best_thr:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_run{run_id}_confmat.png"))
    plt.close()

    if dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{dataset_name}_run{run_id}_metrics.txt"), "w") as f:
            f.write("Evaluation Metrics\n")
            f.write("=" * 30 + "\n")
            f.write(f"Accuracy   : {acc:.4f}\n")
            f.write(f"F1-score   : {best_f1:.4f}\n")
            f.write(f"ROC-AUC    : {auc:.4f}\n" if auc else "ROC-AUC    : N/A\n")
            f.write(f"FAR        : {far:.4f}\n")
            f.write(f"FRR        : {frr:.4f}\n")
            f.write(f"Threshold  : {best_thr:.4f}\n")
    # ------------------------------------------------------------------
    return {
        "accuracy": acc,
        "f1_score": best_f1,
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "f1_threshold": best_thr,
    }

# Parameters
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)  

datasets = {
    "CEDAR": {
        "path": "Dataset/CEDAR",
        "train_writers": list(range(260, 300)),
        "test_writers": list(range(300, 315))
    },
     "BHSig260_Bengali": {
         "path": "Dataset/BHSig260_Bengali",
         "train_writers": list(range(1, 71)),
         "test_writers": list(range(71, 101))
     },
     "BHSig260_Hindi": {
         "path": "Dataset/BHSig260_Hindi",
         "train_writers": list(range(101, 191)),
         "test_writers": list(range(191, 260))
     }
}

os.makedirs("outputs/f1threshold", exist_ok=True)
# Define the path for the results CSV file
results_csv_path = "outputs/f1threshold/results.csv"

# Ensure the CSV file has a header if it doesn't exist
if not os.path.exists(results_csv_path):
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Accuracy", "F1 Score", "ROC AUC", "FAR", "FRR", "F1 Threshold"])

results = []

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing Siamese Model for Dataset: {dataset_name}")
    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )
    train_pairs, train_labels = generator.generate_pairs(split='train')
    train_labels = np.array(train_labels).astype(np.int32)

    # Prepare image pair arrays
    train_img1 = np.array([pair[0] for pair in train_pairs])
    train_img2 = np.array([pair[1] for pair in train_pairs])

    # Build softmax-based Siamese model
    model = build_siamese_network(IMG_SHAPE)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )

    # ========== Training ==========
    start_time = time.time()
    history = model.fit(
        [train_img1, train_img2], train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2
    )

    # Save model
    model_save_path = f"models/{dataset_name}_run{run_id}_f1_model.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"model saved to: {model_save_path}")
    print(f"⏱ Training completed in {time.time() - start_time:.2f} seconds")

    # ========== Evaluation on SOP3 (unseen writers) ==========
    print(f"\n🔍 Running Evaluation for {dataset_name}")
    test_pairs, test_labels = generator.generate_pairs(split='test', use_raw=True)
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    print("Label Distribution:", dict(zip(*np.unique(test_labels, return_counts=True))))

    if len(np.unique(test_labels)) < 2:
        print("⚠ Skipping evaluation — only one class present.")
    else:
        y_pred_probs = model.predict([test_img1, test_img2], batch_size=128)
        metrics = evaluate_classification_metrics(test_labels, y_pred_probs, dataset_name=dataset_name)
        results.append((dataset_name, metrics))
        print(f"✅ Evaluation Complete for {dataset_name}")

                # Append results to the CSV file
        with open(results_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset_name,
                metrics["accuracy"],
                metrics["f1_score"],
                metrics["roc_auc"],
                metrics["far"],
                metrics["frr"],
                metrics["f1_threshold"],
            ])
        print(f"✅ Results saved for {dataset_name}")
