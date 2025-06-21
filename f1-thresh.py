import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.utils import register_keras_serializable
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, silhouette_score, precision_score, recall_score
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


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

    # Step 2c: Combine outputs (e.g., concatenation)
    combined = layers.Concatenate()([encoded_a, encoded_b])

    # Dense layer to learn similarity from combined embeddings
    x = layers.Dense(32, activation='relu')(combined)

    # Step 2d: Final classification layer (genuine vs forged)
    output = layers.Dense(2, activation='softmax')(x)

    # Full Siamese model
    model = Model(inputs=[input_a, input_b], outputs=output)
    return model

def evaluate_classification_metrics(y_true, y_pred_probs, dataset_name=None, output_dir="outputs/f1threshold"):
    """
    Evaluate binary classification metrics from softmax probability outputs using F1-optimal threshold.
    Saves FAR/FRR vs threshold plot, FAR/FRR bar chart, and logs metrics.
    """
    f1_threshold = auc = None
    y_pred = np.zeros_like(y_true)

    if len(np.unique(y_true)) == 2:
        scores = y_pred_probs[:, 1]  # Probability of class '1' (forged)

        # --- F1-Optimal Threshold selection ---
        best_threshold = 0.5
        best_f1 = 0.0
        thresholds = np.linspace(0, 1, 200)

        far_values = []
        frr_values = []
        f1_scores = []

        for thresh in thresholds:
            y_pred_temp = (scores >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred_temp, zero_division=0)
            f1_scores.append(f1)

            cm = confusion_matrix(y_true, y_pred_temp, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                far = fp / (fp + tn + 1e-6)
                frr = fn / (fn + tp + 1e-6)
            else:
                far = frr = 0.0
                print(f"⚠ Confusion matrix is not 2x2 for threshold {thresh:.4f}. Skipping FAR/FRR calculation.")

            far_values.append(far)
            frr_values.append(frr)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        f1_threshold = best_threshold
        y_pred = (scores >= f1_threshold).astype(int)
        print(f"📌 Optimal F1 Threshold: {f1_threshold:.4f} | Best F1: {best_f1:.4f}")

        # Plot F1 score over thresholds
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores, label="F1 Score", color="darkorange")
        plt.axvline(x=f1_threshold, color="green", linestyle="--", label=f'Best F1 = {best_f1:.4f} at {f1_threshold:.4f}')
        plt.title("F1 Score vs Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        f1_plot_path = os.path.join(plot_dir, f"{dataset_name}_f1_threshold_curve.png")
        plt.savefig(f1_plot_path)
        plt.close()

        # Compute FAR and FRR at F1-optimal threshold
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn + 1e-6)
            frr = fn / (fn + tp + 1e-6)
        else:
            far = frr = 0.0
            print("⚠ Confusion matrix incomplete at F1 threshold.")

        from sklearn.metrics import ConfusionMatrixDisplay

        # Save confusion matrix plot at F1 threshold
        if cm.shape == (2, 2):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Genuine', 'Forged'])
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f"Confusion Matrix @ F1 Threshold ({f1_threshold:.4f})")
            confmat_path = os.path.join(plot_dir, f"{dataset_name}_f1_confmat.png")
            plt.savefig(confmat_path)
            plt.close()
            print(f"🧩 Confusion matrix saved to {confmat_path}")

        # Bar plot of FAR and FRR
        plt.figure(figsize=(5, 5))
        bars = plt.bar(['FAR', 'FRR'], [far, frr], color=['red', 'blue'])
        plt.ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom')

        plt.title(f"FAR and FRR at F1 Threshold ({f1_threshold:.4f})")
        plt.ylabel("Error Rate")
        plt.tight_layout()

        bar_path = os.path.join(plot_dir, f"{dataset_name}_f1_farfrr_bar.png")
        plt.savefig(bar_path)
        plt.close()
        print(f"📊 FAR/FRR bar chart saved to {bar_path}")

        # FAR/FRR vs Threshold curve
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, far_values, label='FAR (False Acceptance Rate)', color='red')
        plt.plot(thresholds, frr_values, label='FRR (False Rejection Rate)', color='blue')
        plt.axvline(x=f1_threshold, color='green', linestyle="--", label=f'F1 Threshold = {f1_threshold:.4f}')
        plt.title('FAR and FRR vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        farfrr_curve_path = os.path.join(plot_dir, f"{dataset_name}_far_frr_curve.png")
        plt.savefig(farfrr_curve_path)
        plt.close()
        print(f"📉 FAR/FRR curve saved to {farfrr_curve_path}")

        plt.hist(scores, bins=50, color='darkblue', alpha=0.7)
        plt.axvline(f1_threshold, color='green', linestyle='--', label=f'F1 Threshold = {f1_threshold:.4f}')
        plt.title(f"{dataset_name} – Softmax Output Distribution (Class 1 Prob)")
        plt.xlabel("Predicted Probability for Forged")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{dataset_name}_softmax_distribution.png")
        plt.close()

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Save metrics
    if dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{dataset_name}_metrics.txt")
        with open(filepath, "w") as f:
            f.write(f"Evaluation Metrics for {dataset_name}\n")
            f.write("="*40 + "\n")
            f.write(f"Accuracy       : {acc:.4f}\n")
            f.write(f"F1 Score       : {f1:.4f}\n")
            f.write(f"ROC AUC        : {auc:.4f}\n" if auc is not None else "ROC AUC        : Not available\n")
            f.write(f"FAR (FP Rate)  : {far:.4f}\n")
            f.write(f"FRR (FN Rate)  : {frr:.4f}\n")
            if f1_threshold is not None:
                f.write(f"F1 Threshold   : {f1_threshold:.4f}\n")
        print(f"📝 Metrics saved to {filepath}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "f1_threshold": f1_threshold
    }

# Parameters
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)  
weights_dir = 'base_weights_softmax'
metrics_dir = 'baseline_metrics_softmax'
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

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

results = []

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing Siamese Model for Dataset: {dataset_name}")
    # Load and generate training pairs (all writers in train set)
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
    model_save_path = f"models/{dataset_name}_f1_model.h5"
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
