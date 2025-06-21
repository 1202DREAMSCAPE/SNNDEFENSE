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
from pair_log import save_softmax_predictions
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
    
def evaluate_classification_metrics(y_true, y_pred_probs, dataset_name=None, output_dir="outputs/sop2", threshold=0.5):
    """
    Evaluate binary classification metrics from softmax probability outputs.
    Saves metrics to a .txt file if dataset_name is provided.
    """
    # Predicted labels
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC and EER
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred_probs[:, 1])
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
    else:
        auc = eer = eer_threshold = None
        print("⚠ ROC AUC and EER not computed (only one class in y_true)")

    # Confusion matrix and derived metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-6)
        frr = fn / (fn + tp + 1e-6)
    else:
        tn = fp = fn = tp = far = frr = 0.0
        print("⚠ Confusion matrix incomplete (only one class present)")

    # Print to console
    print("🔎 Evaluation Metrics:")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {auc:.4f}" if auc is not None else "❌ ROC AUC:   Not available")
    print(f"✅ FAR:       {far:.4f}")
    print(f"✅ FRR:       {frr:.4f}")
    print(f"✅ EER:       {eer:.4f} at threshold {eer_threshold:.4f}" if eer is not None else "❌ EER:       Not available")

    # Save to file if dataset name is provided
    if dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{dataset_name}_metrics_clahe.txt")
        with open(filepath, "w") as f:
            f.write(f"Evaluation Metrics for {dataset_name}\n")
            f.write("="*40 + "\n")
            f.write(f"Accuracy       : {acc:.4f}\n")
            f.write(f"F1 Score       : {f1:.4f}\n")
            f.write(f"ROC AUC        : {auc:.4f}\n" if auc is not None else "ROC AUC        : Not available\n")
            f.write(f"FAR (FP Rate)  : {far:.4f}\n")
            f.write(f"FRR (FN Rate)  : {frr:.4f}\n")
            if eer is not None:
                f.write(f"EER            : {eer:.4f} at threshold {eer_threshold:.4f}\n")
            else:
                f.write(f"EER            : Not available\n")
        print(f"📝 Metrics saved to {filepath}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_threshold
    }

def minmax_normalize(img):
    flat = img.flatten().reshape(-1, 1)
    scaled = MinMaxScaler().fit_transform(flat)
    return scaled.reshape(img.shape)

def plot_image_comparison(original, normalized, filename, dataset_name):
    output_dir = os.path.join("sop1_outputs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create side-by-side comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(normalized, cmap='gray')
    axes[1].set_title("CLAHE")
    axes[1].axis("off")

    plt.tight_layout()
    
    # Save image to the correct dataset-specific path
    plt.savefig(os.path.join(output_dir, f"{filename}_comparison_clahe.png"))
    plt.close()

def compute_edge_count(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
    return np.sum(edges > 0)

def generate_sop1_outputs(generator, 
                          save_path="outputs/edge_count_summary_clahe.csv", 
                          avg_path="outputs/edge_count_averages_clahe.csv",
                          max_visualizations=2):

    def apply_clahe(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    rows = [["Writer", "Image", "Original_EdgeCount", "CLAHE_EdgeCount", "PSNR"]]
    edge_stats = defaultdict(lambda: {"original": [], "clahe": [], "psnr": []})

    for dataset_path, writer in generator.test_writers:
        for label_type in ["genuine", "forged"]:
            img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
            if not os.path.exists(img_dir):
                continue

            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png")) and not f.startswith(".")]
            if len(img_files) < 1:
                continue

            # Visualize only a few
            visualize_set = set(random.sample(img_files, min(max_visualizations, len(img_files))))

            for fname in img_files:
                img_path = os.path.join(img_dir, fname)
                original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if original is None:
                    continue

                resized = cv2.resize(original, (220, 155))
                clahe_img = apply_clahe(resized)

                # Only visualize some images
                if fname in visualize_set:
                    filename = f"writer{writer}_{label_type}_{os.path.splitext(fname)[0]}"
                    plot_image_comparison(resized, clahe_img, filename + "_clahe", dataset_name)

                # Compute metrics
                edge_orig = compute_edge_count(resized)
                edge_clahe = compute_edge_count(clahe_img)
                psnr_value = compare_psnr(resized.astype(np.float32), clahe_img.astype(np.float32), data_range=255)

                rows.append([f"writer_{writer}", fname, edge_orig, edge_clahe, round(psnr_value, 2)])
                edge_stats[f"writer_{writer}"]["original"].append(edge_orig)
                edge_stats[f"writer_{writer}"]["clahe"].append(edge_clahe)
                edge_stats[f"writer_{writer}"]["psnr"].append(psnr_value)

    # Save full image-level CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(rows)
    print(f"\n✅ CLAHE edge count + PSNR summary saved to {save_path}")

    # Save average CSV
    avg_rows = [["Writer", "Avg_Original_EdgeCount", "Avg_CLAHE_EdgeCount", "Avg_PSNR"]]
    for writer_id, stats in edge_stats.items():
        avg_orig = np.mean(stats["original"])
        avg_clahe = np.mean(stats["clahe"])
        avg_psnr = np.mean(stats["psnr"])
        avg_rows.append([writer_id, round(avg_orig, 2), round(avg_clahe, 2), round(avg_psnr, 2)])

    with open(avg_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(avg_rows)
    print(f"📊 Per-writer average CLAHE edge counts + PSNR saved to {avg_path}")

# Modify evaluation to include noisy and CLAHE-preprocessed images
def evaluate_with_noise_and_clahe(model, generator, test_img1, test_img2, test_labels, dataset_name, output_dir="outputs/sop2"):
    """
    Evaluate model performance on noisy and CLAHE-preprocessed images.
    Saves metrics for both noisy and CLAHE-preprocessed images.
    """
    # Add noise to test images
    noisy_img1 = np.array([add_noise(img) for img in test_img1])
    noisy_img2 = np.array([add_noise(img) for img in test_img2])

    # Apply CLAHE using preprocess_image_clahe from SignatureDataGenerator
    clahe_img1 = np.array([generator.preprocess_image_clahe_from_array(img) for img in test_img1])
    clahe_img2 = np.array([generator.preprocess_image_clahe_from_array(img) for img in test_img2])

    # Evaluate on noisy images
    print("\n🔍 Evaluating on noisy images...")
    noisy_pred_probs = model.predict([noisy_img1, noisy_img2], batch_size=128)
    noisy_metrics = evaluate_classification_metrics(test_labels, noisy_pred_probs, dataset_name=f"{dataset_name}_noisy", output_dir=output_dir)

    # Evaluate on CLAHE-preprocessed images
    print("\n🔍 Evaluating on CLAHE-preprocessed images...")
    clahe_pred_probs = model.predict([clahe_img1, clahe_img2], batch_size=128)
    clahe_metrics = evaluate_classification_metrics(test_labels, clahe_pred_probs, dataset_name=f"{dataset_name}_clahe", output_dir=output_dir)

    return noisy_metrics, clahe_metrics

def add_noise(img):
    if noise_type == "gaussian":
        row, col = img.shape[:2]
        mean = 0
        sigma = 15  # adjust as needed
        gauss = np.random.normal(mean, sigma, (row, col)).reshape(row, col)
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


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
    model_save_path = f"models/{dataset_name}_clahe_model.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"model saved to: {model_save_path}")
    print(f"⏱ Training completed in {time.time() - start_time:.2f} seconds")

    # ========== SOP 1 ==========
    print(f"\n📊 Running SOP 1 Evaluation for {dataset_name}")
    generate_sop1_outputs(
        generator,
        save_path=f"outputs/sop1/{dataset_name}_edge_count_summary.csv",
        avg_path=f"outputs/sop1/{dataset_name}_edge_count_averages.csv"
    )

    # ========== Evaluate with Noise and CLAHE ==========
    print(f"\n🔍 Evaluating with Noise and CLAHE for {dataset_name}")
    test_pairs, test_labels = generator.generate_pairs(split='test', use_raw=True)
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    if len(np.unique(test_labels)) < 2:
        print("⚠ Skipping evaluation — only one class present.")
    else:
        noisy_metrics, clahe_metrics = evaluate_with_noise_and_clahe(
            model, generator, test_img1, test_img2, test_labels, dataset_name
        )
        print(f"✅ Evaluation Complete for {dataset_name}")
        print(f"Noisy Metrics: {noisy_metrics}")
        print(f"CLAHE Metrics: {clahe_metrics}")