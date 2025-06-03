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
from collections import defaultdict


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
    
def evaluate_classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Evaluate binary classification metrics from softmax probability outputs.
    Handles edge cases where only one class may be present.
    """
    # Predicted labels: argmax over softmax output (e.g., [p0, p1])
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Accuracy and F1 score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC: only valid if both classes exist in y_true
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred_probs[:, 1])
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
    else:
        auc = None
        eer = None
        eer_threshold = None
        print("⚠ ROC AUC and EER not computed (only one class in y_true)")

    # Confusion matrix with forced labels
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-6)
        frr = fn / (fn + tp + 1e-6)
    else:
        tn = fp = fn = tp = far = frr = 0.0
        print("⚠ Confusion matrix incomplete (only one class present)")

    # Print metrics
    print("🔎 Evaluation Metrics:")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {auc:.4f}" if auc is not None else "❌ ROC AUC:   Not available")
    print(f"✅ FAR:       {far:.4f}")
    print(f"✅ FRR:       {frr:.4f}")
    print(f"✅ EER:       {eer:.4f} at threshold {eer_threshold:.4f}" if eer is not None else "❌ EER:       Not available")

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

def plot_image_comparison(original, normalized, filename):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(normalized, cmap='gray')
    axes[1].set_title("MinMax Normalized")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join("sop1_outputs", f"{filename}_comparison.png"))
    plt.close()

def plot_histograms(original, normalized, filename):
    plt.figure(figsize=(6, 4))
    plt.hist(original.ravel(), bins=50, alpha=0.5, label='Original')
    plt.hist(normalized.ravel(), bins=50, alpha=0.5, label='Normalized')
    plt.legend()
    plt.title("Histogram Comparison")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join("sop1_outputs", f"{filename}_histogram.png"))
    plt.close()

def compute_edge_count(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
    return np.sum(edges > 0)

def generate_sop1_outputs(generator, 
                          save_path="outputs/edge_count_summary.csv", 
                          avg_path="outputs/edge_count_averages.csv"):
    rows = [["Writer", "Image", "Original_EdgeCount", "Normalized_EdgeCount"]]
    edge_stats = defaultdict(lambda: {"original": [], "normalized": []})

    for dataset_path, writer in generator.test_writers:
        for label_type in ["genuine", "forged"]:
            img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
            if not os.path.exists(img_dir):
                continue

            img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
            if len(img_files) < 2:
                continue

            sampled = random.sample(img_files, min(2, len(img_files)))

            for fname in sampled:
                img_path = os.path.join(img_dir, fname)
                original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if original is None:
                    continue

                resized = cv2.resize(original, (220, 155))
                normalized = minmax_normalize(resized)

                filename = f"writer{writer}_{label_type}_{os.path.splitext(fname)[0]}"
                plot_image_comparison(resized, normalized, filename)
                plot_histograms(resized, normalized, filename)

                # Edge Count Calculation
                edge_orig = compute_edge_count(resized)
                edge_norm = compute_edge_count(normalized)

                # Save raw per-image results
                rows.append([f"writer_{writer}", fname, edge_orig, edge_norm])
                edge_stats[f"writer_{writer}"]["original"].append(edge_orig)
                edge_stats[f"writer_{writer}"]["normalized"].append(edge_norm)

                print(f"[{filename}] Edges — Original: {edge_orig}, Normalized: {edge_norm}")

    # Save raw image-level edge counts
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(rows)
    print(f"\n✅ Edge count summary saved to {save_path}")

    # Compute and save average edge counts per writer
    avg_rows = [["Writer", "Avg_Original_EdgeCount", "Avg_Normalized_EdgeCount"]]
    for writer_id, stats in edge_stats.items():
        avg_orig = np.mean(stats["original"])
        avg_norm = np.mean(stats["normalized"])
        avg_rows.append([writer_id, round(avg_orig, 2), round(avg_norm, 2)])

    with open(avg_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(avg_rows)
    print(f"📊 Per-writer average edge counts saved to {avg_path}")

def compute_distance_distributions(model, generator, dataset_name, base_output_dir="outputs/sop2", max_samples=5000):
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get unbatched test data
    X_test, y_test = generator.get_unbatched_data()
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Step 2: Extract embeddings from base model
    base_model = model.get_layer("base_network")
    embeddings = base_model.predict(X_test, batch_size=128, verbose=0)

    # Step 3: Compute distances
    intra_dists, inter_dists, seen = [], [], 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if seen >= max_samples:
                break
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if y_test[i] == y_test[j]:
                intra_dists.append(dist)
            else:
                inter_dists.append(dist)
            seen += 1

    # Step 4: Histogram plot
    plt.figure(figsize=(8, 5))
    sns.histplot(intra_dists, label='Genuine-Genuine', color='blue', kde=True, stat="density", bins=50)
    sns.histplot(inter_dists, label='Genuine-Forged', color='red', kde=True, stat="density", bins=50)
    plt.title(f'Distance Distribution - {dataset_name}')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_distribution.png"))
    plt.close()

    # Step 5: UMAP visualization
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=y_test, cmap='tab20', s=10)
    plt.colorbar(scatter, label="Writer ID")
    plt.title(f'UMAP of Embeddings - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap.png"))
    plt.close()

    # Step 6: Save stats
    stats_path = os.path.join(output_dir, "distance_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Intra-class: mean={np.mean(intra_dists):.4f}, std={np.std(intra_dists):.4f}\n")
        f.write(f"Inter-class: mean={np.mean(inter_dists):.4f}, std={np.std(inter_dists):.4f}\n")

    print(f"📊 Distance Distribution Outputs saved to {output_dir}")


def generate_sop3_curves(history, dataset_name, base_output_dir="outputs/sop3"):
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy Curve
    acc = history.history.get('sparse_categorical_accuracy') or history.history.get('accuracy')
    val_acc = history.history.get('val_sparse_categorical_accuracy') or history.history.get('val_accuracy')

    plt.figure()
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(f"{dataset_name} - Training vs Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

    # Loss Curve
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f"{dataset_name} - Training vs Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    print(f"📈 Validation Accuracy curves saved to {output_dir}")


# Parameters
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)  # Grayscale
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
    print(f"\n📦 Processing Softmax-Based Siamese Model for Dataset: {dataset_name}")

    # Load and generate pairs
    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )
    pairs, labels = generator.generate_pairs()
    labels = np.array(labels).astype(np.int32)

    # Split data
    val_split = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:val_split], pairs[val_split:]
    train_labels, val_labels = labels[:val_split], labels[val_split:]

    # Separate image pairs
    train_img1 = np.array([pair[0] for pair in train_pairs])
    train_img2 = np.array([pair[1] for pair in train_pairs])
    val_img1 = np.array([pair[0] for pair in val_pairs])
    val_img2 = np.array([pair[1] for pair in val_pairs])

    # Build model using softmax-based classification
    model = build_siamese_network(IMG_SHAPE)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=SparseCategoricalCrossentropy(name="Loss"),
        metrics=[SparseCategoricalAccuracy(name="Accuracy")]
    )

    # ========== Training ==========
    start_time = time.time()
    history = model.fit(
        [train_img1, train_img2], train_labels,
        validation_data=([val_img1, val_img2], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2
    )
    train_time = time.time() - start_time
    # ========== SOP 1 ==========
    print(f"\n📊 Running SOP 1 Evaluation for {dataset_name}")
    generate_sop1_outputs(
        generator,
        save_path=f"outputs/sop1/{dataset_name}_edge_count_summary.csv",
        avg_path=f"outputs/sop1/{dataset_name}_edge_count_averages.csv"
    )

    # ========== SOP 3 ==========
    print(f"\n📈 Generating SOP 3 Curves for {dataset_name}")
    generate_sop3_curves(history, dataset_name)

    # ========== SOP 2 Evaluation (Writer-Independent) ==========
    print(f"\n🔍 Running SOP 2 Evaluation for {dataset_name}")

    test_pairs, test_labels = generator.generate_pairs(split='test')
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    print("SOP 2 Label Distribution:", dict(zip(*np.unique(test_labels, return_counts=True))))
    if len(np.unique(test_labels)) < 2:
        print("⚠ Skipping SOP 2 evaluation — only one class present.")
    else:
        y_pred_probs = model.predict([test_img1, test_img2], batch_size=128)
        metrics = evaluate_classification_metrics(test_labels, y_pred_probs)
        compute_distance_distributions(model, generator, dataset_name)
        results.append((dataset_name, metrics))
        print(f"✅ SOP 2 Evaluation Complete for {dataset_name}")
        print(metrics)