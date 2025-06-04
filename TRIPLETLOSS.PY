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
    base_network = create_base_network(input_shape)

    # Triplet inputs
    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    encoded_anchor = base_network(anchor_input)
    encoded_positive = base_network(positive_input)
    encoded_negative = base_network(negative_input)

    # L2 Normalization
    encoded_anchor = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded_anchor)
    encoded_positive = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded_positive)
    encoded_negative = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded_negative)

    # Stack into a single tensor with shape (batch, 3, embedding_dim)
    merged_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(
        [encoded_anchor, encoded_positive, encoded_negative]
    )


    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)

    return model

def triplet_loss(margin=0.7):
    def loss(y_true, y_pred):  # y_pred has shape (batch_size, 3, embedding_dim)
        anchor = y_pred[:, 0]
        positive = y_pred[:, 1]
        negative = y_pred[:, 2]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

def evaluate_classification_metrics(y_true, distances, dataset_name=None, output_dir="outputs/sop2", threshold=None):
    """
    Evaluate binary classification metrics for a distance-based model (triplet loss).
    `distances` should be a 1D array of Euclidean distances between embeddings.
    `threshold` is the cutoff: distances < threshold → same class (genuine), else forged.
    If threshold is None, EER threshold will be used automatically.
    """

    y_true = np.array(y_true)
    distances = np.array(distances)

    # Normalize to similarity score if needed (optional)
    # similarities = 1 - distances / np.max(distances)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, -distances)  # invert because lower distance = genuine
    auc = roc_auc_score(y_true, -distances)

    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    # Use EER threshold if not explicitly provided
    if threshold is None:
        threshold = eer_threshold

    # Classify using distance threshold
    y_pred = (distances < threshold).astype(int)  # 1 = genuine, 0 = forged

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-6)
        frr = fn / (fn + tp + 1e-6)
    else:
        far = frr = 0.0
        print("⚠ Confusion matrix incomplete (only one class present)")

    # Print to console
    print("🔎 Evaluation Metrics (Triplet Distance-Based):")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {auc:.4f}")
    print(f"✅ FAR:       {far:.4f}")
    print(f"✅ FRR:       {frr:.4f}")
    print(f"✅ EER:       {eer:.4f} at threshold {eer_threshold:.4f}")

    # Save to file
    if dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{dataset_name}_metrics.txt")
        with open(filepath, "w") as f:
            f.write(f"Evaluation Metrics for {dataset_name} (Triplet)\n")
            f.write("="*40 + "\n")
            f.write(f"Accuracy       : {acc:.4f}\n")
            f.write(f"F1 Score       : {f1:.4f}\n")
            f.write(f"ROC AUC        : {auc:.4f}\n")
            f.write(f"FAR (FP Rate)  : {far:.4f}\n")
            f.write(f"FRR (FN Rate)  : {frr:.4f}\n")
            f.write(f"EER            : {eer:.4f} at threshold {eer_threshold:.4f}\n")
        print(f"📝 Metrics saved to {filepath}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "threshold_used": threshold,
    }

def evaluate_triplet_sop2(model_path, test_img1, test_img2, test_labels, dataset_name=None, output_dir="outputs/sop2"):
    """
    Evaluates verification metrics for a triplet-based model using distance-based comparison.
    """

    # Load the trained triplet model
    triplet_model = load_model(model_path, compile=False)

    # Extract the base embedding model
    embedding_model = triplet_model.get_layer("base_network")

    # Generate embeddings
    emb1 = embedding_model.predict(test_img1, batch_size=128)
    emb2 = embedding_model.predict(test_img2, batch_size=128)

    # Compute L2 distances
    distances = np.linalg.norm(emb1 - emb2, axis=1)

    # Normalize distances to similarity scores (optional)
    y_pred_probs = 1 - distances / np.max(distances)

    # Threshold for classification
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Evaluation metrics
    acc = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)
    auc = roc_auc_score(test_labels, y_pred_probs)
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    cm = confusion_matrix(test_labels, y_pred)

    # FAR / FRR
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-6)
        frr = fn / (fn + tp + 1e-6)
    else:
        far = frr = 0.0

    # Print results
    print("🔎 Triplet-Based SOP2 Evaluation:")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {auc:.4f}")
    print(f"✅ FAR:       {far:.4f}")
    print(f"✅ FRR:       {frr:.4f}")
    print(f"✅ EER:       {eer:.4f} at threshold {eer_threshold:.4f}")

    # Save to file
    if dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{dataset_name}_metrics.txt")
        with open(filepath, "w") as f:
            f.write(f"Triplet-Based Evaluation Metrics for {dataset_name}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Accuracy       : {acc:.4f}\n")
            f.write(f"F1 Score       : {f1:.4f}\n")
            f.write(f"ROC AUC        : {auc:.4f}\n")
            f.write(f"FAR (FP Rate)  : {far:.4f}\n")
            f.write(f"FRR (FN Rate)  : {frr:.4f}\n")
            f.write(f"EER            : {eer:.4f} at threshold {eer_threshold:.4f}\n")
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
    axes[1].set_title("MinMax Normalized")
    axes[1].axis("off")

    plt.tight_layout()
    
    # Save image to the correct dataset-specific path
    plt.savefig(os.path.join(output_dir, f"{filename}_comparison.png"))
    plt.close()

def compute_edge_count(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
    return np.sum(edges > 0)

def generate_sop1_outputs(generator, 
                          save_path="outputs/edge_count_summary.csv", 
                          avg_path="outputs/edge_count_averages.csv"):
    max_visualizations = 5
    rows = [["Writer", "Image", "Original_EdgeCount", "MinMax_EdgeCount", "PSNR"]]
    edge_stats = defaultdict(lambda: {"original": [], "normalized": [], "psnr": []})

    for dataset_path, writer in generator.test_writers:
        for label_type in ["genuine", "forged"]:
            img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
            if not os.path.exists(img_dir):
                continue

            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png")) and not f.startswith(".")]
            if len(img_files) < 1:
                continue

            # Choose a few to visualize
            visualize_set = set(random.sample(img_files, min(max_visualizations, len(img_files))))

            for fname in img_files:
                img_path = os.path.join(img_dir, fname)
                original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if original is None:
                    continue

                resized = cv2.resize(original, (220, 155))
                normalized = minmax_normalize(resized)

                # Visualize only selected samples
                if fname in visualize_set:
                    filename = f"writer{writer}_{label_type}_{os.path.splitext(fname)[0]}"
                    plot_image_comparison(resized, normalized, filename + "_minmax", dataset_name)

                # Edge Count Calculation
                edge_orig = compute_edge_count(resized)
                edge_norm = compute_edge_count(normalized)

                # PSNR Calculation
                psnr_value = compare_psnr(resized.astype(np.float32), (normalized * 255).astype(np.float32), data_range=255)

                rows.append([f"writer_{writer}", fname, edge_orig, edge_norm, round(psnr_value, 2)])
                edge_stats[f"writer_{writer}"]["original"].append(edge_orig)
                edge_stats[f"writer_{writer}"]["normalized"].append(edge_norm)
                edge_stats[f"writer_{writer}"]["psnr"].append(psnr_value)

    # Save full image-level summary
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(rows)
    print(f"\n✅ MinMax edge count + PSNR summary saved to {save_path}")

    # Save per-writer averages
    avg_rows = [["Writer", "Avg_Original_EdgeCount", "Avg_MinMax_EdgeCount", "Avg_PSNR"]]
    for writer_id, stats in edge_stats.items():
        avg_orig = np.mean(stats["original"])
        avg_norm = np.mean(stats["normalized"])
        avg_psnr = np.mean(stats["psnr"])
        avg_rows.append([writer_id, round(avg_orig, 2), round(avg_norm, 2), round(avg_psnr, 2)])

    with open(avg_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(avg_rows)
    print(f"📊 Per-writer average MinMax edge counts + PSNR saved to {avg_path}")

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

    print("📋 History keys:", history.history.keys())

    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure()
    plt.plot(loss, label='Train Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.title(f"{dataset_name} - Training vs Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    print(f"📈 Training Loss Curve saved to {output_dir}")



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

from collections import Counter

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing Triplet-Based Model for Dataset: {dataset_name}")

    # Load data generator
    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )

    # Load triplet data
    train_dataset = generator.get_triplet_train()

    # Build model (triplet loss version)
    model = build_siamese_network(IMG_SHAPE)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=triplet_loss(margin=0.7))

    # ========== Training ==========
    print("🧠 Starting training with triplet loss ...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=len(generator.train_writers),  # or len(generator.train_writers) * samples_per_writer // BATCH_SIZE
        epochs=EPOCHS,
        verbose=2
    )

   # ========== Save the model ==========
    # Save only the weights (recommended for models with Lambda layers)
    weights_path = f"models/{dataset_name}_triplet_weights.h5"
    os.makedirs("models", exist_ok=True)
    model.save_weights(weights_path)
    print(f"✅ Weights saved to: {weights_path}")


    # # ========== SOP 1 ==========
    # print(f"\n📊 Running SOP 1 Evaluation for {dataset_name}")
    # generate_sop1_outputs(
    #     generator,
    #     save_path=f"outputs/sop1/{dataset_name}_edge_count_summary.csv",
    #     avg_path=f"outputs/sop1/{dataset_name}_edge_count_averages.csv"
    # )

    # # ========== SOP 3 ==========
    # print(f"\n📈 Generating SOP 3 Curves for {dataset_name}")
    # generate_sop3_curves(history, dataset_name)

    # ========== SOP 2 Evaluation (Writer-Independent) ==========
    print(f"\n🔍 Running SOP 2 Evaluation for {dataset_name}")

    test_pairs, test_labels = generator.generate_pairs(split='test', use_raw=True)
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    print("Label Distribution:", dict(zip(*np.unique(test_labels, return_counts=True))))
    if len(np.unique(test_labels)) < 2:
        print("⚠ Skipping evaluation — only one class present.")
    else:
        # Rebuild the embedding model (same architecture as base network)
        embedding_model = create_base_network(IMG_SHAPE)

        # Load the trained weights from Triplet model
        weights_path = f"models/{dataset_name}_triplet_weights.h5"
        embedding_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print(f"✅ Loaded embedding weights from {weights_path}")

        # Generate embeddings
        emb1 = embedding_model.predict(test_img1, batch_size=128)
        emb2 = embedding_model.predict(test_img2, batch_size=128)

        # Compute distances
        distances = np.linalg.norm(emb1 - emb2, axis=1)

        # Optionally normalize and convert to similarity scores (for metrics that expect probability-like values)
        y_pred_probs = 1 - distances / np.max(distances)

        # Evaluate metrics using distance-based predictions
        metrics = evaluate_classification_metrics(test_labels, y_pred_probs, dataset_name=dataset_name)
        compute_distance_distributions(embedding_model, generator, dataset_name)
        results.append((dataset_name, metrics))
        print(f"✅ Evaluation Complete for {dataset_name}")
