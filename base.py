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
    
def evaluate_classification_metrics(y_true, y_pred_probs, dataset_name=None, output_dir="outputs/sop2"):
    """
    Evaluate binary classification metrics from softmax probability outputs using Youden's J threshold.
    Also saves FAR/FRR vs threshold plot and logs metrics.
    """
    # --- Youden's J threshold selection from ROC curve ---
    if len(np.unique(y_true)) == 2:
        scores = y_pred_probs[:, 1]  # Probability of class '1' (forged)
        fpr, tpr, thresholds = roc_curve(y_true, scores)

        j_scores = tpr - fpr
        j_best_idx = np.argmax(j_scores)
        youden_threshold = thresholds[j_best_idx]

        # Plot Youden's J statistic across thresholds
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, j_scores, label="Youden’s J (TPR - FPR)", color="purple")
        plt.axvline(x=youden_threshold, color="green", linestyle="--", label=f"Best J = {j_scores[j_best_idx]:.4f} at {youden_threshold:.4f}")
        plt.title("Youden’s J Statistic vs Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Youden’s J (TPR - FPR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{dataset_name}_youden_j_curve.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"📈 Youden’s J curve saved to {plot_path}")

                # Compute FAR and FRR at Youden threshold
        y_pred_youden = (scores >= youden_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_youden, labels=[0, 1])

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far_youden = fp / (fp + tn + 1e-6)
            frr_youden = fn / (fn + tp + 1e-6)
        else:
            far_youden = frr_youden = 0.0
            print("⚠ Confusion matrix incomplete at Youden threshold.")

        # Bar plot of FAR and FRR
        plt.figure(figsize=(5, 5))
        bars = plt.bar(['FAR', 'FRR'], [far_youden, frr_youden], color=['red', 'blue'])
        plt.ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom')

        plt.title(f"FAR and FRR at Youden’s J Threshold ({youden_threshold:.4f})")
        plt.ylabel("Error Rate")
        plt.tight_layout()

        # Save bar chart
        bar_path = os.path.join(plot_dir, f"{dataset_name}_youden_farfrr_bar.png")
        plt.savefig(bar_path)
        plt.close()
        print(f"📊 FAR/FRR bar chart saved to {bar_path}")

        # Use Youden's threshold for binary classification
        y_pred = (scores >= youden_threshold).astype(int)

        # AUC, EER, FNR
        auc = roc_auc_score(y_true, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
    else:
        auc = eer = eer_threshold = youden_threshold = None
        y_pred = np.zeros_like(y_true)
        print("⚠ ROC AUC, EER, Youden threshold not computed (only one class in y_true)")

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

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
    print(f"✅ Youden J Threshold: {youden_threshold:.4f}" if youden_threshold is not None else "❌ Youden J: Not available")

    # FAR/FRR Curve Visualization
    if len(np.unique(y_true)) == 2:
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, fpr, label='FAR (False Acceptance Rate)', color='red')
        plt.plot(thresholds, 1 - tpr, label='FRR (False Rejection Rate)', color='blue')
        plt.axvline(x=youden_threshold, color='green', linestyle='--', label=f'Youden J = {youden_threshold:.4f}')
        plt.title('FAR and FRR vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{dataset_name}_far_frr_youden.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📉 FAR/FRR curve saved to {plot_path}")

    # Save to file if dataset name is provided
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
            if eer is not None:
                f.write(f"EER            : {eer:.4f} at threshold {eer_threshold:.4f}\n")
            else:
                f.write(f"EER            : Not available\n")
            if youden_threshold is not None:
                f.write(f"Youden J        : {youden_threshold:.4f}\n")
        print(f"📝 Metrics saved to {filepath}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "youden_threshold": youden_threshold
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

    # Directly access correct keys now
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    plt.figure()
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(f"{dataset_name} - Training vs Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

    # Loss plot
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
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    print(f"📈 Accuracy and Training Curves saved to {output_dir}")


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

    # Load and generate pairs
    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )
    pairs, labels = generator.generate_pairs()
    labels = np.array(labels).astype(np.int32)

    # Shuffle before splitting to avoid label imbalance
    combined = list(zip(pairs, labels))
    np.random.shuffle(combined)
    pairs, labels = zip(*combined)
    pairs = list(pairs)
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
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
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

    # ========== Save the model ==========
    model_save_path = f"models/{dataset_name}_siamese_model.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"model saved to: {model_save_path}")

    train_time = time.time() - start_time
    print(f"⏱ Training completed in {train_time:.2f} seconds")
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
        y_pred_probs = model.predict([test_img1, test_img2], batch_size=128)
        metrics = evaluate_classification_metrics(test_labels, y_pred_probs, dataset_name=dataset_name)
        compute_distance_distributions(model, generator, dataset_name)
        results.append((dataset_name, metrics))
        print(f"✅ Evaluation Complete for {dataset_name}")