import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.utils import register_keras_serializable
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
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

    # Step 2c: Combine outputs (e.g., concatenation)
    combined = layers.Concatenate()([encoded_a, encoded_b])

    # Dense layer to learn similarity from combined embeddings
    x = layers.Dense(32, activation='relu')(combined)

    # Step 2d: Final classification layer (genuine vs forged)
    output = layers.Dense(2, activation='softmax')(x)

    # Full Siamese model
    model = Model(inputs=[input_a, input_b], outputs=output)
    return model
    
def evaluate_classification_metrics(y_true, y_pred_probs, dataset_name=None, output_dir="outputs/base"):
    if len(np.unique(y_true)) == 2:
        scores = y_pred_probs[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, scores)

        j_scores = tpr - fpr
        j_best_idx = np.argmax(j_scores)
        youden_threshold = thresholds[j_best_idx]

        y_pred = (scores >= youden_threshold).astype(int)
        auc = roc_auc_score(y_true, scores)
    else:
        auc = youden_threshold = None
        y_pred = np.zeros_like(y_true)
        print("⚠ ROC AUC, Youden threshold not computed (only one class in y_true)")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-6)
        frr = fn / (fn + tp + 1e-6)

        far_percent = far * 100
        frr_percent = frr * 100

        print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"FAR (manual %): {far_percent:.2f}%")
        print(f"FRR (manual %): {frr_percent:.2f}%")
    else:
        tn = fp = fn = tp = far = frr = far_percent = frr_percent = 0.0
        print("⚠ Confusion matrix incomplete")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "youden_threshold": youden_threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "far_percent": round(far_percent, 2),
        "frr_percent": round(frr_percent, 2)
    }

def plot_image_comparison(original, normalized, filename, dataset_name):
    output_dir = os.path.join("preprocess_outputs", dataset_name)
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

def generate_sop1_outputs(generator, save_path="outputs/preprocessing"):
    max_visualizations = 5

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
                original = generator.preprocess_image_raw(img_path)  # Original preprocessing
                normalized = generator.preprocess_image(img_path)  # MinMax normalization preprocessing

                # Visualize only selected samples
                if fname in visualize_set:
                    filename = f"writer{writer}_{label_type}_{os.path.splitext(fname)[0]}"
                    plot_image_comparison(original.squeeze(), normalized.squeeze(), filename + "_minmax", generator.dataset_name)

    print(f"\n✅ Visualizations generated and saved to {save_path}")

def compute_distance_distributions(model, generator, dataset_name, base_output_dir="outputs/base", max_samples=5000):
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
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_run{run_id}_distance_distribution.png"))
    plt.close()

    # Step 5: UMAP visualization
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=y_test, cmap='tab20', s=10)
    plt.colorbar(scatter, label="Writer ID")
    plt.title(f'UMAP of Embeddings - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_run{run_id}_umap.png"))
    plt.close()

    # Step 6: Save stats
    stats_path = os.path.join(output_dir, f"{dataset_name}_run{run_id}_distance_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Intra-class: mean={np.mean(intra_dists):.4f}, std={np.std(intra_dists):.4f}\n")
        f.write(f"Inter-class: mean={np.mean(inter_dists):.4f}, std={np.std(inter_dists):.4f}\n")

    print(f"📊 Distance Distribution Outputs saved to {output_dir}")


# Parameters
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)  

datasets = {
    "CEDAR": {
        "path": "Dataset/CEDAR",
        "train_writers": list(range(260, 300)),
        "test_writers": list(range(300, 315))
    }
}

os.makedirs("outputs/base", exist_ok=True)
results_csv_path = "outputs/base/results.csv"
if not os.path.exists(results_csv_path):
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Accuracy", "F1 Score", "ROC AUC",
            "FAR", "FAR (%)", "FRR", "FRR (%)",
            "Youden Threshold", "TP", "FP", "TN", "FN",
            "CNR Mean", "CNR StdDev"
        ])

results = []

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing Siamese Model for Dataset: {dataset_name}")

    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )

    pairs, labels, meta = generator.generate_pairs(
        use_raw=False,
        return_metadata=True,
        log_csv_path=f"outputs/logs/{dataset_name}_run{run_id}_pairs.csv"
    )

    # --- Report and export CNR Statistics ---
    cnr_stats = generator.report_cnr_statistics()
    if cnr_stats:
        print(f"📈 CNR Statistics for {dataset_name}: {cnr_stats}")
    else:
        cnr_stats = {'average': None, 'std_dev': None}

    cnr_output_path = f"outputs/base/{dataset_name}_run{run_id}_cnr_values.csv"
    generator.export_cnr_to_csv(cnr_output_path)

    labels = np.array(labels).astype(np.int32)
    img1 = np.array([pair[0] for pair in pairs])
    img2 = np.array([pair[1] for pair in pairs])

    model = build_siamese_network(IMG_SHAPE)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )

    start_time = time.time()
    history = model.fit(
        [img1, img2], labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2
    )

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f"{model_dir}/{dataset_name}_run{run_id}_siamese_model.keras"
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}")
    print(f"⏱ Training completed in {time.time() - start_time:.2f} seconds")

    print(f"\n📊 Pre-processing metrics (MinMax) for {dataset_name}")
    generate_sop1_outputs(generator)

    print(f"\n🔍 Running real world metrics for {dataset_name}")
    test_pairs, test_labels = generator.generate_pairs(split='test', use_raw=True)
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    print("Label Distribution:", dict(zip(*np.unique(test_labels, return_counts=True))))
    if len(np.unique(test_labels)) < 2:
        print("⚠ Skipping evaluation — only one class present.")
        metrics = {
            'accuracy': None, 'f1_score': None, 'roc_auc': None, 
            'far': None, 'far_percent': None, 'frr': None, 'frr_percent': None,
            'youden_threshold': None, 'tp': None, 'fp': None, 'tn': None, 'fn': None
        }
    else:
        y_pred_probs = model.predict([test_img1, test_img2], batch_size=128)
        metrics = evaluate_classification_metrics(test_labels, y_pred_probs, dataset_name=dataset_name)
        compute_distance_distributions(model, generator, dataset_name)
        print(f"✅ Evaluation Complete for {dataset_name}")

    results.append((dataset_name, metrics))

    # --- Write complete metrics and CNR summary to results.csv ---
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset_name,
            metrics.get("accuracy"),
            metrics.get("f1_score"),
            metrics.get("roc_auc"),
            metrics.get("far"),
            metrics.get("far_percent"),
            metrics.get("frr"),
            metrics.get("frr_percent"),
            metrics.get("youden_threshold"),
            metrics.get("tp"),
            metrics.get("fp"),
            metrics.get("tn"),
            metrics.get("fn"),
            cnr_stats.get('average'),
            cnr_stats.get('std_dev')
        ])
    print(f"✅ Results summary saved for {dataset_name}")
    print(f"📄 CNR values per image saved to: {cnr_output_path}")

