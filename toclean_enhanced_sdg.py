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
    ])
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

    # Output all three embeddings
    model = Model(inputs=[anchor_input, positive_input, negative_input],
                  outputs=[encoded_anchor, encoded_positive, encoded_negative])

    return model

def triplet_loss(margin=0.7):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss


    
def evaluate_classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Evaluate classification metrics for binary softmax output.
    """
    # Predicted labels: take argmax (0 = genuine, 1 = forged)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Accuracy & F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # ROC AUC (use probability of positive class, assumed class 1 = forged)
    auc = roc_auc_score(y_true, y_pred_probs[:, 1])

    # FAR / FRR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn + 1e-6)  # False accept: forged accepted as genuine
    frr = fn / (fn + tp + 1e-6)  # False reject: genuine rejected as forged

    # EER: where FAR = FRR
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    print("🔎 Evaluation Metrics:")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {auc:.4f}")
    print(f"✅ FAR:       {far:.4f}")
    print(f"✅ FRR:       {frr:.4f}")
    print(f"✅ EER:       {eer:.4f} at threshold {eer_threshold:.4f}")

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

# ========== Sample and Process Images from Test Writers ==========

def generate_sop1_outputs(generator):
    rows = [["Writer", "Image", "Original_EdgeCount", "Normalized_EdgeCount"]]

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

                # Save comparisons
                filename = f"writer{writer}_{label_type}_{os.path.splitext(fname)[0]}"
                plot_image_comparison(resized, normalized, filename)
                plot_histograms(resized, normalized, filename)

                # Edge Count
                edge_orig = compute_edge_count(resized)
                edge_norm = compute_edge_count(normalized)
                rows.append([f"writer_{writer}", fname, edge_orig, edge_norm])
                print(f"[{filename}] Edges — Original: {edge_orig}, Normalized: {edge_norm}")

    # Save edge counts summary
    with open(EDGE_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\n✅ Edge count summary saved to {EDGE_LOG_PATH}")

def compute_distance_distributions(model, generator, dataset_name, output_dir="sop2_outputs", max_samples=5000):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get unbatched test data
    X_test, y_test = generator.get_unbatched_data()
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Step 2: Extract embeddings from base model
    base_model = model.get_layer("base_network")
    embeddings = base_model.predict(X_test, batch_size=128, verbose=0)

    # Step 3: Compute intra-class and inter-class distances
    intra_dists = []
    inter_dists = []
    seen = 0

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

    # Step 4: Plot histogram of distances
    plt.figure(figsize=(8, 5))
    sns.histplot(intra_dists, label='Genuine-Genuine', color='blue', kde=True, stat="density", bins=50)
    sns.histplot(inter_dists, label='Genuine-Forged', color='red', kde=True, stat="density", bins=50)
    plt.title(f'Distance Distribution - {dataset_name}')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_distance_distribution.png"))
    plt.close()

    # Step 5: UMAP plot
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=y_test, cmap='tab20', s=10)
    plt.colorbar(scatter, label="Writer ID")
    plt.title(f'UMAP of Embeddings - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_umap.png"))
    plt.close()

    # Step 6: Save intra/inter class stats
    stats_path = os.path.join(output_dir, f"{dataset_name}_distance_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Intra-class: mean={np.mean(intra_dists):.4f}, std={np.std(intra_dists):.4f}\n")
        f.write(f"Inter-class: mean={np.mean(inter_dists):.4f}, std={np.std(inter_dists):.4f}\n")

def generate_sop3_curves(history, dataset_name):
    os.makedirs("sop3_outputs", exist_ok=True)

    # Training vs Validation Accuracy
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
    plt.savefig(f"sop3_outputs/{dataset_name}_accuracy_curve.png")
    plt.close()

    # Training Loss Curve
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
    plt.savefig(f"sop3_outputs/{dataset_name}_loss_curve.png")
    plt.close()

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
    labels = np.array(labels).astype(np.float32)

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

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        os.path.join(weights_dir, f"{dataset_name}_best_model.h5"),
        save_best_only=True, monitor="val_loss", verbose=1
    )
    earlystop_cb = EarlyStopping(patience=3, restore_best_weights=True)

    # ========== Training ==========
    start_time = time.time()
    history = model.fit(
        [train_img1, train_img2], train_labels,
        validation_data=([val_img1, val_img2], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=2
    )
    train_time = time.time() - start_time
    generate_sop1_outputs(generator)
    generate_sop3_curves(history, dataset_name)

    # ========== Save Final Model ==========
    model.save(os.path.join(weights_dir, f"{dataset_name}_final_model.h5"))

    # SOP 1 Visualizations
    generate_sop1_outputs(generator)

    # SOP 2 Evaluation
    X_test, y_test = generator.get_unbatched_data()
    test_pairs, test_labels = [], []
    for i in range(0, len(X_test) - 1, 2):
        test_pairs.append((X_test[i], X_test[i + 1]))
        test_labels.append(1 if y_test[i] == y_test[i + 1] else 0)

    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    y_pred_probs = model.predict([test_img1, test_img2], batch_size=128)
    metrics = evaluate_classification_metrics(test_labels, y_pred_probs, dataset_name, metrics_dir)

    # SOP 2 Distance Distribution
    compute_distance_distributions(model, test_img1, test_img2, test_labels, dataset_name, metrics_dir)

    # Placeholder for results collection if needed
    results.append((dataset_name, metrics))