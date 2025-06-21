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
    
def evaluate_classification_metrics(y_true, y_pred_probs, dataset_name=None, output_dir="outputs/visualizations_clahe", threshold=0.5):
    """
    Evaluate binary classification metrics from softmax probability outputs.
    Saves metrics to a .txt file if dataset_name is provided.
    """
    # Predicted labels
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred_probs[:, 1])
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
    else:
        auc = None
        print("⚠ ROC AUC not computed (only one class in y_true)")

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
    print("Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}" if auc is not None else "❌ ROC AUC:   Not available")
    print(f"FAR:       {far:.4f}")
    print(f"FRR:       {frr:.4f}")

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
        print(f"📝 Metrics saved to {filepath}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "far": far,
        "frr": frr
    }

def plot_image_comparison(original, normalized, filename, dataset_name):
    output_dir = os.path.join("outputs/visualizations_clahe", dataset_name)
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

def generate_sop1_outputs(generator, save_path="outputs/visualizations_clahe"):
    max_visualizations = 5

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
                original = generator.preprocess_image_raw(img_path)  # Original preprocessing
                clahe_img = generator.preprocess_image_clahe(img_path)  # CLAHE preprocessing

                # Only visualize selected images
                if fname in visualize_set:
                    filename = f"writer{writer}_{label_type}_{os.path.splitext(fname)[0]}"
                    plot_image_comparison(original.squeeze(), clahe_img.squeeze(), filename + "_clahe", generator.dataset_name)

    print(f"\n✅ CLAHE visualizations generated and saved to {save_path}")

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
os.makedirs("outputs/visualizations_clahe", exist_ok=True)
# Define the path for the results CSV file
results_csv_path = "outputs/visualizations_clahe/CLAHE_results.csv"

# Ensure the CSV file has a header if it doesn't exist
if not os.path.exists(results_csv_path):
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Accuracy", "F1 Score", "ROC AUC", "FAR", "FRR", "Youden Threshold"])

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

    # ========== Save the model ==========
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Dynamically determine the run_id based on existing files
    existing_files = [f for f in os.listdir(model_dir) if f.startswith(f"{dataset_name}_run") and f.endswith("_siamese_model.h5")]
    run_id = len(existing_files) + 1  # Increment based on existing files

    model_save_path = f"{model_dir}/{dataset_name}_run{run_id}_siamese_model.h5"
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}")

    print(f"⏱ Training completed in {time.time() - start_time:.2f} seconds")

    # ========== SOP 1 ==========
    print(f"\n📊 Pre-processing metrics (CLAHE) for {dataset_name}")
    generate_sop1_outputs(
        generator)
    
    # ====================
    print(f"\n🔍 Running real world metrics for {dataset_name}")

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
                metrics["youden_threshold"],
            ])
        print(f"✅ Results saved for {dataset_name}")