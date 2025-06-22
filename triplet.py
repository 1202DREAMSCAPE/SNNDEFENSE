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
from tensorflow.keras.optimizers import Adam
from SignatureDataGenerator import SignatureDataGenerator
from collections import Counter
import seaborn as sns
import umap
import csv
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
        layers.Conv2D(96, (11, 11), activation='relu', strides=(4, 4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.ZeroPadding2D((2, 2)),
        layers.Conv2D(256, (5, 5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Dropout(0.3),

        layers.ZeroPadding2D((1, 1)),
        layers.Conv2D(384, (3, 3), activation='relu'),
        layers.ZeroPadding2D((1, 1)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0005))  # Final embedding
    ], name="base_network")
    return model

def triplet_loss(margin=1):
    """
    Triplet loss function with margin.
    """
    def loss(y_true, y_pred):  # y_pred has shape (batch_size, 3, embedding_dim)
        anchor = y_pred[:, 0]
        positive = y_pred[:, 1]
        negative = y_pred[:, 2]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

def build_triplet_network(input_shape):
    """
    Build a triplet network using the base CNN for embeddings.
    """
    base_network = create_base_network(input_shape)

    # Triplet inputs
    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    # Pass inputs through the shared base network
    encoded_anchor = base_network(anchor_input)
    encoded_positive = base_network(positive_input)
    encoded_negative = base_network(negative_input)

    # L2 normalization for embeddings
    encoded_anchor = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded_anchor)
    encoded_positive = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded_positive)
    encoded_negative = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded_negative)

    # Stack embeddings into a single tensor (batch_size, 3, embedding_dim)
    merged_output = layers.Lambda(lambda x: tf.stack(x, axis=1))(
        [encoded_anchor, encoded_positive, encoded_negative]
    )

    # Create the triplet model
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)
    return model

def evaluate_classification_metrics(y_true, distances, dataset_name=None, output_dir="outputs/tripletloss"):
    """
    Evaluate binary classification metrics from Euclidean distances using Youden's J statistic.
    Lower distance = more similar (genuine), so thresholding is done with <=.
    """
    if len(np.unique(y_true)) == 2:
        fpr, tpr, thresholds = roc_curve(y_true, -distances)  # Invert for correct direction

        j_scores = tpr - fpr
        j_best_idx = np.argmax(j_scores)
        youden_threshold = thresholds[j_best_idx] * -1  # Undo inversion

        # Plot Youden's J
        plt.figure(figsize=(8, 5))
        plt.plot(-thresholds, j_scores, label="Youden’s J (TPR - FPR)", color="purple")
        plt.axvline(x=youden_threshold, color="green", linestyle="--", label=f"Best J = {j_scores[j_best_idx]:.4f} at {youden_threshold:.4f}")
        plt.title("Youden’s J Statistic vs Distance Threshold")
        plt.xlabel("Distance Threshold")
        plt.ylabel("Youden’s J (TPR - FPR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{dataset_name}_run{run_id}_youden_j_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Youden’s J curve saved to {plot_path}")

        # Classification
        y_pred = (distances <= youden_threshold).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn + 1e-6)
            frr = fn / (fn + tp + 1e-6)
        else:
            far = frr = 0.0
            print("⚠ Confusion matrix incomplete.")

        # Bar plot
        plt.figure(figsize=(5, 5))
        bars = plt.bar(['FAR', 'FRR'], [far, frr], color=['red', 'blue'])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom')
        plt.ylim(0, 1)
        plt.title(f"FAR and FRR at Youden’s Threshold ({youden_threshold:.4f})")
        plt.tight_layout()
        bar_path = os.path.join(plot_dir, f"{dataset_name}_run{run_id}_youden_farfrr_bar.png")
        plt.savefig(bar_path)
        plt.close()
        print(f"📊 FAR/FRR bar chart saved to {bar_path}")

        auc = roc_auc_score(y_true, -distances)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("Evaluation Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        print(f"FAR:       {far:.4f}")
        print(f"FRR:       {frr:.4f}")
        print(f"Youden J Threshold: {youden_threshold:.4f}")

        # FAR/FRR line plot
        plt.figure(figsize=(8, 5))
        plt.plot(-thresholds, fpr, label='FAR (False Acceptance Rate)', color='red')
        plt.plot(-thresholds, 1 - tpr, label='FRR (False Rejection Rate)', color='blue')
        plt.axvline(x=youden_threshold, color='green', linestyle='--', label=f'Youden J = {youden_threshold:.4f}')
        plt.xlabel("Distance Threshold")
        plt.ylabel("Error Rate")
        plt.title("FAR and FRR vs Distance Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        far_frr_plot_path = os.path.join(plot_dir, f"{dataset_name}_run{run_id}_far_frr_youden.png")
        plt.savefig(far_frr_plot_path)
        plt.close()
        print(f"📉 FAR/FRR curve saved to {far_frr_plot_path}")

        # Save to txt
        if dataset_name:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{dataset_name}_run{run_id}_metrics.txt")
            with open(filepath, "w") as f:
                f.write(f"Evaluation Metrics for {dataset_name}\n")
                f.write("="*40 + "\n")
                f.write(f"Accuracy       : {acc:.4f}\n")
                f.write(f"F1 Score       : {f1:.4f}\n")
                f.write(f"ROC AUC        : {auc:.4f}\n")
                f.write(f"FAR (FP Rate)  : {far:.4f}\n")
                f.write(f"FRR (FN Rate)  : {frr:.4f}\n")
                f.write(f"Youden J        : {youden_threshold:.4f}\n")
            print(f"📝 Metrics saved to {filepath}")

        return {
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "far": far,
            "frr": frr,
            "youden_threshold": youden_threshold
        }

    else:
        print("⚠ ROC AUC and thresholding skipped — only one class present.")
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "far": 0.0,
            "frr": 0.0,
            "youden_threshold": None
        }


def compute_distance_distributions(
    weights_path, generator, dataset_name,
    base_output_dir="outputs/tripletloss", max_samples=5000,
    img_shape=(155, 220, 1)
):
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    X_test, y_test = generator.get_unbatched_data()
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Load model from weights
    base_model = create_base_network(img_shape)
    base_model.load_weights(weights_path)
    embeddings = base_model.predict(X_test, batch_size=128, verbose=0)

    # Compute distances
    intra_dists, inter_dists, seen = [], [], 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if seen >= max_samples:
                break
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            (intra_dists if y_test[i] == y_test[j] else inter_dists).append(dist)
            seen += 1

    # Distance histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(intra_dists, label='Genuine-Genuine', color='blue', kde=True, stat="density", bins=50)
    sns.histplot(inter_dists, label='Genuine-Forged', color='red', kde=True, stat="density", bins=50)
    plt.title(f'Distance Distribution - {dataset_name}')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"run{run_id}_distance_distribution_triplet.png"))
    plt.close()

    # UMAP visualization
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=y_test, cmap='tab20', s=10)
    plt.colorbar(scatter, label="Writer ID")
    plt.title(f'UMAP of Embeddings - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"run{run_id}_umap_triplet.png"))
    plt.close()

    # Save stats
    with open(os.path.join(output_dir, "distance_stats.txt"), "w") as f:
        f.write(f"Intra-class: mean={np.mean(intra_dists):.4f}, std={np.std(intra_dists):.4f}\n")
        f.write(f"Inter-class: mean={np.mean(inter_dists):.4f}, std={np.std(inter_dists):.4f}\n")

    print(f"📊 Distance Distribution Outputs saved to {output_dir}")

# Parameters
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)  
weights_dir = 'triplet_weights'
os.makedirs(weights_dir, exist_ok=True)

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

os.makedirs("outputs/tripletloss", exist_ok=True)
# Define the path for the results CSV file
results_csv_path = "outputs/tripletloss/results.csv"

# Ensure the CSV file has a header if it doesn't exist
if not os.path.exists(results_csv_path):
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Accuracy", "F1 Score", "ROC AUC", "FAR", "FRR", "Youden Threshold"])

results = []

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
    train_dataset = generator.get_triplet_train(use_clahe=False)
    # Build model (triplet loss version)
    model = build_triplet_network(IMG_SHAPE)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=triplet_loss(margin=1))

    # ========== Training ==========
    print("🧠 Starting training with triplet loss ...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=len(generator.train_writers),
        epochs=EPOCHS,
        verbose=2
    )

    # Save the model weights
    triplet_weights_path = f"{weights_dir}/{dataset_name}.weights.h5"
    model.get_layer("base_network").save_weights(triplet_weights_path)
    print(f"✅ Triplet model weights saved to: {triplet_weights_path}")

    print(f"\n🔍 Running Evaluation for {dataset_name}")

    test_pairs, test_labels = generator.generate_pairs(split='test', use_raw=True)
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    if len(np.unique(test_labels)) < 2:
        print("⚠ Skipping evaluation — only one class present.")
    else:
        # Load the saved embedding model for inference
        embedding_model = create_base_network(IMG_SHAPE)
        embedding_model.load_weights(triplet_weights_path)
        print(f"✅ Loaded embedding weights from {triplet_weights_path}")

        # Generate embeddings
        emb1 = embedding_model.predict(test_img1, batch_size=128)
        emb2 = embedding_model.predict(test_img2, batch_size=128)

        # Compute Euclidean distances
        distances = np.linalg.norm(emb1 - emb2, axis=1)

        # Normalize distances to similarity scores
        y_pred_probs = 1 - distances / np.max(distances)

        print(f"y_pred_probs: {y_pred_probs}")
        print(f"y_true: {test_labels}")

        metrics = evaluate_classification_metrics(test_labels, distances, dataset_name=dataset_name)
        results.append((dataset_name, metrics))
        # def calling
        compute_distance_distributions(
            weights_path=triplet_weights_path,
            generator=generator,
            dataset_name=dataset_name
        )
        print(f"✅ Evaluation Complete for {dataset_name}")

        with open(results_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset_name,
                metrics["accuracy"],
                metrics["f1_score"],
                metrics["roc_auc"],
                metrics["far"],
                metrics["frr"],
                metrics["youden_threshold"]
            ])
        print(f"✅ Results saved for {dataset_name}")
