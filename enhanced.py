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

def evaluate_classification_metrics(y_true, distances, dataset_name=None, output_dir="outputs/enhanced"):
    """
    Evaluate binary classification metrics using F1-optimal threshold.
    """
    # Normalize distances to similarity scores
    scores = 1 - distances / np.max(distances)

    # --- F1-Optimal Threshold selection ---
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.linspace(0, 1, 200)

    for thresh in thresholds:
        y_pred_temp = (scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_temp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    f1_threshold = best_threshold
    y_pred = (scores >= f1_threshold).astype(int)
    print(f"📌 Optimal F1 Threshold: {f1_threshold:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)
    tpr = tp / (tp + fn + 1e-6)
    tnr = tn / (tn + fp + 1e-6)

    # Save metrics
    if dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{dataset_name}_metrics.txt")
        with open(filepath, "w") as f:
            f.write(f"Evaluation Metrics for {dataset_name}\n")
            f.write("="*40 + "\n")
            f.write(f"Accuracy       : {acc:.4f}\n")
            f.write(f"FAR (FP Rate)  : {far:.4f}\n")
            f.write(f"FRR (FN Rate)  : {frr:.4f}\n")
            f.write(f"TPR (Sensitivity): {tpr:.4f}\n")
            f.write(f"TNR (Specificity): {tnr:.4f}\n")
        print(f"📝 Metrics saved to {filepath}")

    return {
        "accuracy": acc,
        "far": far,
        "frr": frr,
        "tpr": tpr,
        "tnr": tnr
    }

def compute_distance_distributions(model, generator, dataset_name, base_output_dir="outputs/enhanced", max_samples=5000):
    """
    Compute and visualize distance distributions and UMAP embeddings for enhanced.py.
    """
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
    plt.savefig(os.path.join(output_dir, f"{filename}_comparison_enhanced.png"))
    plt.close()

def generate_sop1_outputs(generator, save_path="outputs/enhanced"):
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
                original = generator.preprocess_image(img_path)  # Original preprocessing
                clahe_img = generator.preprocess_image_clahe(img_path)  # CLAHE preprocessing

                # Only visualize selected images
                if fname in visualize_set:
                    filename = f"writer{writer}_{label_type}_{os.path.splitext(fname)[0]}"
                    plot_image_comparison(original.squeeze(), clahe_img.squeeze(), filename + "_clahe", generator.dataset_name)

    print(f"\n✅ visualizations generated and saved to {save_path}")


# Parameters
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)  
weights_dir = 'enhanced_weights'
metrics_dir = 'outputs/enhanced'
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
    print(f"\n📦 Processing Enhanced Model for Dataset: {dataset_name}")

    # Load data generator
    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )

    # Load triplet data
    train_dataset = generator.get_triplet_train(use_clahe=True)

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
    enhanced_weights_path = f"{weights_dir}/{dataset_name}.weights.h5"
    model.save_weights(enhanced_weights_path)
    print(f"✅ Enhanced model weights saved to: {enhanced_weights_path}")

    base_net_path = f"{weights_dir}/{dataset_name}_base.weights.h5"
    base_net = model.get_layer('base_network')
    base_net.save_weights(base_net_path)
    print(f"✅ Base network weights saved to: {base_net_path}")

    print(f"\n🔍 Running Evaluation for {dataset_name}")

    test_pairs, test_labels = generator.generate_pairs(split='test', use_clahe=True)
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])
    test_labels = np.array(test_labels)

    print("Label Distribution:", dict(zip(*np.unique(test_labels, return_counts=True))))
    if len(np.unique(test_labels)) < 2:
        print("⚠ Skipping evaluation — only one class present.")
    else:
        # Load the saved embedding model for inference
        embedding_model = create_base_network(IMG_SHAPE)
        embedding_model.load_weights(f"{weights_dir}/{dataset_name}_base.weights.h5")
        print(f"✅ Loaded embedding weights from {enhanced_weights_path}")

        # Generate embeddings
        emb1 = embedding_model.predict(test_img1, batch_size=128)
        emb2 = embedding_model.predict(test_img2, batch_size=128)

        # Save reference embeddings
        reference_embeddings_path = f"{weights_dir}/{dataset_name}_reference_embeddings.npy"
        np.save(reference_embeddings_path, emb1)
        print(f"✅ Reference embeddings saved to: {reference_embeddings_path}")

        distances = np.linalg.norm(emb1 - emb2, axis=1)
        metrics = evaluate_classification_metrics(test_labels, distances, dataset_name=dataset_name)
        results.append((dataset_name, metrics))
        compute_distance_distributions(model, generator, dataset_name)
        generate_sop1_outputs(generator, save_path=metrics_dir)
        print(f"✅ Evaluation Complete for {dataset_name}")
