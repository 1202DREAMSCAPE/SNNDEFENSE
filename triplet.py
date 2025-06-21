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

def compute_distance_distributions(
    weights_path, generator, dataset_name,
    base_output_dir="outputs/sop2", max_samples=5000,
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
    plt.savefig(os.path.join(output_dir, "distance_distribution_triplet.png"))
    plt.close()

    # UMAP visualization
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=y_test, cmap='tab20', s=10)
    plt.colorbar(scatter, label="Writer ID")
    plt.title(f'UMAP of Embeddings - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_triplet.png"))
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
metrics_dir = 'triplet_metrics'
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

        # def calling
        compute_distance_distributions(
            weights_path=triplet_weights_path,
            generator=generator,
            dataset_name=dataset_name
        )
        print(f"✅ Evaluation Complete for {dataset_name}")