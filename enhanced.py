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

# Parameters
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)  
weights_dir = 'enhanced_weights'
metrics_dir = 'enhanced_metrics'
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
        print(f"✅ Evaluation Complete for {dataset_name}")
