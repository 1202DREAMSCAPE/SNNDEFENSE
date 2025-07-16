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
import csv
import umap
import sys

run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
np.random.seed(1337)
random.seed(1337)
tf.random.set_seed(1337)

@register_keras_serializable()
def create_base_network(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        layers.Conv2D(96, (11,11), activation='relu', strides=(4,4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3,3), strides=(2,2)),

        layers.ZeroPadding2D((2,2)),
        layers.Conv2D(256, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3,3), strides=(2,2)),
        layers.Dropout(0.3),

        layers.ZeroPadding2D((1,1)),
        layers.Conv2D(384, (3,3), activation='relu'),
        layers.ZeroPadding2D((1,1)),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((3,3), strides=(2,2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='linear')
    ], name="base_network")
    return model

def triplet_loss(margin=1):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:,0], y_pred[:,1], y_pred[:,2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

def build_triplet_network(input_shape):
    base_network = create_base_network(input_shape)

    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')

    encoded_anchor = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(base_network(anchor_input))
    encoded_positive = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(base_network(positive_input))
    encoded_negative = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(base_network(negative_input))

    merged_output = layers.Lambda(lambda x: tf.stack(x, axis=1))(
        [encoded_anchor, encoded_positive, encoded_negative]
    )

    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)

def evaluate_classification_metrics(y_true, distances, dataset_name=None, output_dir="outputs/enhanced"):
    best_threshold, best_f1 = 0.0, 0.0
    thresholds = np.linspace(0, 2, 2001)
    for thresh in thresholds:
        y_pred_temp = (distances <= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_temp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh

    y_pred = (distances <= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

    acc = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)
    tpr = tp / (tp + fn + 1e-6)
    tnr = tn / (tn + fp + 1e-6)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rocauc = roc_auc_score(y_true, -distances) if len(np.unique(y_true)) == 2 else float('nan')

    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    return {
        "accuracy": acc, "f1": f1, "rocauc": rocauc,
        "far": far, "frr": frr, "tpr": tpr, "tnr": tnr,
        "f1_threshold": best_threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

# Params
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (155, 220, 1)
weights_dir = 'enhanced_weights'
os.makedirs(weights_dir, exist_ok=True)
results_csv_path = "outputs/enhanced/results.csv"
os.makedirs("outputs/enhanced", exist_ok=True)

datasets = {
    "CEDAR": {
        "path": "Dataset/CEDAR",
        "train_writers": list(range(260, 300)),
        "test_writers": list(range(300, 315))
    }
}

if not os.path.exists(results_csv_path):
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Accuracy", "F1 Score", "ROC AUC", "FAR", "FRR", "TPR", "TNR", "F1 Threshold", "TP", "FP", "TN", "FN"])

for dataset_name, config in datasets.items():
    print(f"\n📦 Processing Enhanced Model for Dataset: {dataset_name}")

    generator = SignatureDataGenerator(
        dataset={dataset_name: config},
        img_height=IMG_SHAPE[0],
        img_width=IMG_SHAPE[1],
        batch_sz=BATCH_SIZE,
    )

    train_dataset = generator.get_triplet_train(use_clahe=True)

    model = build_triplet_network(IMG_SHAPE)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=triplet_loss(margin=1))

    print("🧠 Training...")
    model.fit(train_dataset, steps_per_epoch=len(generator.train_writers), epochs=EPOCHS, verbose=2)

    base_net = model.get_layer("base_network")
    base_net.save_weights(f"{weights_dir}/{dataset_name}_base_run{run_id}.weights.h5")
    print(f"✅ Base network weights saved for {dataset_name}")

    test_pairs, test_labels = generator.generate_pairs(split='test', use_clahe=True)
    test_img1 = np.array([pair[0] for pair in test_pairs])
    test_img2 = np.array([pair[1] for pair in test_pairs])

    embedding_model = create_base_network(IMG_SHAPE)
    embedding_model.load_weights(f"{weights_dir}/{dataset_name}_base_run{run_id}.weights.h5")

    emb1 = embedding_model.predict(test_img1, batch_size=128)
    emb2 = embedding_model.predict(test_img2, batch_size=128)

    emb1 /= (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
    emb2 /= (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)

    distances = np.linalg.norm(emb1 - emb2, axis=1)

    metrics = evaluate_classification_metrics(test_labels, distances, dataset_name=dataset_name)

    with open(results_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset_name,
            metrics["accuracy"],
            metrics["f1"],
            metrics["rocauc"],
            metrics["far"],
            metrics["frr"],
            metrics["tpr"],
            metrics["tnr"],
            metrics["f1_threshold"],
            metrics["tp"],
            metrics["fp"],
            metrics["tn"],
            metrics["fn"]
        ])

    print(f"✅ Metrics saved for {dataset_name}")
