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


if __name__ == "__main__":
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

    os.makedirs("outputs/base", exist_ok=True)
    results_csv_path = "outputs/base/results.csv"

    # Ensure the CSV file has a header if it doesn't exist
    if not os.path.exists(results_csv_path):
        with open(results_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Accuracy", "F1 Score", "ROC AUC", "FAR", "FRR", "Youden Threshold"])

    results = []

    for dataset_name, config in datasets.items():
        print(f"\n📦 Processing Siamese Model for Dataset: {dataset_name}")

        # Load and generate all training pairs
        generator = SignatureDataGenerator(
            dataset={dataset_name: config},
            img_height=IMG_SHAPE[0],
            img_width=IMG_SHAPE[1],
            batch_sz=BATCH_SIZE,
        )
        pairs, labels, meta = generator.generate_pairs(
            use_raw=False,
            return_metadata=True,
            log_csv_path=f"outputs/logs/{dataset_name}_run{run_id}_pairs.csv")
        labels = np.array(labels).astype(np.int32)

        # Shuffle pairs
        combined = list(zip(pairs, labels))
        np.random.shuffle(combined)
        pairs, labels = zip(*combined)
        pairs = list(pairs)
        labels = np.array(labels).astype(np.int32)

        # Separate image pairs
        img1 = np.array([pair[0] for pair in pairs])
        img2 = np.array([pair[1] for pair in pairs])

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
            [img1, img2], labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2
        )

        # ========== Save the model ==========
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = f"{model_dir}/base_{dataset_name}_siamese_model.h5"
        model.save(model_save_path)
        print(f"💾 Model saved to: {model_save_path}")

        train_time = time.time() - start_time
        print(f"⏱ Training completed in {train_time:.2f} seconds")

