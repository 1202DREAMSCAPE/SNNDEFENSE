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
    encoded_anchor = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        output_shape=lambda input_shape: input_shape
    )(encoded_anchor)
    encoded_positive = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        output_shape=lambda input_shape: input_shape
    )(encoded_positive)
    encoded_negative = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        output_shape=lambda input_shape: input_shape
    )(encoded_negative)

    # Stack embeddings into a single tensor (batch_size, 3, embedding_dim)
    merged_output = layers.Lambda(
        lambda x: tf.stack(x, axis=1),
        output_shape=lambda input_shapes: (input_shapes[0][0], 3, input_shapes[0][1])
    )([encoded_anchor, encoded_positive, encoded_negative])

    # Create the triplet model
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)
    return model

if __name__ == "__main__":
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
        # "BHSig260_Bengali": {
        #     "path": "Dataset/BHSig260_Bengali",
        #     "train_writers": list(range(1, 71)),
        #     "test_writers": list(range(71, 101))
        # },
        # "BHSig260_Hindi": {
        #     "path": "Dataset/BHSig260_Hindi",
        #     "train_writers": list(range(101, 191)),
        #     "test_writers": list(range(191, 260))
        # }
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
        train_dataset = generator.get_triplet_train(use_clahe=True,
                                                    log_csv_path=f"outputs/logs/{dataset_name}_run{run_id}__triplets.csv")
    
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
    
        # Save the model in TensorFlow SavedModel format
        enhanced_model_path = f"{weights_dir}/enhanced_{dataset_name}.keras"
        model.save(enhanced_model_path)  # No need for save_format argument
        print(f"✅ Enhanced model saved to: {enhanced_model_path}")
