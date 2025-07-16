import os
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from tensorflow.keras.models import load_model
from keras.config import enable_unsafe_deserialization
from tensorflow.keras.saving import register_keras_serializable
from SignatureDataGenerator import SignatureDataGenerator  # Update path if needed

# === Custom functions for Keras deserialization
enable_unsafe_deserialization()

@register_keras_serializable()
def l2_normalize_layer(x):
    return tf.math.l2_normalize(x, axis=1)

@register_keras_serializable()
def stack_triplet(x):
    return tf.stack(x, axis=1)

@register_keras_serializable(package="Custom", name="loss")
def triplet_loss(margin=1):
    def loss(y_true, y_pred):
        anchor = y_pred[:, 0]
        positive = y_pred[:, 1]
        negative = y_pred[:, 2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

# === Configuration
ENHANCED_MODEL_PATH = "../../enhanced_weights/enhanced_CEDAR.keras"
IMG_SHAPE = (155, 220)
BATCH_SIZE = 128
MARGIN = 1.0

datasets = {
    "CEDAR": {
        "path": "../Dataset/CEDAR",
        "train_writers": list(range(260, 300)),
        "test_writers": list(range(300, 315))
    }
}

# === Load Enhanced Model
enhanced_weights_dir = "../../enhanced_weights"
enhanced_model_path = os.path.join(enhanced_weights_dir, "enhanced_CEDAR.keras")

enhanced_model = load_model(
    enhanced_model_path,
    custom_objects={
        "triplet_loss": triplet_loss,
        "l2_normalize_layer": l2_normalize_layer,
        "stack_triplet": stack_triplet
    }
)
base_network = enhanced_model.get_layer('base_network')
print("✅ Enhanced model loaded and base network extracted.")

# === Function to compute distances and triplet loss
def calculate_triplet_loss_on_embeddings(model, anchor, positive, negative, margin=1.0):
    anchor_emb = model.predict(np.expand_dims(anchor, axis=0), verbose=0)
    positive_emb = model.predict(np.expand_dims(positive, axis=0), verbose=0)
    negative_emb = model.predict(np.expand_dims(negative, axis=0), verbose=0)

    post_dist = euclidean(anchor_emb.flatten(), positive_emb.flatten())
    neg_dist = euclidean(anchor_emb.flatten(), negative_emb.flatten())
    
    raw_loss_value = post_dist - neg_dist + margin
    loss = max(raw_loss_value, 0)
    
    return post_dist, neg_dist, raw_loss_value, loss


# === Function to evaluate triplets and export CSV
def evaluate_triplets_and_export(generator, model, output_csv, margin=1.0, split='test'):
    writers = generator.test_writers if split == 'test' else generator.train_writers
    triplets = generator.get_triplet_data(writers)
    flat_triplets = [(a.numpy(), p.numpy(), n.numpy()) for (a, p, n), _ in triplets.unbatch()]

    records = []
    for idx, (anchor, positive, negative) in enumerate(flat_triplets):
        post_dist, neg_dist, raw_loss_value, loss = calculate_triplet_loss_on_embeddings(
            model, anchor, positive, negative, margin
        )
        records.append({
            'triplet_index': idx,
            'post_dist': round(post_dist, 4),
            'neg_dist': round(neg_dist, 4),
            'raw_loss_value': round(raw_loss_value, 4),
            'loss': round(loss, 4)
        })


    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"📄 Exported: {output_csv} with {len(records)} triplets")
    return df


# === Main execution
if __name__ == "__main__":
    for dataset_name, config in datasets.items():
        print(f"\n📦 Processing Enhanced Model for Dataset: {dataset_name}")

        generator = SignatureDataGenerator(
            dataset={dataset_name: config},
            img_height=IMG_SHAPE[0],
            img_width=IMG_SHAPE[1],
            batch_sz=BATCH_SIZE
        )
        print("Test Writers:", generator.test_writers)
        output_csv = f"{dataset_name}_enhanced_triplet_loss_test_writers.csv"
        df = evaluate_triplets_and_export(generator, base_network, output_csv, margin=MARGIN, split='test')        
        print(df.head())
