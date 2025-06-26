import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from embeddings import generate_reference_embeddings
from keras.config import enable_unsafe_deserialization
from tensorflow.keras.saving import register_keras_serializable
from preprocess import preprocess_signature

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

def get_reference_signatures(dataset_path):
    reference_signatures = {}
    cedar_path = os.path.join(dataset_path, "cedar")

    for writer_folder in os.listdir(cedar_path):
        genuine_dir = os.path.join(cedar_path, writer_folder, "genuine")
        if os.path.isdir(genuine_dir):
            img_paths = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if img_paths:
                reference_signatures[writer_folder] = img_paths
    return reference_signatures

# Paths
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "../Dataset")
enhanced_weights_dir = os.path.join(base_dir, "../../enhanced_weights")
base_weights_dir = os.path.join(base_dir, "../../models")

# Load reference images
reference_signatures = get_reference_signatures(dataset_path)

# --- Enhanced Model ---
try:
    enhanced_model_path = os.path.join(enhanced_weights_dir, "enhanced_CEDAR.keras")
    enhanced_model = load_model(
        enhanced_model_path,
        custom_objects={
            "triplet_loss": triplet_loss,
            "l2_normalize_layer": l2_normalize_layer,
            "stack_triplet": stack_triplet
        }
    )
    generate_reference_embeddings(
        reference_signatures,
        enhanced_model,
        output_path="enhanced_reference_embeddings.pkl",
        model_type="enhanced"
    )
except Exception as e:
    print(f"[ERROR] Enhanced model loading failed: {e}")

# --- Base Model ---
try:
    base_model_path = os.path.join(base_weights_dir, "base_CEDAR_siamese_model.keras")
    base_siamese_model = load_model(base_model_path)
    base_model = base_siamese_model.get_layer("base_network")

    generate_reference_embeddings(
        reference_signatures,
        base_model,
        output_path="base_reference_embeddings.pkl",
        model_type="base"
    )
except Exception as e:
    print(f"[ERROR] Base model loading failed: {e}")
