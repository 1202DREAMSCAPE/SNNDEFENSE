import os
from tensorflow.keras.models import load_model
from embeddings import generate_reference_embeddings
from preprocess import preprocess_signature
from keras.config import enable_unsafe_deserialization
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

enable_unsafe_deserialization()

def get_reference_signatures(dataset_path):
    reference_signatures = {}
    cedar_path = os.path.join(dataset_path, "cedar")
    
    for writer_folder in os.listdir(cedar_path):
        writer_dir = os.path.join(cedar_path, writer_folder)
        if os.path.isdir(writer_dir):
            genuine_dir = os.path.join(writer_dir, "genuine")
            if os.path.exists(genuine_dir):
                signature_paths = []
                for fname in os.listdir(genuine_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        signature_paths.append(os.path.join(genuine_dir, fname))
                if signature_paths:
                    reference_signatures[writer_folder] = signature_paths
    return reference_signatures


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

# Step 1: Define paths for the dataset and weights directory
dataset_path = os.path.join(os.path.dirname(__file__), "../../Dataset")
enhanced_weights_dir = os.path.join(os.path.dirname(__file__), "../../enhanced_weights")
base_weights_dir = os.path.join(os.path.dirname(__file__), "../../models")


# Step 2: Automatically build the reference_signatures dictionary
reference_signatures = get_reference_signatures(dataset_path)

# Step 3: Load the enhanced model directly
try:
    enhanced_model_path = os.path.join(enhanced_weights_dir, "enhanced_CEDAR.keras")
    try:
        enhanced_model = load_model(
            enhanced_model_path,
            custom_objects={"triplet_loss": triplet_loss,
        "l2_normalize_layer": l2_normalize_layer,
        "stack_triplet": stack_triplet},
        )
    except Exception as e:
        print(f"Error loading enhanced model: {e}")
        print(f"Registered functions: {dir(tf.keras.losses)}")
        exit()
except Exception as e:
    print(f"Error loading enhanced model: {e}")
    exit()

# Step 4: Generate reference embeddings for the enhanced model
generate_reference_embeddings(reference_signatures, enhanced_model, "enhanced_reference_embeddings.pkl", model_type="enhanced")
print("Enhanced reference embeddings generated successfully!")

# Step 5: Load the base Siamese model
try:
    base_model_path = os.path.join(base_weights_dir, "base_CEDAR_siamese_model.keras")
    base_siamese_model = load_model(base_model_path)
    
    # Step 5.1: Extract the base network for embedding generation
    base_model = base_siamese_model.get_layer("base_network")
except Exception as e:
    print(f"Error loading base model: {e}")
    exit()

# Step 6: Generate reference embeddings using the extracted base model
generate_reference_embeddings(reference_signatures, base_model, "base_reference_embeddings.pkl", model_type="base")
print("Base reference embeddings generated successfully!")
