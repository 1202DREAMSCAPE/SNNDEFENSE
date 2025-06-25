import os
from tensorflow.keras.models import load_model
from embeddings import generate_reference_embeddings
from preprocess import preprocess_signature

def get_reference_signatures(dataset_path):
    reference_signatures = {}
    cedar_path = os.path.join(dataset_path, "cedar")
    for writer_folder in os.listdir(cedar_path):
        writer_dir = os.path.join(cedar_path, writer_folder)
        if os.path.isdir(writer_dir):
            genuine_dir = os.path.join(writer_dir, "genuine")
            if os.path.exists(genuine_dir):
                genuine_signatures = os.listdir(genuine_dir)
                if genuine_signatures:
                    reference_signature_path = os.path.join(genuine_dir, genuine_signatures[0])
                    reference_signatures[writer_folder] = reference_signature_path
    return reference_signatures

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
dataset_path = "../dataset"
enhanced_weights_dir = "../enhanced_weights"
base_weights_dir = "../base_weights"

# Step 2: Automatically build the reference_signatures dictionary
reference_signatures = get_reference_signatures(dataset_path)

# Step 3: Load the enhanced model directly
try:
    enhanced_model_path = os.path.join(enhanced_weights_dir, "enhanced_CEDAR.keras")
    enhanced_model = load_model(
        enhanced_model_path,
        custom_objects={"triplet_loss": triplet_loss()}
    )
except Exception as e:
    print(f"Error loading enhanced model: {e}")
    exit()

# Step 4: Generate reference embeddings for the enhanced model
generate_reference_embeddings(reference_signatures, enhanced_model, "enhanced_reference_embeddings.pkl")
print("Enhanced reference embeddings generated successfully!")

# Step 5: Load the base model directly
try:
    base_model_path = os.path.join(base_weights_dir, "base_CEDAR_siamese_model.keras")
    base_model = load_model(base_model_path)
except Exception as e:
    print(f"Error loading base model: {e}")
    exit()

# Step 6: Generate reference embeddings for the base model
generate_reference_embeddings(reference_signatures, base_model, "base_reference_embeddings.pkl")
print("Base reference embeddings generated successfully!")