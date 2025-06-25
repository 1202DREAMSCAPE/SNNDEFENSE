import os
from tensorflow.keras.models import load_model
from embeddings import generate_reference_embeddings
from preprocess import preprocess_signature

def get_reference_signatures(dataset_path):
    """
    Automatically scan the dataset folder to find reference signatures for all writers.
    Assumes each writer has a folder containing 'genuine/' and 'forged/' subfolders.
    Uses the first signature in the 'genuine/' folder as the reference signature.
    """
    reference_signatures = {}
    cedar_path = os.path.join(dataset_path, "cedar")  # Path to the cedar dataset
    for writer_folder in os.listdir(cedar_path):
        writer_dir = os.path.join(cedar_path, writer_folder)
        if os.path.isdir(writer_dir):  # Ensure it's a writer folder
            genuine_dir = os.path.join(writer_dir, "genuine")
            if os.path.exists(genuine_dir):  # Ensure the genuine folder exists
                genuine_signatures = os.listdir(genuine_dir)
                if genuine_signatures:  # Ensure there are genuine signatures
                    # Use the first genuine signature as the reference
                    reference_signature_path = os.path.join(genuine_dir, genuine_signatures[0])
                    reference_signatures[writer_folder] = reference_signature_path
    return reference_signatures

# Step 1: Define paths for the dataset and weights directory
dataset_path = "../dataset"  # Path to the dataset folder
enhanced_weights_dir = "../enhanced_weights"  # Path to the folder containing enhanced model weights
base_weights_dir = "../base_weights"  # Path to the folder containing base model weights

# Step 2: Automatically build the reference_signatures dictionary
reference_signatures = get_reference_signatures(dataset_path)

# Step 3: Load the enhanced model directly
enhanced_model_path = os.path.join(enhanced_weights_dir, "enhanced_CEDAR.h5")
enhanced_model = load_model(
    enhanced_model_path,
    custom_objects={"triplet_loss": triplet_loss}  # Register custom loss
)

# Step 4: Generate reference embeddings for the enhanced model
generate_reference_embeddings(reference_signatures, enhanced_model, "enhanced_reference_embeddings.pkl")
print("Enhanced reference embeddings generated successfully!")

# Step 5: Load the base model directly
base_model_path = os.path.join(base_weights_dir, "base_CEDAR_siamese_model.h5")
base_model = load_model(base_model_path)

# Step 6: Generate reference embeddings for the base model
generate_reference_embeddings(reference_signatures, base_model, "base_reference_embeddings.pkl")
print("Base reference embeddings generated successfully!")