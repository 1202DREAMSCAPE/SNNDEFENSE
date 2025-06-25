from tensorflow.keras.models import load_model
import sys
import os

# Ensure root-level imports for base and enhanced architectures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from base import build_siamese_network         # For base model (contrastive loss)
from enhanced import build_triplet_network     # For enhanced model (triplet loss)

def load_siamese_model(model_path):
    """
    Load the base Siamese model trained with contrastive loss (MinMax).
    """
    model = build_siamese_network(input_shape=(155, 220, 1))
    model.load_weights(model_path)
    return model

def load_enhanced_siamese_model(model_path):
    """
    Load the enhanced Siamese model trained with triplet loss (CLAHE).
    """
    model = build_triplet_network(input_shape=(155, 220, 1))
    model.load_weights(model_path)
    return model
