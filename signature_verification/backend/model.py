from tensorflow.keras.models import load_model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from base import build_siamese_network
from enhanced import build_triplet_network

def load_siamese_model(model_path):
    model = build_siamese_network(input_shape=(155, 220, 1))
    model.load_weights(model_path)
    return model

def load_enhanced_siamese_model(model_path):
    model = build_enhanced_siamese_network(input_shape=(155, 220, 1))
    model.load_weights(model_path)
    return model