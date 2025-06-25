import pickle
import numpy as np
from preprocess import preprocess_signature

def generate_reference_embeddings(reference_signatures, model, output_path):
    reference_embeddings = {}
    for writer_id, signature_path in reference_signatures.items():
        signature = preprocess_signature(signature_path)
        reference_embeddings[writer_id] = model.predict(np.expand_dims(signature, axis=0))
    with open(output_path, "wb") as f:
        pickle.dump(reference_embeddings, f)

def load_reference_embeddings(embeddings_path):
    with open(embeddings_path, "rb") as f:
        return pickle.load(f)