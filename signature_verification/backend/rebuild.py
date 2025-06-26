import os
import numpy as np
import pickle
from preprocess import preprocess_signature
from model import load_enhanced_siamese_model
import sys

# Ensure root-level imports for base and enhanced architectures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
model = load_enhanced_siamese_model("../../enhanced_weights/enhanced_CEDAR.keras")
 
IMG_DIR = "static/temp"
OUTPUT_FILE = "enhanced_reference_embeddings.pkl"

base_model = model.get_layer("base_network")

output = {}

print(f"🛠 Rebuilding {OUTPUT_FILE} from images in: {IMG_DIR}")

# Organize by writer based on filenames like: genuine_19.png
for fname in os.listdir(IMG_DIR):
    if not fname.startswith("genuine_"):
        continue

    writer_id = "writer_261"  # 👈 hardcoded for now, update if multiple writers
    if writer_id not in output:
        output[writer_id] = []

    image_path = os.path.join(IMG_DIR, fname)

    try:
        img = preprocess_signature(image_path, preprocessing_type="clahe")
        embedding = base_model.predict(np.expand_dims(img, axis=0), verbose=0)[0].flatten()
        output[writer_id].append({
            "embedding": embedding,
            "path": os.path.join("static", "temp", fname)  # relative for serving
        })
        print(f"✅ Processed: {fname}")
    except Exception as e:
        print(f"❌ Skipped {fname}: {e}")

# Save
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(output, f)

print(f"\n✅ Done! Saved to {OUTPUT_FILE}")
