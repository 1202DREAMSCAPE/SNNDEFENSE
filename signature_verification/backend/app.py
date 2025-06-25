from flask import Flask, request, jsonify, render_template, send_file
import os
import sys
import cv2
import pickle
from tensorflow.keras.models import Model
# Ensure root-level imports for base and enhanced architectures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import SignatureDataGenerator

# Import custom modules
from preprocess import preprocess_signature
from model import load_siamese_model, load_enhanced_siamese_model
from utils import verify_signature

# Setup Flask
app = Flask(__name__, template_folder="../frontend/templates")
os.makedirs("temp", exist_ok=True)

# Load precomputed reference embeddings
with open("../base_reference_embeddings.pkl", "rb") as f:
    base_reference_embeddings = pickle.load(f)

with open("../enhanced_reference_embeddings.pkl", "rb") as f:
    enhanced_reference_embeddings = pickle.load(f)

# Load models
base_siamese_model = load_siamese_model("../../models/base_CEDAR_siamese_model.keras")
base_model = base_siamese_model.get_layer("base_network")

enhanced_siamese_model = load_enhanced_siamese_model("../../enhanced_weights/enhanced_CEDAR.keras")
enhanced_model = enhanced_siamese_model.get_layer("base_network")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/base", methods=["POST"])
def verify_base():
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]
    
    # Save uploaded file
    file_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(file_path)

    # Preprocess with MinMax
    minmax_img = preprocess_signature(file_path, preprocessing_type="minmax")
    minmax_img_path = f"temp/minmax_{uploaded_file.filename}"
    cv2.imwrite(minmax_img_path, minmax_img.squeeze() * 255)

    result = verify_signature(
        claimed_writer_id=writer_id,
        uploaded_signature_path=file_path,
        reference_embeddings=base_reference_embeddings,
        model=base_model,
        threshold=0.5
    )

    result.update({
        "raw_image_url": f"/temp/{uploaded_file.filename}",
        "minmax_image_url": f"/temp/minmax_{uploaded_file.filename}",
        "clahe_image_url": ""  # not used in base model
    })
    return jsonify(result)


@app.route("/enhanced", methods=["POST"])
def verify_enhanced():
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]

    # Save uploaded file
    raw_image_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(raw_image_path)

    # Preprocess uploaded image
    processed_clahe = preprocess_signature(raw_image_path, preprocessing_type="clahe")
    clahe_image_path = f"temp/clahe_{uploaded_file.filename}"
    cv2.imwrite(clahe_image_path, processed_clahe.squeeze() * 255)

    # Get embedding from model
    anchor_embedding = enhanced_model.get_layer("base_network").predict(
        np.expand_dims(processed_clahe, axis=0)
    )[0]

    # Find POSITIVE: closest sample from same writer
    min_pos_dist = float("inf")
    positive_path = None

    for ref in enhanced_reference_embeddings[writer_id]:
        dist = np.linalg.norm(anchor_embedding - ref["embedding"])
        if dist < min_pos_dist:
            min_pos_dist = dist
            positive_path = ref["path"]

    # Find NEGATIVE: closest sample from any *other* writer
    min_neg_dist = float("inf")
    negative_path = None

    for other_writer, refs in enhanced_reference_embeddings.items():
        if other_writer == writer_id:
            continue
        for ref in refs:
            dist = np.linalg.norm(anchor_embedding - ref["embedding"])
            if dist < min_neg_dist:
                min_neg_dist = dist
                negative_path = ref["path"]

    # Run classification for base result (optional — reuse your verify_signature)
    result_data = verify_signature(
        claimed_writer_id=writer_id,
        uploaded_signature_path=raw_image_path,
        reference_embeddings=enhanced_reference_embeddings,
        model=enhanced_model.get_layer("base_network"),
        threshold=0.5
    )

    # Append all images for frontend preview
    result_data.update({
        "raw_image_url": f"/temp/{uploaded_file.filename}",
        "clahe_image_url": f"/temp/clahe_{uploaded_file.filename}",
        "minmax_image_url": "",  # optional
        "positive_image_url": f"/{positive_path}",
        "negative_image_url": f"/{negative_path}"
    })

    return jsonify(result_data)



@app.route("/get_writers", methods=["GET"])
def get_writers():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/cedar"))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    writer_ids = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    return jsonify(writer_ids)


@app.route("/verify", methods=["POST"])
def verify_signature_images():
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]

    file_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(file_path)

    # CLAHE
    clahe_img = preprocess_signature(file_path, preprocessing_type="clahe")
    cv2.imwrite(f"temp/clahe_{uploaded_file.filename}", clahe_img.squeeze() * 255)

    # MinMax
    minmax_img = preprocess_signature(file_path, preprocessing_type="minmax")
    cv2.imwrite(f"temp/minmax_{uploaded_file.filename}", minmax_img.squeeze() * 255)

    return jsonify({
        "result": "Verification complete!",
        "raw_image_url": f"/temp/{uploaded_file.filename}",
        "clahe_image_url": f"/temp/clahe_{uploaded_file.filename}",
        "minmax_image_url": f"/temp/minmax_{uploaded_file.filename}"
    })

@app.route("/get_triplet_example", methods=["POST"])
def get_triplet_example():
    """
    Return an anchor-positive-negative triplet given the selected writer and uploaded signature.
    """
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]

    # Save uploaded file
    raw_image_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(raw_image_path)

    # Initialize your SignatureDataGenerator with the same config
    generator = SignatureDataGenerator(
        dataset={"CEDAR": {
            "path": "Dataset/CEDAR",  # Or wherever your dataset path is
            "train_writers": list(range(260, 300))
        }},
        img_height=155,
        img_width=220,
        batch_sz=1
    )

    # Create a synthetic triplet using the uploaded image as anchor
    triplets = generator.generate_triplets(
        dataset_path="../Dataset/CEDAR",
        writer=int(writer_id),
        use_clahe=True
    )

    if not triplets:
        return jsonify({"error": "Triplet not generated"}), 400

    # Save first triplet to temp/
    anchor, positive, negative = triplets[0]

    anchor_path = f"temp/anchor_{uploaded_file.filename}"
    positive_path = f"temp/positive_{uploaded_file.filename}"
    negative_path = f"temp/negative_{uploaded_file.filename}"

    cv2.imwrite(anchor_path, anchor.squeeze() * 255)
    cv2.imwrite(positive_path, positive.squeeze() * 255)
    cv2.imwrite(negative_path, negative.squeeze() * 255)

    return jsonify({
        "anchor_url": f"/temp/anchor_{uploaded_file.filename}",
        "positive_url": f"/temp/positive_{uploaded_file.filename}",
        "negative_url": f"/temp/negative_{uploaded_file.filename}"
    })

@app.route("/get_triplet_example", methods=["POST"])
def get_triplet_example():
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]
    raw_image_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(raw_image_path)

    generator = SignatureDataGenerator(
        dataset={"CEDAR": {
            "path": "Dataset/CEDAR",
            "train_writers": list(range(260, 300))
        }},
        img_height=155,
        img_width=220,
        batch_sz=1
    )

    triplets = generator.generate_triplets(
        dataset_path="Dataset/CEDAR",
        writer=int(writer_id),
        use_clahe=True
    )

    if not triplets:
        return jsonify({"error": "Triplet not generated"}), 400

    anchor, positive, negative = triplets[0]

    # Save triplet images
    anchor_path = f"temp/anchor_{uploaded_file.filename}"
    positive_path = f"temp/positive_{uploaded_file.filename}"
    negative_path = f"temp/negative_{uploaded_file.filename}"

    cv2.imwrite(anchor_path, anchor.squeeze() * 255)
    cv2.imwrite(positive_path, positive.squeeze() * 255)
    cv2.imwrite(negative_path, negative.squeeze() * 255)

    # Get embeddings
    base_model = enhanced_model.get_layer("base_network")
    a_emb = base_model.predict(np.expand_dims(anchor, axis=0), verbose=0)
    p_emb = base_model.predict(np.expand_dims(positive, axis=0), verbose=0)
    n_emb = base_model.predict(np.expand_dims(negative, axis=0), verbose=0)

    # Normalize if necessary
    a_emb = tf.math.l2_normalize(a_emb, axis=1)
    p_emb = tf.math.l2_normalize(p_emb, axis=1)
    n_emb = tf.math.l2_normalize(n_emb, axis=1)

    # Compute distances
    ap_dist = np.linalg.norm(a_emb - p_emb)
    an_dist = np.linalg.norm(a_emb - n_emb)

    return jsonify({
        "anchor_url": f"/temp/anchor_{uploaded_file.filename}",
        "positive_url": f"/temp/positive_{uploaded_file.filename}",
        "negative_url": f"/temp/negative_{uploaded_file.filename}",
        "anchor_positive_dist": round(float(ap_dist), 4),
        "anchor_negative_dist": round(float(an_dist), 4)
    })

@app.route("/temp/<filename>")
def serve_temp_file(filename):
    return send_file(f"temp/{filename}")


if __name__ == "__main__":
    app.run(debug=True)
