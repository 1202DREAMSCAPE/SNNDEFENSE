from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import cv2
import pickle
import random
import shutil
import numpy as np
import tensorflow as tf

# Ensure root-level imports for base and enhanced architectures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from SignatureDataGenerator import SignatureDataGenerator
from preprocess import preprocess_signature
from model import load_siamese_model, load_enhanced_siamese_model
from utils import verify_signature

# Setup Flask
app = Flask(__name__, template_folder="../frontend/templates")
STATIC_TEMP = "static/temp"
os.makedirs(STATIC_TEMP, exist_ok=True)

# Load reference embeddings
with open("enhanced_reference_embeddings.pkl", "rb") as f:
    enhanced_reference_embeddings = pickle.load(f)

with open("base_reference_embeddings.pkl", "rb") as f:
    base_reference_embeddings = pickle.load(f)

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
    
    raw_path = f"{STATIC_TEMP}/{uploaded_file.filename}"
    minmax_path = f"{STATIC_TEMP}/minmax_{uploaded_file.filename}"
    uploaded_file.save(raw_path)

    # Preprocess (MinMax)
    minmax_img = preprocess_signature(raw_path, preprocessing_type="minmax")
    cv2.imwrite(minmax_path, minmax_img.squeeze() * 255)

    result = verify_signature(
        claimed_writer_id=writer_id,
        uploaded_signature_path=raw_path,
        reference_embeddings=base_reference_embeddings,
        model=base_model,
        threshold=0.5
    )

    result.update({
        "raw_image_url": f"/static/temp/{uploaded_file.filename}",
        "minmax_image_url": f"/static/temp/minmax_{uploaded_file.filename}",
        "clahe_image_url": ""
    })
    return jsonify(result)


@app.route("/enhanced", methods=["POST"])
def verify_enhanced():
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]

    raw_image_path = f"{STATIC_TEMP}/{uploaded_file.filename}"
    clahe_path = f"{STATIC_TEMP}/clahe_{uploaded_file.filename}"
    uploaded_file.save(raw_image_path)

    # Preprocess CLAHE
    processed_clahe = preprocess_signature(raw_image_path, preprocessing_type="clahe")
    cv2.imwrite(clahe_path, processed_clahe.squeeze() * 255)

    uploaded_emb = enhanced_model.predict(np.expand_dims(processed_clahe, axis=0), verbose=0)[0].flatten()

    # Positive
    min_pos_dist = float("inf")
    positive_path = None
    for ref_obj in enhanced_reference_embeddings[writer_id]:
        dist = np.linalg.norm(uploaded_emb - ref_obj["embedding"])
        if dist < min_pos_dist:
            min_pos_dist = dist
            positive_path = ref_obj["path"]

    # Negative
    min_neg_dist = float("inf")
    negative_path = None
    for other_writer, ref_list in enhanced_reference_embeddings.items():
        if other_writer == writer_id:
            continue
        for ref_obj in ref_list:
            dist = np.linalg.norm(uploaded_emb - ref_obj["embedding"])
            if dist < min_neg_dist:
                min_neg_dist = dist
                negative_path = ref_obj["path"]

    result_data = verify_signature(
        claimed_writer_id=writer_id,
        uploaded_signature_path=raw_image_path,
        reference_embeddings=enhanced_reference_embeddings,
        model=enhanced_model,
        threshold=0.5
    )

    result_data.update({
        "raw_image_url": f"/static/temp/{uploaded_file.filename}",
        "clahe_image_url": f"/static/temp/clahe_{uploaded_file.filename}",
        "positive_image_url": f"/{positive_path}",
        "negative_image_url": f"/{negative_path}",
        "positive_distance": float(round(min_pos_dist, 4)),
        "negative_distance": float(round(min_neg_dist, 4))
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


@app.route("/get_writer_counts", methods=["GET"])
def get_writer_counts():
    writer_counts = {}
    for writer_id, refs in enhanced_reference_embeddings.items():
        writer_counts[writer_id] = len(refs)
    return jsonify(writer_counts)


@app.route("/verify", methods=["POST"])
def verify_signature_images():
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]

    raw_path = f"{STATIC_TEMP}/{uploaded_file.filename}"
    uploaded_file.save(raw_path)

    clahe_img = preprocess_signature(raw_path, preprocessing_type="clahe")
    minmax_img = preprocess_signature(raw_path, preprocessing_type="minmax")

    clahe_path = f"{STATIC_TEMP}/clahe_{uploaded_file.filename}"
    minmax_path = f"{STATIC_TEMP}/minmax_{uploaded_file.filename}"
    cv2.imwrite(clahe_path, clahe_img.squeeze() * 255)
    cv2.imwrite(minmax_path, minmax_img.squeeze() * 255)

    return jsonify({
        "result": "Verification complete!",
        "raw_image_url": f"/static/temp/{uploaded_file.filename}",
        "clahe_image_url": f"/static/temp/clahe_{uploaded_file.filename}",
        "minmax_image_url": f"/static/temp/minmax_{uploaded_file.filename}"
    })


@app.route("/get_triplet_example", methods=["POST"])
def get_triplet_example():
    writer_id = request.form.get("writer_id")
    file = request.files.get("signature")
    uploaded_filename = request.form.get("filename")

    if not writer_id or not file:
        return jsonify({"error": "Missing writer ID or signature file"}), 400

    try:
        anchor_path = os.path.join(STATIC_TEMP, "anchor_uploaded.png")
        file.save(anchor_path)

        with open("enhanced_reference_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

        if writer_id not in embeddings:
            return jsonify({"error": "Writer not found"}), 400

        writer_refs = embeddings[writer_id]
        positive_refs = [r for r in writer_refs if uploaded_filename not in os.path.basename(r["path"])]
        if len(positive_refs) == 0:
            print(f"[WARNING] No valid positive reference left after excluding: {uploaded_filename}")
            return jsonify({"error": "Could not find valid positive sample"}), 400

        positive_ref = random.choice(positive_refs)
        positive_path = positive_ref["path"]
        positive_emb = positive_ref["embedding"]

        all_negatives = []
        for other_writer, refs in embeddings.items():
            if other_writer != writer_id:
                all_negatives.extend(refs)

        negative_ref = min(all_negatives, key=lambda ref: np.linalg.norm(positive_emb - ref["embedding"]))
        negative_path = negative_ref["path"]

        # Anchor Embedding
        anchor_img = preprocess_signature(anchor_path)
        anchor_input = np.expand_dims(anchor_img, axis=0)
        model_output = enhanced_model.predict([anchor_input]*3, verbose=0)
        anchor_emb = model_output[0][0]

        ap_dist = float(np.linalg.norm(anchor_emb - positive_emb))
        an_dist = float(np.linalg.norm(anchor_emb - negative_ref["embedding"]))

        # Copy to static/temp
        pos_dest = os.path.join(STATIC_TEMP, os.path.basename(positive_path))
        neg_dest = os.path.join(STATIC_TEMP, os.path.basename(negative_path))
        shutil.copy(positive_path, pos_dest)
        shutil.copy(negative_path, neg_dest)

        return jsonify({
            "anchor_url": f"/static/temp/{os.path.basename(anchor_path)}",
            "positive_url": f"/static/temp/{os.path.basename(positive_path)}",
            "negative_url": f"/static/temp/{os.path.basename(negative_path)}",
            "anchor_positive_dist": round(ap_dist, 4),
            "anchor_negative_dist": round(an_dist, 4),
        })

    except Exception as e:
        print(f"[ERROR] Triplet generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/static/temp/<path:filename>')
def serve_temp(filename):
    return send_from_directory(STATIC_TEMP, filename)


if __name__ == "__main__":
    app.run(debug=True)
