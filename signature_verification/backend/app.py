from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import cv2
import pickle
import numpy as np
import tensorflow as tf
import uuid

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from preprocess import preprocess_signature
from model import load_siamese_model, load_enhanced_siamese_model
from utils import verify_signature

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

    raw_path = f"{STATIC_TEMP}/{uploaded_file.filename}"
    clahe_path = f"{STATIC_TEMP}/clahe_{uploaded_file.filename}"
    uploaded_file.save(raw_path)

    clahe_img = preprocess_signature(raw_path, preprocessing_type="clahe")
    cv2.imwrite(clahe_path, clahe_img.squeeze() * 255)

    uploaded_emb = enhanced_model.predict(np.expand_dims(clahe_img, axis=0), verbose=0)[0].flatten()

    min_pos_dist, positive_path = float("inf"), None
    for ref in enhanced_reference_embeddings[writer_id]:
        dist = np.linalg.norm(uploaded_emb - ref["embedding"])
        if dist < min_pos_dist:
            min_pos_dist = dist
            positive_path = ref["path"]

    min_neg_dist, negative_path = float("inf"), None
    for other_writer, refs in enhanced_reference_embeddings.items():
        if other_writer == writer_id:
            continue
        for ref in refs:
            dist = np.linalg.norm(uploaded_emb - ref["embedding"])
            if dist < min_neg_dist:
                min_neg_dist = dist
                negative_path = ref["path"]

    result = verify_signature(
        claimed_writer_id=writer_id,
        uploaded_signature_path=raw_path,
        reference_embeddings=enhanced_reference_embeddings,
        model=enhanced_model,
        threshold=0.5
    )

    result.update({
        "raw_image_url": f"/static/temp/{uploaded_file.filename}",
        "clahe_image_url": f"/static/temp/clahe_{uploaded_file.filename}",
        "positive_image_url": f"/{positive_path}",
        "negative_image_url": f"/{negative_path}",
        "positive_distance": float(round(float(min_pos_dist), 4)),
        "negative_distance": float(round(float(min_neg_dist), 4)),
        "distance": float(result["distance"]),
        "threshold": float(result["threshold"]),
        "confidence": float(result["confidence"])
    })


    return jsonify(result)

@app.route("/get_triplet_example", methods=["POST"])
def get_triplet_example():
    try:
        uploaded_file = request.files["signature"]
        writer_id = request.form["writer_id"]
        filename = request.form.get("filename", f"anchor_{uuid.uuid4().hex}.png")
        anchor_path = os.path.join(STATIC_TEMP, filename)
        uploaded_file.save(anchor_path)

        anchor_img = preprocess_signature(anchor_path, preprocessing_type="clahe")
        anchor_emb = enhanced_model.predict(np.expand_dims(anchor_img, axis=0), verbose=0)[0]

        min_pos_dist, pos_path = float("inf"), None
        for ref in enhanced_reference_embeddings[writer_id]:
            dist = np.linalg.norm(anchor_emb - ref["embedding"])
            if dist < min_pos_dist:
                min_pos_dist = dist
                pos_path = ref["path"]

        min_neg_dist, neg_path = float("inf"), None
        for other_writer, refs in enhanced_reference_embeddings.items():
            if other_writer == writer_id:
                continue
            for ref in refs:
                dist = np.linalg.norm(anchor_emb - ref["embedding"])
                if dist < min_neg_dist:
                    min_neg_dist = dist
                    neg_path = ref["path"]

        def save_preview_image(src, dest_name):
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            dest = os.path.join(STATIC_TEMP, dest_name)
            cv2.imwrite(dest, img)
            return f"/static/temp/{dest_name}"

        anchor_url = f"/static/temp/{filename}"
        positive_url = save_preview_image(pos_path, "triplet_positive.png")
        negative_url = save_preview_image(neg_path, "triplet_negative.png")

        return jsonify({
            "anchor_url": anchor_url,
            "positive_url": positive_url,
            "negative_url": negative_url,
            "anchor_positive_dist": float(round(min_pos_dist, 4)),
            "anchor_negative_dist": float(round(min_neg_dist, 4))
        })


    except Exception as e:
        print("[ERROR] Triplet generation failed:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/get_writers", methods=["GET"])
def get_writers():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/cedar"))
    writer_ids = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])
    return jsonify(writer_ids)

@app.route("/static/temp/<path:filename>")
def serve_temp(filename):
    return send_from_directory(STATIC_TEMP, filename)

if __name__ == "__main__":
    app.run(debug=True)
