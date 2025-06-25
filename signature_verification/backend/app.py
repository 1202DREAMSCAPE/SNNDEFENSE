from flask import Flask, request, jsonify, render_template, send_file
print("Importing preprocess...")
from preprocess import preprocess_signature

print("Importing embeddings...")
from embeddings import load_reference_embeddings

print("Importing model...")
from model import load_siamese_model, load_enhanced_siamese_model

print("Importing utils...")
from utils import verify_signature
import cv2
import os

app = Flask(__name__)

# Load precomputed reference embeddings
base_reference_embeddings = load_reference_embeddings("base_reference_embeddings.pkl")
enhanced_reference_embeddings = load_reference_embeddings("enhanced_reference_embeddings.pkl")

# Load models
base_model = load_siamese_model("base_CEDAR_siamese_model.h5")
enhanced_model = load_enhanced_siamese_model("enhanced_CEDAR.h5")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # HTML form for upload and writer selection

@app.route("/base", methods=["POST"])
def verify_base():
    """
    Verify the signature using the base model with MinMax preprocessing.
    """
    # Get uploaded file and writer_id
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]
    
    # Save the uploaded file temporarily
    uploaded_file_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(uploaded_file_path)
    
    # Preprocess the image using MinMax normalization
    processed_image = preprocess_signature(uploaded_file_path, preprocessing_type="minmax")
    
    # Verify the signature using the base model
    result = verify_signature(
        claimed_writer_id=writer_id,
        uploaded_signature_path=uploaded_file_path,
        reference_embeddings=base_reference_embeddings,
        model=base_model,
        threshold_type="youden"
    )
    return jsonify(result)

@app.route("/enhanced", methods=["POST"])
def verify_enhanced():
    """
    Verify the signature using the enhanced model with CLAHE preprocessing.
    """
    # Get uploaded file and writer_id
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]
    
    # Save the uploaded file temporarily
    uploaded_file_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(uploaded_file_path)
    
    # Preprocess the image using CLAHE
    processed_image = preprocess_signature(uploaded_file_path, preprocessing_type="clahe")
    
    # Verify the signature using the enhanced model
    result = verify_signature(
        claimed_writer_id=writer_id,
        uploaded_signature_path=uploaded_file_path,
        reference_embeddings=enhanced_reference_embeddings,
        model=enhanced_model,
        threshold_type="f1"
    )
    return jsonify(result)

@app.route("/get_writers", methods=["GET"])
def get_writers():
    """
    Fetch all writer IDs from the dataset folder.
    Assumes the dataset is structured as 'dataset/cedar/writer/genuine'.
    """
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/cedar"))

    # Debugging: Print and verify the dataset path
    print("Resolved dataset path:", dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    writer_ids = []

    for writer_folder in os.listdir(dataset_path):
        writer_dir = os.path.join(dataset_path, writer_folder)
        if os.path.isdir(writer_dir):  # Ensure it's a writer folder
            writer_ids.append(writer_folder)

    return jsonify(writer_ids)

@app.route("/verify", methods=["POST"])
def verify_signature_images():
    """
    Verify the uploaded signature and return raw, MinMax-preprocessed, and CLAHE-enhanced images.
    """
    # Get uploaded file
    uploaded_file = request.files["signature"]
    writer_id = request.form["writer_id"]

    # Save the uploaded file temporarily
    raw_image_path = f"temp/{uploaded_file.filename}"
    uploaded_file.save(raw_image_path)

    # Preprocess the image using CLAHE
    processed_clahe_image = preprocess_signature(raw_image_path, preprocessing_type="clahe")
    clahe_image_path = f"temp/clahe_{uploaded_file.filename}"
    cv2.imwrite(clahe_image_path, processed_clahe_image.squeeze() * 255)  # Convert back to uint8 for saving

    # Preprocess the image using MinMax normalization
    processed_minmax_image = preprocess_signature(raw_image_path, preprocessing_type="minmax")
    minmax_image_path = f"temp/minmax_{uploaded_file.filename}"
    cv2.imwrite(minmax_image_path, processed_minmax_image.squeeze() * 255)  # Convert back to uint8 for saving

    # Return raw, MinMax-preprocessed, and CLAHE-enhanced images
    return jsonify({
        "result": "Verification complete!",
        "raw_image_url": f"/temp/{uploaded_file.filename}",
        "clahe_image_url": f"/temp/clahe_{uploaded_file.filename}",
        "minmax_image_url": f"/temp/minmax_{uploaded_file.filename}"
    })

@app.route("/temp/<filename>")
def serve_temp_file(filename):
    """
    Serve temporary files (raw and CLAHE-enhanced images).
    """
    return send_file(f"temp/{filename}")

if __name__ == "__main__":
    app.run(debug=True)