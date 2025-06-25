# Signature Verification System

This repository implements a signature verification system using Siamese neural networks. It includes preprocessing, model training, evaluation, and visualization scripts, along with pre-trained models and datasets.

## Folder Structure

### `Dataset/`
Contains datasets for signature verification:
- `BHSig260_Bengali/` - Bengali signature dataset.
- `BHSig260_Hindi/` - Hindi signature dataset.
- `CEDAR/` - CEDAR signature dataset.

### `enhanced_weights/`
Contains enhanced model weights:
- `enhanced_CEDAR.keras` - Pre-trained enhanced model for CEDAR dataset.

### `models/`
Contains pre-trained base models:
- `base_BHSig260_Bengali_siamese_model.h5` - Bengali dataset model.
- `base_BHSig260_Hindi_siamese_model.h5` - Hindi dataset model.
- `base_CEDAR_siamese_model.keras` - CEDAR dataset model.

### `outputs/`
Stores generated outputs:
- `logs/` - Logs for dataset runs (e.g., `CEDAR_run1_pairs.csv`).
- `base/` - Results and visualizations for base models.

### `individual_sop/`
Scripts for individual preprocessing and evaluation:
- `base_raw.py` - Handles raw preprocessing and model training.
- `baseweval.py` - Evaluates base models.
- `clahe.py` - Applies CLAHE preprocessing.
- `collatefarfrr.py` - Collates FAR/FRR metrics.
- `enhancedweval.py` - Evaluates enhanced models.
- `triplet.py` - Implements triplet loss-based training.

### `signature_verification/`
Contains the frontend and backend for the signature verification system:
- `frontend/templates/index.html` - Web interface for uploading signatures and selecting models.
- `SignatureDataGenerator.py` - Generates data for training and evaluation.

### `venv/`
Virtual environment for Python dependencies.

## Scripts

### `base.py`
Processes datasets using base models and generates training pairs. Outputs logs to `outputs/logs/`.

### `enhanced.py`
Processes datasets using enhanced models with CLAHE preprocessing.

### `f1-thresh.py`
Calculates F1 thresholds for model evaluation.

### `modelcheck.py`
Checks model integrity and compatibility.

## How to Run

1. **Set Up Environment**:
   - Install dependencies using `pip install -r requirements.txt`.

2. **Run the Web Interface**:
   - Navigate to `signature_verification/`.
   - Start the Flask server:
     ```sh
     flask run
     ```
   - Access the interface at `http://127.0.0.1:5000`.

3. **Train Models**:
   - Use scripts in  for training and evaluation:
     ```sh
     python individual_sop/base_raw.py
     ```

4. **Generate Visualizations**:
   - Run preprocessing scripts:
     ```sh
     python individual_sop/clahe.py
     ```

## License
This project is licensed under the MIT License.