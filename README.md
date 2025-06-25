# ✍️ Signature Verification System (Base & Enhanced Models)

A web-based signature verification system with two models:
- **Base Model** — MinMax normalized preprocessing.
- **Enhanced Model** — CLAHE preprocessing + Triplet Loss.
- Includes Triplet Viewer (anchor–positive–negative comparison).

---

## 🧰 Requirements

- Python 3.8+
- pip
- virtualenv (recommended)
- Modern browser (Chrome/Edge/Firefox)

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/1202DREAMSCAPE/signature-verification-system.git
cd signature-verification-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
flask
tensorflow
opencv-python
numpy
```

---

## 📁 Project Structure

```
signature_verification/
├── backend/
│   ├── app.py
│   ├── preprocess.py
│   ├── model.py
│   ├── utils.py
│   ├── SignatureDataGenerator.py
│   ├── temp/                             ← Temp folder for images (auto)
│   ├── base_reference_embeddings.pkl     ← Precomputed base model embeddings
│   └── enhanced_reference_embeddings.pkl ← Precomputed enhanced model embeddings
│
├── frontend/
│   └── templates/
│       └── index.html                    ← Web UI (TailwindCSS)
│
├── models/
│   └── base_CEDAR_siamese_model.keras
│
├── enhanced_weights/
│   └── enhanced_CEDAR.keras
│
├── dataset/
│   └── cedar/
│       └── writer_001/
│           ├── genuine_1.png
│           └── forged_1.png
```

---

## 📸 Dataset Format

Structure must be:

```
dataset/cedar/
├── writer_001/
│   ├── genuine_1.png
│   ├── genuine_2.png
│   ├── forged_1.png
│   └── forged_2.png
├── writer_002/
│   └── ...
```

---

## 🚀 Run the App

```bash
cd backend
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🔍 Features

- 🔁 Select writer and upload signature
- 🧠 Choose Base (MinMax) or Enhanced (CLAHE + Triplet)
- 🧪 Triplet Viewer to see Anchor, Positive, and Negative comparisons
- 📊 Shows distances and confidence values

---

## 📌 Notes

- Ensure `.pkl` files are correctly placed if using precomputed embeddings.
- You may regenerate embeddings using `SignatureDataGenerator` if dataset is updated.

---

## 👨‍💻 Maintainers

- Cj

---

