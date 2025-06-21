import numpy as np
import os
import cv2
import random
import tensorflow as tf
import pandas as pd
import skimage
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from utils import (
    add_noise_to_image
)
#import faiss

# Ensure reproducibility
np.random.seed(1337)
random.seed(1337)

class SignatureDataGenerator:
    def __init__(self, dataset, img_height=155, img_width=220, batch_sz=8):
        self.dataset = dataset
        self.dataset_name = list(dataset.keys())[0]
        self.img_height = img_height
        self.img_width = img_width
        self.batch_sz = batch_sz
        self.train_writers = []
        self.test_writers = []
        self._load_writers()
        

    def _load_writers(self):
        """Load writer directories and validate existence."""
        for dataset_name, dataset_info in self.dataset.items():
            dataset_path = dataset_info["path"]
            train_writers = dataset_info["train_writers"]
            test_writers = dataset_info["test_writers"]

            for writer in train_writers + test_writers:
                if isinstance(writer, dict):
                    writer_path = os.path.join(writer["path"], f"writer_{writer['writer']:03d}")
                else:
                    writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")

                if os.path.exists(writer_path):
                    if writer in train_writers:
                        self.train_writers.append((dataset_path, writer))
                    else:
                        self.test_writers.append((dataset_path, writer))
                else:
                    print(f"⚠ Warning: Writer directory not found: {writer_path}")

    def preprocess_image(self, img_path):
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"⚠ Warning: Missing image file: {img_path if isinstance(img_path, str) else 'Invalid Path Type'}")
            return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠ Warning: Unable to read image {img_path}")
                return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

            img = cv2.resize(img, (self.img_width, self.img_height))

            # Flatten for MinMax scaling
            img_flat = img.flatten().reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            img_scaled = scaler.fit_transform(img_flat).reshape(self.img_height, self.img_width)

            # Expand dims for channel (grayscale 1-channel image)
            img_scaled = np.expand_dims(img_scaled, axis=-1)

            return img_scaled.astype(np.float32)

        except Exception as e:
            print(f"⚠ Error processing image {img_path}: {e}")
            return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)
            
    def get_all_data_with_labels(self):
        """
        Collect all images and their labels (0 = genuine, 1 = forged).
        """
        images = []
        labels = []
        for dataset_path, writer in self.train_writers:
            genuine_path = os.path.join(dataset_path, f"writer_{writer:03d}", "genuine")
            forged_path = os.path.join(dataset_path, f"writer_{writer:03d}", "forged")

            # Collect genuine images (label 0)
            if os.path.exists(genuine_path):
                for img_file in os.listdir(genuine_path):
                    img_path = os.path.join(genuine_path, img_file)
                    img = self.preprocess_image(img_path)
                    images.append(img)
                    labels.append(0)
            else:
                print(f"⚠ Warning: Missing genuine folder for writer {writer}")

            # Collect forged images (label 1)
            if os.path.exists(forged_path):
                for img_file in os.listdir(forged_path):
                    img_path = os.path.join(forged_path, img_file)
                    img = self.preprocess_image(img_path)
                    images.append(img)
                    labels.append(1)
            else:
                print(f"⚠ Warning: Missing forged folder for writer {writer}")

        return np.array(images), np.array(labels)

    def get_all_data_with_writer_ids(self):
        """
        Return CLAHE‑preprocessed images + WRITER‑ID labels
        (label = writer id, not 0/1).
        """
        images, writer_ids = [], []
        for dataset_path, writer in self.train_writers:
            for label_type in ["genuine", "forged"]:
                img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
                if not os.path.exists(img_dir):
                    continue
                for fn in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, fn)
                    images.append(self.preprocess_image(img_path))
                    writer_ids.append(writer)
        return np.array(images), np.array(writer_ids)

    def get_train_data(self):
        """Generate triplet training data using TensorFlow Dataset."""
        return self.get_triplet_data(self.train_writers).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    def get_test_data(self):
        """Fetch test data WITHOUT generating triplets (to keep it untouched)."""
        test_images = []
        test_labels = []

        for dataset_path, writer in self.test_writers:
            writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
            genuine_path = os.path.join(writer_path, "genuine")
            forged_path = os.path.join(writer_path, "forged")

            if os.path.exists(genuine_path):
                for img_file in os.listdir(genuine_path):
                    img_path = os.path.join(genuine_path, img_file)
                    test_images.append(self.preprocess_image(img_path))
                    test_labels.append(0)

            if os.path.exists(forged_path):
                for img_file in os.listdir(forged_path):
                    img_path = os.path.join(forged_path, img_file)
                    test_images.append(self.preprocess_image(img_path))
                    test_labels.append(1)

        return tf.data.Dataset.from_tensor_slices((np.array(test_images), np.array(test_labels))).batch(self.batch_sz)

    def get_unbatched_data(self, noisy=False):
        dataset = self.get_noisy_test_data() if noisy else self.get_test_data()
        images, labels = [], []
        for img, label in dataset.unbatch():
            images.append(img.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)


    def generate_pairs(self):
        """
        Generate positive and negative pairs for contrastive loss training.
        Returns:
            pairs: list of (img1, img2)
            labels: list of 1 (genuine pair) or 0 (forged/different writer pair)
        """
        import random

        all_images, all_labels = self.get_all_data_with_labels()
        label_to_images = {}

        for img, label in zip(all_images, all_labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img)

        pairs = []
        labels = []

        # Create positive pairs (same label)
        for label in label_to_images:
            images = label_to_images[label]
            if len(images) > 1:
                for i in range(len(images) - 1):
                    pairs.append((images[i], images[i + 1]))
                    labels.append(1)

        # Create negative pairs (different labels)
        all_labels_set = list(label_to_images.keys())
        for _ in range(len(pairs)):  # match the number of positive pairs
            label1, label2 = random.sample(all_labels_set, 2)
            img1 = random.choice(label_to_images[label1])
            img2 = random.choice(label_to_images[label2])
            pairs.append((img1, img2))
            labels.append(0)

        return pairs, labels
