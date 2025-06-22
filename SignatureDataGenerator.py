import numpy as np
import os
import cv2
import random
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
import csv

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
    
    def preprocess_image_clahe(self, img_path):
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"⚠ Warning: Missing image file: {img_path if isinstance(img_path, str) else 'Invalid Path Type'}")
            return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠ Warning: Unable to read image {img_path}")
                return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

            # Resize to expected dimensions
            img = cv2.resize(img, (self.img_width, self.img_height))

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img)

            # Normalize to [0, 1] after CLAHE (optional but often beneficial)
            img_clahe = img_clahe.astype(np.float32) / 255.0

            # Expand dims for grayscale channel
            img_clahe = np.expand_dims(img_clahe, axis=-1)

            return img_clahe

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

    def get_train_data_with_labels(self):
        """
        Return all training images and their writer IDs (not binary labels).
        Used for contrastive pair generation.
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

    def get_test_data(self):
        """Fetch test data"""
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
    
    def get_test_data_with_labels(self):
        """
        Return test images and their writer IDs (used for generating SOP 2 pairs).
        """
        images, writer_ids = [], []
        for dataset_path, writer in self.test_writers:
            for label_type in ["genuine", "forged"]:
                img_dir = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
                if not os.path.exists(img_dir):
                    continue
                for fn in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, fn)
                    images.append(self.preprocess_image(img_path))
                    writer_ids.append(writer)
        return np.array(images), np.array(writer_ids)

    def get_unbatched_data(self):
            dataset = self.get_test_data()
            images, labels = [], []
            for img, label in dataset.unbatch():
                images.append(img.numpy())
                labels.append(label.numpy())
            return np.array(images), np.array(labels)

    def preprocess_image_raw(self, img_path):
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"⚠ Missing image file: {img_path}")
            return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠ Could not read: {img_path}")
                return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype(np.float32) / 255.0  # Light normalization (0–1)
            img = np.expand_dims(img, axis=-1)
            return img
        except Exception as e:
            print(f"⚠ Error in raw preprocess: {e}")
            return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

    def generate_pairs(self, split='train', return_metadata=False, use_clahe=False, use_raw=False, log_csv_path=None):
        """
        Generate positive and negative pairs for training or evaluation.

        Args:
            split (str): 'train', 'test', or 'all'.
            return_metadata (bool): If True, returns filenames and writer IDs for logging.
            use_clahe (bool): If True, apply CLAHE preprocessing.
            use_raw (bool): If True, use raw preprocessing instead of normalized.

        Returns:
            pairs (list of tuple): Each tuple contains two preprocessed images (img1, img2).
            labels (list of int): 1 if same writer (genuine pair), else 0.
            meta (optional): List of tuples (filename1, filename2, writer_id or writer1_vs_writer2).
        """

        # === Load data and writer info based on split
        if split == 'train':
            all_images, all_labels = self.get_train_data_with_labels()
            writer_set = sorted(set(writer for _, writer in self.train_writers))
            writer_list = self.train_writers
        elif split == 'test':
            all_images, all_labels = self.get_test_data_with_labels()
            writer_set = sorted(set(writer for _, writer in self.test_writers))
            writer_list = self.test_writers
        elif split == 'all':
            all_images, all_labels = self.get_all_data_with_labels()
            writer_set = sorted(set(writer for _, writer in self.train_writers + self.test_writers))
            writer_list = self.train_writers + self.test_writers
        else:
            raise ValueError("Invalid split.")

        print(f"📋 Writers used in '{split}' split: {writer_set}")

        # === Group images and filenames by writer
        label_to_images = {}
        label_to_filenames = {}

        for (dataset_path, writer) in writer_list:
            for label_type in ["genuine", "forged"]:
                folder = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
                if not os.path.exists(folder):
                    continue
                for file in os.listdir(folder):
                    img_path = os.path.join(folder, file)
                    if use_raw:
                        img = self.preprocess_image_raw(img_path)
                    elif use_clahe:
                        img = self.preprocess_image_clahe(img_path)
                    else:
                        img = self.preprocess_image(img_path)
                    label_to_images.setdefault(writer, []).append(img)
                    label_to_filenames.setdefault(writer, []).append(file)

        pairs, labels = [], []
        meta = []

        # === Get positive pairs
        for writer in label_to_images:
            imgs = label_to_images[writer]
            fns = label_to_filenames[writer]
            for i in range(len(imgs) - 1):
                pairs.append((imgs[i], imgs[i + 1]))
                labels.append(1)
                if return_metadata:
                    meta.append((fns[i], fns[i + 1], f"writer_{writer:03d}"))

        # === Get negative pairs
        writer_ids = list(label_to_images.keys())
        for writer in label_to_images:
            pos_imgs = label_to_images[writer]
            pos_fns = label_to_filenames[writer]

            for i in range(len(pos_imgs) - 1):
                anchor_img = pos_imgs[i]
                anchor_fn = pos_fns[i]

                # Random negative pairing
                w2 = random.choice([w for w in writer_ids if w != writer])
                img2 = random.choice(label_to_images[w2])
                fn2 = random.choice(label_to_filenames[w2])
                pairs.append((anchor_img, img2))
                labels.append(0)
                if return_metadata:
                    meta.append((anchor_fn, fn2, f"{writer}_vs_{w2}"))

        combined = list(zip(pairs, labels, meta if return_metadata else [None]*len(pairs)))
        random.shuffle(combined)
        pairs, labels, meta_shuffled = zip(*combined)
        pairs = list(pairs)
        labels = list(labels)
        if return_metadata:
            meta = list(meta_shuffled)

        # Log to CSV 
        if log_csv_path is not None and return_metadata and meta:
            os.makedirs(os.path.dirname(log_csv_path), exist_ok=True) 
            write_header = not os.path.exists(log_csv_path)
            with open(log_csv_path, "a", newline="") as csvfile:
                writer_csv = csv.writer(csvfile)
                if write_header:
                    writer_csv.writerow(["img1", "img2", "meta"])
                writer_csv.writerows(meta)

        if return_metadata:
            return pairs, labels, meta
        else:
            return pairs, labels
        
    def generate_triplets(self, dataset_path, writer, use_clahe=False, use_raw=False, log_csv_path=None):
        writer_path = os.path.join(dataset_path, f"writer_{writer:03d}")
        genuine_path = os.path.join(writer_path, "genuine")
        forged_path = os.path.join(writer_path, "forged")

        if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
            return []

        genuine_imgs = [os.path.join(genuine_path, f) for f in os.listdir(genuine_path) if f.endswith((".png", ".jpg"))]
        forged_imgs  = [os.path.join(forged_path, f) for f in os.listdir(forged_path) if f.endswith((".png", ".jpg"))]
        if len(genuine_imgs) < 2:
            return []

        # Negative pool: add all forgeries of current writer (skilled forgeries)
        negative_candidates = []
        negative_candidates.extend(forged_imgs)  # Add forged signatures of this writer (forgeries by others)

        # Add all images (genuine/forged) of other writers as additional negatives
        for (other_dataset_path, other_writer) in self.train_writers:
            if other_writer == writer:
                continue
            for label_type in ["genuine", "forged"]:
                neg_dir = os.path.join(other_dataset_path, f"writer_{other_writer:03d}", label_type)
                if os.path.exists(neg_dir):
                    negative_candidates.extend(
                        [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith((".png", ".jpg"))]
                    )

        if not negative_candidates:
            return []

        triplets = []
        triplet_logs = []

        if use_raw:
            preprocess = self.preprocess_image_raw
        elif use_clahe:
            preprocess = self.preprocess_image_clahe
        else:
            preprocess = self.preprocess_image

        for i in range(len(genuine_imgs) - 1):
            anchor_path = genuine_imgs[i]
            positive_path = genuine_imgs[i + 1]
            negative_path = random.choice(negative_candidates)

            anchor_img   = preprocess(anchor_path)
            positive_img = preprocess(positive_path)
            negative_img = preprocess(negative_path)

            triplets.append((anchor_img, positive_img, negative_img))
            triplet_logs.append([anchor_path, positive_path, negative_path])

            # Write to CSV (as before)
            if log_csv_path is not None and triplet_logs:
                os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                write_header = not os.path.exists(log_csv_path)
                with open(log_csv_path, "a", newline="") as csvfile:
                    writer_csv = csv.writer(csvfile)
                    if write_header:
                        writer_csv.writerow(["anchor", "positive", "negative"])
                    writer_csv.writerows(triplet_logs)
        return triplets


    def get_triplet_data(self, writers_list,use_clahe=False, use_raw=False, log_csv_path=None):
        """Generate triplet data formatted for TensorFlow's Dataset API."""
        triplets = []

        for dataset_path, writer in writers_list:
            if (dataset_path, writer) not in self.train_writers:
                print(f"⚠ Skipping writer {writer} (not in train_writers)")
                continue  

            writer_triplets = self.generate_triplets(dataset_path, writer, 
                                                     use_clahe=use_clahe, 
                                                     use_raw=use_raw,
                                                     log_csv_path=log_csv_path)
            if writer_triplets:
                triplets.extend(writer_triplets)

        random.shuffle(triplets)
        
        def generator():
            for anchor, positive, negative in triplets:
                yield (anchor, positive, negative), 0.0  # dummy label

        output_signature = (
            (
                tf.TensorSpec(shape=(self.img_height, self.img_width, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.img_height, self.img_width, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.img_height, self.img_width, 1), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32)  # dummy label
        )

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature).batch(self.batch_sz)

    def get_triplet_train(self, use_clahe=False, log_csv_path=None):
        """Generate triplet training data using TensorFlow Dataset."""
        return (self.get_triplet_data(self.train_writers, use_clahe=use_clahe, log_csv_path=log_csv_path)
                    .repeat()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        