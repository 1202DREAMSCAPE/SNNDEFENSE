

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

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)

        # Normalize to [0, 1]
        img_normalized = img_clahe.astype(np.float32) / 255.0

        # Expand dims for channel (grayscale 1-channel image)
        img_normalized = np.expand_dims(img_normalized, axis=-1)

        return img_normalized

    except Exception as e:
        print(f"⚠ Error processing image {img_path}: {e}")
        return np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)

        def generate_triplets_with_hard_negatives(self, model, all_images, all_labels):
    """
    Generate triplets (anchor, positive, hard negative) for triplet loss training.
    Hard negative is selected based on closest embedding distance to anchor among impostors.

    Args:
        model: Trained embedding model (outputting L2-normalized 128-D vectors)
        all_images: list of image arrays
        all_labels: list of writer labels

    Returns:
        triplets: list of (anchor, positive, hard negative)
    """
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances

    # Group images by label
    label_to_images = {}
    for img, label in zip(all_images, all_labels):
        label_to_images.setdefault(label, []).append(img)

    triplets = []

    for label in label_to_images:
        same_class_images = label_to_images[label]
        if len(same_class_images) < 2:
            continue  

        for i in range(len(same_class_images) - 1):
            anchor = same_class_images[i]
            positive = same_class_images[i + 1]

            anchor_embedding = model.predict(np.expand_dims(anchor, axis=0), verbose=0)
            hardest_negative = None
            min_distance = float('inf')

            for neg_label in label_to_images:
                if neg_label == label:
                    continue  

                for neg_candidate in label_to_images[neg_label]:
                    neg_embedding = model.predict(np.expand_dims(neg_candidate, axis=0), verbose=0)
                    distance = euclidean_distances(anchor_embedding, neg_embedding)[0][0]
                    if distance < min_distance:
                        min_distance = distance
                        hardest_negative = neg_candidate

            if hardest_negative is not None:
                triplets.append((anchor, positive, hardest_negative))

    return triplets

