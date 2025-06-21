import os
import csv
import numpy as np
import tensorflow as tf
import random
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def save_softmax_predictions(generator, model, dataset_name="default", split="train", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset_name}_{split}_pair_predictions.csv")

    # Build debug model that outputs softmax
    base_network = model.get_layer("base_network")
    input_a, input_b = model.input[0], model.input[1]
    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)
    combined_vec = tf.keras.layers.Concatenate()([encoded_a, encoded_b])
    dense32 = tf.keras.layers.Dense(32, activation='relu')(combined_vec)
    softmax_out = tf.keras.layers.Dense(2, activation='softmax')(dense32)
    debug_model = tf.keras.Model(inputs=[input_a, input_b], outputs=[softmax_out])

    # === Pair generation with metadata ===
    def generate_pairs_with_meta(generator):
        label_to_images, label_to_names = {}, {}
        writer_set = generator.train_writers if split == 'train' else generator.test_writers
        print(f"📚 Using writers: {[w for _, w in writer_set]}")

        for (dataset_path, writer) in writer_set:
            for label_type in ["genuine", "forged"]:
                dir_path = os.path.join(dataset_path, f"writer_{writer:03d}", label_type)
                if not os.path.exists(dir_path):
                    continue
                for fn in os.listdir(dir_path):
                    img = generator.preprocess_image(os.path.join(dir_path, fn))
                    label_to_images.setdefault(writer, []).append(img)
                    label_to_names.setdefault(writer, []).append(fn)

        pairs, labels, meta = [], [], []

        # Positive (genuine) pairs
        for writer in label_to_images:
            imgs, fns = label_to_images[writer], label_to_names[writer]
            for i in range(len(imgs) - 1):
                pairs.append((imgs[i], imgs[i + 1]))
                labels.append(1)  # Genuine
                meta.append((fns[i], fns[i + 1], f"writer_{writer:03d}"))

        # Negative (forged) pairs
        writer_ids = list(label_to_images.keys())
        for _ in range(len(pairs)):
            w1, w2 = random.sample(writer_ids, 2)
            img1 = random.choice(label_to_images[w1])
            img2 = random.choice(label_to_images[w2])
            fn1 = random.choice(label_to_names[w1])
            fn2 = random.choice(label_to_names[w2])
            pairs.append((img1, img2))
            labels.append(0)  # Forged
            meta.append((fn1, fn2, f"{w1}_vs_{w2}"))

        return pairs, labels, meta

    # === Prediction and Logging ===
    pairs, labels, meta_info = generate_pairs_with_meta(generator)
    y_true, y_pred = [], []

    with open(csv_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "pair_index", "image_A", "image_B", "writer_id",
            "true_label", "predicted_class", "softmax_forged", "softmax_genuine"
        ])

        for i, ((imgA, imgB), label, meta) in enumerate(zip(pairs, labels, meta_info)):
            img_a = np.expand_dims(imgA, axis=0)
            img_b = np.expand_dims(imgB, axis=0)
            softmax_pred = debug_model.predict([img_a, img_b], verbose=0)[0]

            softmax_forged = float(softmax_pred[0])   # class 0
            softmax_genuine = float(softmax_pred[1])  # class 1
            predicted_class = int(np.argmax([softmax_forged, softmax_genuine]))

            y_true.append(label)
            y_pred.append(predicted_class)

            writer.writerow([
                f"pair_{i+1}", meta[0], meta[1], meta[2],
                label, predicted_class,
                round(softmax_forged, 6), round(softmax_genuine, 6)
            ])

    print(f"📄 Predictions saved to {csv_path}")
