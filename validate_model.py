import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
MODEL_PATH = "fruit_fresh_rotten.h5"
LABELS_PATH = "labels.txt"
VAL_DIR = "data/validation"   # Folder to test

# -------------------------------------------------------
# LOAD MODEL + LABELS
# -------------------------------------------------------
print("🔹 Loading model and labels...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
print("✅ Loaded labels:", labels)

# -------------------------------------------------------
# VALIDATE MODEL
# -------------------------------------------------------
correct = 0
total = 0

for class_name in os.listdir(VAL_DIR):
    class_path = os.path.join(VAL_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array, verbose=0)
        pred_label = labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        total += 1
        if pred_label == class_name:
            correct += 1

        print(f"🖼️ {img_name} → Predicted: {pred_label} ({confidence:.2f}%) | Actual: {class_name}")

# -------------------------------------------------------
# ACCURACY REPORT
# -------------------------------------------------------
accuracy = (correct / total) * 100 if total > 0 else 0
print("\n---------------------------------------------")
print(f"✅ Validation Accuracy: {accuracy:.2f}%")
print("---------------------------------------------")
