import tensorflow as tf
import numpy as np
import cv2
import json
import os

# -------------------------------------------------------
# LOAD MODEL AND LABELS
# -------------------------------------------------------
MODEL_PATH = "fruit_fresh_rotten.h5"
LABELS_PATH = "labels.txt"
IMAGE_PATH = "testimage2.png"

print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load labels (auto-detect JSON or plain text)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    text = f.read().strip()
    try:
        labels = json.loads(text)  # JSON format
        if isinstance(labels, str):  # handle stringified JSON list
            labels = json.loads(labels)
    except json.JSONDecodeError:
        labels = [line.strip() for line in text.splitlines() if line.strip()]

print("✅ Loaded labels:", labels)

# -------------------------------------------------------
# LOAD AND PREPROCESS IMAGE
# -------------------------------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")

img_resized = cv2.resize(img, (224, 224))
img_array = np.expand_dims(img_resized / 255.0, axis=0)

# -------------------------------------------------------
# PREDICT
# -------------------------------------------------------
prediction = model.predict(img_array, verbose=0)[0]
class_index = int(np.argmax(prediction))
confidence = float(np.max(prediction)) * 100
pred_label = labels[class_index] if class_index < len(labels) else f"class_{class_index}"

print("\n🧠 PREDICTION RESULT")
print(f"Prediction: {pred_label}")
print(f"Confidence: {confidence:.2f}%")

# -------------------------------------------------------
# DISPLAY IMAGE WITH RESULT
# -------------------------------------------------------
color = (0, 255, 0) if "fresh" in pred_label.lower() else (0, 0, 255)
cv2.putText(img, f"{pred_label} ({confidence:.1f}%)", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
cv2.imshow("Prediction Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
