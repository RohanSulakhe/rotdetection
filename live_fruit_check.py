#!/usr/bin/env python3
# live_fruit_check.py
import cv2
import numpy as np
import sys
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_PATH = "fruit_fresh_rotten.h5"
LABELS_PATH = "labels.txt"
CAM_INDEX = 0   # 0 = default webcam

# -------------------------------------------------------
# LOAD MODEL + LABELS
# -------------------------------------------------------
print("🔹 Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

print("🔹 Loaded labels:", labels)
target_size = (224, 224)

# -------------------------------------------------------
# START CAMERA
# -------------------------------------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    sys.exit(1)

print("🎥 Live Fruit Detection started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from camera.")
        break

    # Convert and preprocess frame
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict
    preds = model.predict(img, verbose=0)[0]
    pred_idx = np.argmax(preds)
    pred_label = labels[pred_idx]
    confidence = preds[pred_idx] * 100

    # Display results
    text = f"{pred_label} ({confidence:.2f}%)"
    color = (0, 255, 0) if "fresh" in pred_label.lower() else (0, 0, 255)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Show the frame
    cv2.imshow("🍎 Fruit Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped cleanly.")
