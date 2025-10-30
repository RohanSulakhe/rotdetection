import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("apple_fresh_rotten.h5")

# Load your test image (put your image path here)
img_path = "test_apple.jpg"  # 👈 change this to your test image name
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (224, 224))
img_array = np.expand_dims(img_resized / 255.0, axis=0)

# Predict
pred = model.predict(img_array)
class_names = ["freshapple", "rottenapple"]  # 👈 same as your training folder names
predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred) * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
