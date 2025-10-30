import tensorflow as tf
import numpy as np
import cv2
import os

# Load your trained model
model = tf.keras.models.load_model("apple_fresh_rotten.h5")

# Path to the image you want to test
image_path = "testimage.png" 

# Load and preprocess image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

img_resized = cv2.resize(img, (224, 224))
img_array = np.expand_dims(img_resized, axis=0) / 255.0  # normalize

# Predict
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

# Labels (adjust if reversed)
labels = ["freshapple", "rottenapple"]

# Show result
print(f"Prediction: {labels[class_index]}")
print(f"Confidence: {np.max(prediction) * 100:.2f}%")

# Optional — show image window with label
cv2.putText(img, f"{labels[class_index]} ({np.max(prediction)*100:.1f}%)",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
