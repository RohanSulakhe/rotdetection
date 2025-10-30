import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("apple_fresh_rotten.h5")

# Labels (adjust order if needed)
labels = ["Fresh", "Rotten"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = default laptop camera

print("🎥 Starting camera... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Preprocess frame for prediction
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array, verbose=0)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    label = f"{labels[class_index]} ({confidence*100:.1f}%)"

    # Display prediction on screen
    color = (0, 255, 0) if class_index == 0 else (0, 0, 255)
    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imshow("🍎 Live Fruit Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped")
