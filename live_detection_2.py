import cv2
import numpy as np
import tensorflow as tf

# --- Load model ---
model = tf.keras.models.load_model("apple_fresh_rotten.h5")
labels = ["Fresh", "Rotten"]

# --- Start camera ---
cap = cv2.VideoCapture(0)
print("🎥 Starting apple detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 🍏 Define red color range (tune if lighting differs)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Create masks for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Reduce noise
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    apple_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # ignore small blobs
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 20:
                apple_detected = True
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

                # Crop the apple region for classification
                x1, y1 = max(int(x - radius), 0), max(int(y - radius), 0)
                x2, y2 = min(int(x + radius), frame.shape[1]), min(int(y + radius), frame.shape[0])
                crop = frame[y1:y2, x1:x2]

                # Predict freshness
                img = cv2.resize(crop, (224, 224))
                img = np.expand_dims(img / 255.0, axis=0)
                pred = model.predict(img, verbose=0)[0]
                label = labels[np.argmax(pred)]
                conf = np.max(pred) * 100

                cv2.putText(frame, f"{label} ({conf:.1f}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if label == "Fresh" else (0, 0, 255), 2)
                break  # only one apple

    if not apple_detected:
        cv2.putText(frame, "No Apple Detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("🍎 Apple Detector + Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Stopped camera.")
