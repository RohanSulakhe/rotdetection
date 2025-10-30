import cv2
import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
import time
import json
from tensorflow.keras.applications.inception_v3 import preprocess_input

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
MODEL_PATH = "apple_fresh_rotten.h5"
BROKER = "demo.thingsboard.io"  # or "thingsboard.cloud" if using cloud
PORT = 1883
ACCESS_TOKEN = "rruk5xzxn21znzr5dspv"   # 🔹 Replace with your device token

labels = ["Fresh", "Rotten"]

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
print("🔹 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------------------------------
# SETUP MQTT CONNECTION
# -------------------------------------------------------
client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)

try:
    client.connect(BROKER, PORT, 60)
    print("✅ Connected to ThingsBoard MQTT broker.")
except Exception as e:
    print("❌ MQTT Connection failed:", e)
    exit()

client.loop_start()

# -------------------------------------------------------
# CAMERA INITIALIZATION
# -------------------------------------------------------
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("❌ Error: Cannot access camera.")
    client.loop_stop()
    client.disconnect()
    exit()

print("🍎 Starting live fruit detection... Press 'q' to quit.")

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
def detect_apple(frame):
    """Simple detection: checks if enough color/edges are present."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 80)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density > 0.015  # Adjust threshold if needed

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        apple_present = detect_apple(frame)

        if not apple_present:
            display_text = "No Apple Detected"
            color = (255, 255, 0)
            payload = {"prediction": "No Apple", "confidence": 0.0}
        else:
            # --- Preprocess image ---
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # --- Predict ---
            prediction = model.predict(img, verbose=0)[0]
            pred_label = labels[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

            display_text = f"{pred_label} ({confidence:.1f}%)"
            color = (0, 255, 0) if pred_label == "Fresh" else (0, 0, 255)
            payload = {"prediction": pred_label, "confidence": round(confidence, 2)}

        # --- Display result ---
        cv2.putText(frame, display_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("🍎 Fruit Detection", frame)

        # --- Send telemetry to ThingsBoard ---
        try:
            client.publish("v1/devices/me/telemetry", json.dumps(payload))
            print(f"📤 Sent to ThingsBoard → {payload}")
        except Exception as e:
            print("⚠️ Failed to send telemetry:", e)

        # --- Wait or quit ---
        for i in range(7):
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                raise KeyboardInterrupt
            print(f"⏳ Next capture in {7 - i - 1}s...", end="\r")

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

finally:
    camera.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()
    print("✅ Camera and MQTT disconnected cleanly.")
