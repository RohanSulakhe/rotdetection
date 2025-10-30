import cv2
import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
import time
import json

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
MODEL_PATH = "apple_fresh_rotten.h5"
BROKER = "demo.thingsboard.io"
PORT = 1883
ACCESS_TOKEN = "rruk5xzxn21znzr5dspv"   # 🔹 Replace with your actual token

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
print("🔹 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
labels = ["Fresh", "Rotten"]

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

print("🍎 Starting 7-sec interval fruit rot detection... Press 'q' to quit.")

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        # --- Preprocess image ---
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img / 255.0, axis=0)

        # --- Predict ---
        prediction = model.predict(img, verbose=0)[0]
        pred_label = labels[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        # --- Display result ---
        display_frame = frame.copy()
        color = (0, 255, 0) if pred_label == "Fresh" else (0, 0, 255)
        cv2.putText(display_frame, f"{pred_label} ({confidence:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Fruit Rot Detection (Every 7s)", display_frame)

        # --- Prepare and send telemetry ---
        payload = {
            "prediction": pred_label,
            "confidence": round(confidence, 2)
        }

        try:
            client.publish("v1/devices/me/telemetry", json.dumps(payload))
            print(f"📤 Sent to ThingsBoard → {payload}")
        except Exception as e:
            print("⚠️ Failed to send telemetry:", e)

        # --- Wait for 7 seconds ---
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
