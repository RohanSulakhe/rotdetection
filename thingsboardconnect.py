import paho.mqtt.client as mqtt
import json
import time

THINGSBOARD_HOST = "thingsboard.cloud"   # or your local IP
ACCESS_TOKEN = "omw5rm2jozpp45y9a98t"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)

client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.on_connect = on_connect
client.connect(THINGSBOARD_HOST, 1883, 60)
client.loop_start()

for i in range(3):
    payload = {"test": i}
    print("Sending:", payload)
    result = client.publish("v1/devices/me/telemetry", json.dumps(payload), 1)
    print("Publish result:", result.rc)
    time.sleep(2)

client.loop_stop()
client.disconnect()
