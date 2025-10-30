# print_model_info.py
from tensorflow.keras.models import load_model
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "apple_fresh_rotten.h5"

print("Loading model:", model_path)
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    print("ERROR loading model:", e)
    print("If there are custom layers, you'll need their definitions and pass custom_objects.")
    raise

print("\n=== MODEL SUMMARY ===")
model.summary()

print("\nInput shape:", model.input_shape)
print("Output shape:", model.output_shape)

# If final activation looks like softmax and output dim > 1 -> classification
out_shape = model.output_shape
if isinstance(out_shape, tuple) and len(out_shape) == 2:
    num_classes = out_shape[1]
    print("Likely number of classes:", num_classes)
else:
    print("Model output is not a simple vector. It may be detection or segmentation.")
