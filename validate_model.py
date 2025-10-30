import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Load your trained model
model = tf.keras.models.load_model("apple_fresh_rotten.h5")

# ✅ Setup validation data generator
val_dir = "data/validation"  # change this if your folder name is different

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ✅ Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")
