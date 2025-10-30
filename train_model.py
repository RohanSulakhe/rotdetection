import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import layers, models
import json
import os

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
train_dir = 'data/train'
val_dir = 'data/validation'
model_name = 'fruit_fresh_rotten.h5'
labels_file = 'labels.txt'

# -------------------------------------------------------
# BASE MODEL (Transfer Learning)
# -------------------------------------------------------
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Custom classifier
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(4, activation='softmax')(x)   # 🔹 Adjust dynamically later if needed

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------------------------------
# DATA GENERATORS
# -------------------------------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# 🔹 Auto-detect class count and relink model output
num_classes = len(train_gen.class_indices)
model.layers[-1] = layers.Dense(num_classes, activation='softmax')(model.layers[-2].output)
model = models.Model(inputs=base_model.input, outputs=model.layers[-1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"🔹 Detected {num_classes} classes:", list(train_gen.class_indices.keys()))

# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# -------------------------------------------------------
# SAVE MODEL + LABELS
# -------------------------------------------------------
model.save(model_name)
with open(labels_file, "w") as f:
    json.dump(list(train_gen.class_indices.keys()), f)

print(f"✅ Training complete! Model saved as {model_name}")
print(f"✅ Labels saved as {labels_file}")
