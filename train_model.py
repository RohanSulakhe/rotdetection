import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import layers, models

train_dir = 'data/train'
val_dir = 'data/validation'

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# Add classifier head
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(2, activation='softmax')(x)  # 2 classes: fresh, rotten

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save
model.save('apple_fresh_rotten.h5')
print("✅ Training complete! Saved as apple_fresh_rotten.h5")
