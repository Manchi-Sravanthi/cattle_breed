# =========================
# STEP 1: IMPORTS
# =========================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# =========================
# STEP 2: DATASET PATH (UPDATED)
# =========================
dataset_path = "cattle_dataset"   # ✅ changed only this line

# =========================
# STEP 3: DATA PREPROCESSING
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Classes:", train_data.class_indices)

# =========================
# STEP 4: LOAD PRETRAINED MODEL
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# =========================
# STEP 5: BUILD MODEL
# =========================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.summary()

# =========================
# STEP 6: COMPILE MODEL
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# STEP 7: TRAIN MODEL
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# =========================
# STEP 8: SAVE MODEL
# =========================
model.save("cattle_model.h5")
print("Model saved successfully!")

# =========================
# STEP 9: MULTIPLE IMAGE TESTING
# =========================
print("\n--- Testing Multiple Images ---")

test_folder = "test_images"

class_labels = list(train_data.class_indices.keys())

if os.path.exists(test_folder):
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)

        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)

            print(f"\nImage: {img_name}")
            print("Predicted:", class_labels[class_index])
            print("Confidence:", float(np.max(prediction)))

        except:
            print(f"Error loading {img_name}")
else:
    print("test_images folder not found!")