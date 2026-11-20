# import os
# import json
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# # --- 1. Define Paths and Hyperparameters ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# TRAIN_DIR = os.path.join(DATA_DIR, 'train')
# TEST_DIR = os.path.join(DATA_DIR, 'test')
# MODEL_DIR = os.path.join(BASE_DIR, 'models')
# MODEL_PATH = os.path.join(MODEL_DIR, 'ewaste_cnn.h5')

# # Create the models directory if it doesn't exist
# if not os.path.exists(MODEL_DIR):
#     os.makedirs(MODEL_DIR)

# # Hyperparameters
# IMG_SIZE = (128, 128)
# BATCH_SIZE = 32
# EPOCHS = 50  # Increased epochs for better training
# LR = 1e-4

# # --- 2. Data Generators with Augmentation ---
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     shear_range=0.2,
#     brightness_range=[0.8,1.2],
#     horizontal_flip=True,
#     vertical_flip=True
# )

# val_datagen = ImageDataGenerator(rescale=1./255)

# # Load data from directories
# print("Loading training data from:", TRAIN_DIR)
# train_gen = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# print("\nLoading validation data from:", TEST_DIR)
# val_gen = val_datagen.flow_from_directory(
#     TEST_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# num_classes = train_gen.num_classes
# print(f"\nFound {num_classes} classes.")

# # --- 3. Build the CNN Model ---
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(64, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(128, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(256, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # --- 4. Define Callbacks and Train the Model ---
# checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
# early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

# print("\nStarting model training...")
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=EPOCHS,
#     callbacks=[checkpoint, early]
# )

# # The best model is already saved by ModelCheckpoint, but let's save the final model as well.
# model.save(MODEL_PATH)

# # --- 5. Save Class Indices ---
# with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
#     json.dump(train_gen.class_indices, f)

# print("\nModel trained and saved to:", MODEL_PATH)
# if 'val_accuracy' in history.history:
#     best_val_acc = max(history.history['val_accuracy'])
#     print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_dir = 'D:\ewaste\data\test'
val_dir = 'D:\ewaste\data\val'
test_dir = 'D:\ewaste\data\test'

# Hyperparameters
num_classes = 10
num_epochs = 20
batch_size = 32
learning_rate = 0.001
image_size = 224  # For consistency with your CNN

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Data loaders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define a simple CNN
def build_cnn(input_shape=(image_size, image_size, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn(num_classes=num_classes)

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# Save the model
model.save('ewaste_cnn_tf.h5')
print("Model saved as ewaste_cnn_tf.h5")

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")
