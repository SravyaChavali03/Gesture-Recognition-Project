import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ========== 1️⃣ Load Dataset ==========
dataset_path = "asl_alphabet_train"  # Path to the ASL Alphabet dataset
categories = sorted(os.listdir(dataset_path))  # List of categories (A-Z, space, delete, nothing)

data = []
labels = []

# Debug: Print dataset path and categories
print(f"Dataset path: {dataset_path}")
print(f"Categories: {categories}")

# Load images from the dataset
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    if not os.path.isdir(folder_path):
        print(f"Skipping {folder_path}, not a directory.")
        continue  # Skip if it's not a folder

    print(f"Loading images from {folder_path}...")

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_name}, unable to load.")
            continue

        img = cv2.resize(img, (64, 64))  # Resize images to 64x64
        data.append(img)
        labels.append(categories.index(category))  # Assign numerical label

# Convert lists to NumPy arrays
data = np.array(data) / 255.0  # Normalize pixel values
labels = np.array(labels)

if len(data) == 0:
    print("Error: No images found! Check dataset path.")
    exit()

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Dataset loaded successfully! {len(data)} images found.")
print(f"Training set: {len(X_train)} images, Test set: {len(X_test)} images")

# ========== 2️⃣ Define CNN Model ==========
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Add dropout to prevent overfitting
    layers.Dense(len(categories), activation='softmax')  # Output layer with softmax activation
])

# ========== 3️⃣ Compile and Train the Model ==========
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# ========== 4️⃣ Save the Model ==========
model.save("sign_language_model.h5")
print("Model training complete! 🎉")