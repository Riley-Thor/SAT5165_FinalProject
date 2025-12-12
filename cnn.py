import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# Disable GPU if on a VM without one to avoid cluttering logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("--- Loading Data ---")
# This automatically reads all 16 parquet files into one DataFrame
df = pd.read_parquet("processed_malware_data")

print(f"Total rows loaded: {len(df)}")
print("Class distribution before cleaning:")
print(df['label'].value_counts())

# Count occurrences of each label
class_counts = df['label'].value_counts()
# Find classes that have at least 2 samples
valid_classes = class_counts[class_counts >= 10].index
# Filter the dataframe
df_clean = df[df['label'].isin(valid_classes)].copy()

print(f"\nRows after removing single-instance classes: {len(df_clean)}")
dropped_classes = len(df) - len(df_clean)
if dropped_classes > 0:
    print(f"WARNING: Dropped {dropped_classes} rows because their class had only 1 sample.")
# -------------------------------------------------------

# Reshape features back to 64x64x1
X = np.array(df_clean['features'].tolist()).reshape(-1, 64, 64, 1)

# Normalize pixel values
X = X / 255.0

# One-hot encode labels
y_dummies = pd.get_dummies(df_clean['label'])
y = y_dummies.values
num_classes = y.shape[1]

print(f"Training on {num_classes} Malware Families.")

# 2. Split Data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# 3. Data Augmentation
# Only apply to training data to prevent data leakage
datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# 4. Model Definition
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    
    # Block 2
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Block 3
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5), # Regularization
    Dense(num_classes, activation='softmax') # Output layer dynamic to valid classes
])

# 5. Compile with Hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train
print("\n--- Starting Training ---")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10 # Reduced to 10 for quick testing
)

# 7. Evaluation
print("\n--- Evaluating ---")
results = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {results[1]}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

class_names = y_dummies.columns.tolist()

print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
