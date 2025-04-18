import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load labels
labels = pd.read_csv("labels.csv")

# Print column names and first few rows to inspect
print("Column names:", labels.columns.tolist())
print("First 5 rows:\n", labels.head())

class_counts = labels.iloc[:, 0].value_counts()

# B1a: Visualization for class distribution
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar')
plt.title('Distribution of Seedling Species')
plt.xlabel('Species')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png')  # Save for screenshot
plt.show()

# B1b: Sample images with associated labels
images = np.load("images.npy")  # Shape: (4750, 128, 128, 3)

# Get unique classes and one sample per class
unique_classes = labels.iloc[:, 0].unique()
plt.figure(figsize=(15, 5))

for i, species in enumerate(unique_classes):
    idx = labels[labels.iloc[:, 0] == species].index[0]  # First occurrence
    plt.subplot(2, 6, i + 1)  # 2 rows, 6 cols for 12 classes
    plt.imshow(images[idx])
    plt.title(species, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_images.png')  # Save for screenshot
plt.show()

# B2: Perform Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    fill_mode='nearest'
)

# B3: Normalize the Images
images_normalized = images.astype('float32') / 255.0
print(images_normalized.min(), images_normalized.max())  # Should print 0.0, 1.0

X_train, X_temp, y_train, y_temp = train_test_split(
    images_normalized, labels.iloc[:, 0], test_size=0.3, stratify=labels.iloc[:, 0], random_state=42
)

# B4: Perform Training (70%), Validation (15%), and  Test (15%) Split
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(X_train.shape, X_val.shape, X_test.shape)  # (3325, 128, 128, 3), (712, 128, 128, 3), (713, 128, 128, 3)

# B5: Encode the target feature for all datasets
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)  # y_train from split
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)

y_train_onehot = to_categorical(y_train_encoded, num_classes=12)
y_val_onehot = to_categorical(y_val_encoded, num_classes=12)
y_test_onehot = to_categorical(y_test_encoded, num_classes=12)
print(y_train_onehot.shape)  # (3325, 12)

# B6: Provide a copy of all datasets
np.save('task1_X_train.npy', X_train)
np.save('task1_X_val.npy', X_val)
np.save('task1_X_test.npy', X_test)
np.save('task1_y_train_onehot.npy', y_train_onehot)
np.save('task1_y_val_onehot.npy', y_val_onehot)
np.save('task1_y_test_onehot.npy', y_test_onehot)

# Before E1: Load the saved datasets
X_train = np.load('task1_X_train.npy')
X_val = np.load('task1_X_val.npy')
X_test = np.load('task1_X_test.npy')
y_train_onehot = np.load('task1_y_train_onehot.npy')
y_val_onehot = np.load('task1_y_val_onehot.npy')
y_test_onehot = np.load('task1_y_test_onehot.npy')

# E1: Output of Model Summary
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(12, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Early stopping for E3d and E4
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# E4: Confusion Matrix
# Train the model
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=50, batch_size=32, callbacks=[early_stopping]
)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
y_test_classes = np.argmax(y_test_onehot, axis=1)  # Convert one-hot to class indices

# Generate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task1_confusion_matrix.png')
plt.show()

# F1a: Final training epoch
print("Final training epoch:", len(history.history['loss']))
print("Training loss:", history.history['loss'][-1])
print("Validation loss:", history.history['val_loss'][-1])
print("Training accuracy:", history.history['accuracy'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])

# F1b: Compare training vs. validation accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# F1c: Training vs. validation loss plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('task1_train_val_loss.png')
plt.show()