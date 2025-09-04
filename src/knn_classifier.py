# src/knn_classifier.py

"""
KNN Classifier for Digit Recognition
Part of Computer Vision Lab II – MSc AI @ AAU
Author: Tsion Bizuayehu
"""

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 🧼 Step 1: Load and preprocess image data
def load_digit_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at {path}")
    resized = cv2.resize(img, (64, 64))  # Resize for consistency
    flattened = resized.flatten()
    return flattened

# 📦 Step 2: Prepare dataset (placeholder for multiple samples)
def prepare_dataset():
    # TODO: Replace with actual dataset loading
    X = []
    y = []
    for i in range(10):  # Simulate 10 samples
        sample = np.random.randint(0, 256, (64 * 64))  # Dummy data
        X.append(sample)
        y.append(i % 2)  # Binary labels for demo
    return np.array(X), np.array(y)

# 🧪 Step 3: Train and evaluate KNN
def train_knn(X, y, k=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("🔍 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 🚀 Entry point
if __name__ == "__main__":
    print("📁 Loading dataset...")
    X, y = prepare_dataset()
    print("🧠 Training KNN classifier...")
    train_knn(X, y)
