# src/knn_classifier.py

"""
KNN Classifier for Digit Recognition
Part of Computer Vision Lab II – MSc AI @ AAU
Author: Tsion Bizuayehu
"""

import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 🧼 Load and preprocess a single digit image
def load_digit_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at {path}")
    resized = cv2.resize(img, (64, 64))
    flattened = resized.flatten()
    return flattened

# 📦 Prepare dataset using placeholder logic
def prepare_dataset():
    X = []
    y = []
    for i in range(10):  # Simulate 10 samples
        try:
            sample = load_digit_image("data/digit_sample.png")
        except ValueError:
            # If image is missing, use synthetic data instead
            sample = np.random.randint(0, 256, size=(64 * 64)).astype("uint8")
        X.append(sample)
        y.append(i % 2)  # Binary labels for demo
    return np.array(X), np.array(y)

# 🧪 Train and evaluate the KNN classifier
def train_knn(X, y, k=3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
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
