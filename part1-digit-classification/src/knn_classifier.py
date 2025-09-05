# src/knn_classifier.py

"""
KNN Classifier for Digit Recognition
Part of Computer Vision Lab II – MSc AI @ AAU
Author: Tsion Bizuayehu
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 📦 Load and prepare digit dataset from sklearn
def prepare_dataset():
    digits = load_digits()
    X = digits.data  # Each image is 8x8 pixels, flattened to 64 features
    y = digits.target  # Labels: digits 0–9
    return X, y

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
    print("📁 Loading digit dataset from sklearn...")
    X, y = prepare_dataset()
    print("🧠 Training KNN classifier...")
    train_knn(X, y)
