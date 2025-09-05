# src/decision_tree.py

"""
Decision Tree Classifier for Digit Recognition
Part of Computer Vision Lab II – MSc AI @ AAU
Author: Tsion Bizuayehu
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 📦 Prepare synthetic dataset
def prepare_dataset(num_samples=100, image_size=(64, 64)):
    X = []
    y = []
    for i in range(num_samples):
        sample = np.random.randint(0, 256, size=image_size).astype("uint8")
        flattened = sample.flatten()
        X.append(flattened)
        y.append(i % 2)  # Binary labels
    return np.array(X), np.array(y)

# 🧪 Train and evaluate Decision Tree
def train_decision_tree(X, y, max_depth=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("🔍 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 🚀 Entry point
if __name__ == "__main__":
    print("📁 Generating synthetic dataset...")
    X, y = prepare_dataset()
    print("🧠 Training Decision Tree classifier...")
    train_decision_tree(X, y)
