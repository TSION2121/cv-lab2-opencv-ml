
# src/utils/preprocessing.py

"""
Image Preprocessing Utilities
Part of Computer Vision Lab II – MSc AI @ AAU
Author: Tsion Bizuayehu
"""

import cv2
import numpy as np

# 🧽 Normalize pixel values to [0, 1]
def normalize_image(img: np.ndarray) -> np.ndarray:
    return img.astype("float32") / 255.0

# 📐 Resize image to target dimensions
def resize_image(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    return cv2.resize(img, size)

# 📦 Flatten image for ML input
def flatten_image(img: np.ndarray) -> np.ndarray:
    return img.flatten()

# 🧰 Full preprocessing pipeline
def preprocess_image(path: str, size=(64, 64)) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    resized = resize_image(img, size)
    normalized = normalize_image(resized)
    flattened = flatten_image(normalized)
    return flattened
