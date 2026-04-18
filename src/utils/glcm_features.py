import numpy as np
import cv2
import os
import pickle
from skimage.feature import graycomatrix, graycoprops


# ================================
# 1. GLCM FEATURE EXTRACTION
# ================================
def extract_glcm_features(
    image,
    img_size=128,
    distances=[1],
    angles=[0],
    props=None
):
    """
    Extract GLCM features from an image.

    Parameters:
    ----------
    image : np.ndarray
        Input image

    img_size : int
        Resize dimension

    distances : list
        GLCM distances

    angles : list
        GLCM angles

    props : list
        Properties to extract

    Returns:
    -------
    np.ndarray
        Feature vector
    """

    if image is None:
        return None

    if props is None:
        props = ["contrast", "energy", "homogeneity", "correlation"]

    # Convert to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    image = cv2.resize(image, (img_size, img_size))

    # Normalize to uint8 (important for GLCM)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract features
    features = []
    for prop in props:
        val = graycoprops(glcm, prop)
        features.append(np.mean(val))  # handles multiple angles/distances

    return np.array(features, dtype=np.float32)


# ================================
# 2. DATASET PROCESSING
# ================================
def process_dataset(dataset_path, extractor):
    X, y = [], []

    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = os.path.join(dataset_path, label_name)

        if not os.path.isdir(folder):
            continue

        files = os.listdir(folder)

        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, file)
            img = cv2.imread(path)

            if img is None:
                continue

            feat = extractor(img)

            if feat is not None:
                X.append(feat)
                y.append(label_val)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ================================
# 3. SAVE FEATURES
# ================================
def save_features(dataset_path, output_file, extractor=extract_glcm_features):
    """
    Extract and save features.
    """

    X, y = process_dataset(dataset_path, extractor)

    if X.size == 0:
        print("⚠️ No data found")
        return None

    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    print(f"Saved: {output_file}, Shape: {X.shape}")
    return X.shape