import numpy as np
import cv2
import os
import pickle
from skimage.feature import local_binary_pattern


# ==============================
# 1. LBP FEATURE EXTRACTION
# ==============================
def extract_lbp_features(
    image,
    img_size=128,
    P=8,
    R=1,
    method="uniform"
):
    """
    Extract LBP features from an image.

    Parameters:
    ----------
    image : np.ndarray
        Input image

    img_size : int
        Resize dimension

    P : int
        Number of circularly symmetric neighbour points

    R : int
        Radius

    method : str
        LBP method (default = 'uniform')

    Returns:
    -------
    np.ndarray
        Feature vector (histogram)
    """

    if image is None:
        return None

    # Convert to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    image = cv2.resize(image, (img_size, img_size))

    # Compute LBP
    lbp_map = local_binary_pattern(image, P=P, R=R, method=method)

    # Number of bins (important for uniform LBP)
    n_bins = P + 2 if method == "uniform" else int(lbp_map.max() + 1)

    # Histogram
    hist, _ = np.histogram(
        lbp_map.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    # Normalize manually (safer than density=True)
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)

    return hist


# ==============================
# 2. DATASET PROCESSING
# ==============================
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


# ==============================
# 3. SAVE FEATURES
# ==============================
def save_features(dataset_path, output_file, extractor=extract_lbp_features):
    """
    Extract and save LBP features.
    """

    X, y = process_dataset(dataset_path, extractor)

    if X.size == 0:
        print("⚠️ No data found")
        return None

    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    print(f"Saved: {output_file}, Shape: {X.shape}")
    return X.shape