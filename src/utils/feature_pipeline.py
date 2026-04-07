import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm

print("🚀 FFT Feature Extraction Pipeline Started...")


# =====================================
# 1. FFT Feature Extraction (Improved)
# =====================================
def extract_fft_features(image):
    """
    Extract enhanced FFT-based features from image
    """

    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize for consistency
    image = cv2.resize(image, (128, 128))

    # FFT transformation
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    ch, cw = h // 2, w // 2

    # ============================
    # Frequency Band Separation
    # ============================

    # Low-frequency region (center)
    low_freq = magnitude[ch-10:ch+10, cw-10:cw+10]

    # High-frequency region (everything else)
    high_freq = magnitude.copy()
    high_freq[ch-10:ch+10, cw-10:cw+10] = 0

    features = []

    # ============================
    # Statistical Features
    # ============================
    for region in [low_freq, high_freq]:
        features.append(np.mean(region))
        features.append(np.std(region))
        features.append(np.sum(region ** 2))

    # Global features
    features.append(np.mean(magnitude))
    features.append(np.std(magnitude))

    # ============================
    # Radial Frequency Features
    # ============================
    y, x = np.indices((h, w))
    r = np.sqrt((x - cw) ** 2 + (y - ch) ** 2)
    r = r.astype(int)

    radial_mean = np.bincount(r.ravel(), magnitude.ravel()) / np.bincount(r.ravel())

    # Take first 20 radial bins
    features.extend(radial_mean[:20])

    return np.array(features)


# =====================================
# 2. Dataset Processing
# =====================================
def process_dataset(dataset_path):
    features = []
    labels = []

    for label in ["real", "fake"]:
        folder = os.path.join(dataset_path, label)

        print(f"\n📂 Processing folder: {folder}")

        if not os.path.exists(folder):
            print("❌ Folder not found!")
            continue

        files = os.listdir(folder)
        print(f"📊 Total files: {len(files)}")

        processed = 0
        skipped = 0

        for file in tqdm(files):

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                skipped += 1
                continue

            try:
                feat = extract_fft_features(img)
                features.append(feat)
                labels.append(0 if label == "real" else 1)
                processed += 1

            except Exception as e:
                skipped += 1

        print(f"✅ Processed: {processed}, Skipped: {skipped}")

    return np.array(features), np.array(labels)


# =====================================
# 3. Save Features
# =====================================
def save_features(dataset_path, output_file):
    print(f"\n🚀 Processing dataset: {dataset_path}")

    X, y = process_dataset(dataset_path)

    # Save features
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    print(f"💾 Saved: {output_file}")
    print(f"📐 Shape: {X.shape}")


# =====================================
# 4. Main Execution
# =====================================
if __name__ == "__main__":

    base_path = "real-vs-fake"

    # Create output directory
    os.makedirs("features", exist_ok=True)

    print("\n🔥 Starting full pipeline...\n")

    # Train
    save_features(
        os.path.join(base_path, "train"),
        "features/train_fft_features.pkl"
    )

    # Validation
    save_features(
        os.path.join(base_path, "valid"),
        "features/valid_fft_features.pkl"
    )

    # Test
    save_features(
        os.path.join(base_path, "test"),
        "features/test_fft_features.pkl"
    )

    print("\n🎉 All datasets processed successfully!")