import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm

print("🚀 Strong FFT-Only Feature Extraction Pipeline Started...")


# =====================================
# 1. FFT Feature Extraction (Stronger)
# =====================================
def extract_fft_features(image):
    """
    Strong FFT-only feature extractor.

    Key Improvements:
    - Hann window → reduces edge artifacts in FFT
    - 5 frequency bands → better frequency separation
    - Radial mean + std → captures distribution consistency
    - High-frequency ratio → strong fake detector
    - Phase + gradients → GAN inconsistency detection
    - Angular sectors → directional artifact detection
    """

    # Convert image to grayscale if it's RGB
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to fixed size (important for consistent FFT features)
    image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0

    # Get center coordinates
    h, w = image.shape
    ch, cw = h // 2, w // 2

    # ─────────────────────────────────────
    # Apply Hann Window
    # Reduces spectral leakage (sharp edges → noise in FFT)
    # ─────────────────────────────────────
    hann = np.outer(np.hanning(h), np.hanning(w))
    image = image * hann

    # Compute FFT and shift zero-frequency to center
    fshift = np.fft.fftshift(np.fft.fft2(image))

    # Magnitude → strength of frequencies
    magnitude = np.log1p(np.abs(fshift))

    # Phase → structural information
    phase = np.angle(fshift)

    features = []

    # Create radial distance map from center
    y_grid, x_grid = np.ogrid[:h, :w]
    r = np.hypot(x_grid - cw, y_grid - ch)

    # ─────────────────────────────────────────
    # 1. Frequency Bands (5 bands)
    # Captures energy distribution across frequencies
    # ─────────────────────────────────────────
    bands = [
        (0,  8),    # very low (DC component)
        (8,  16),   # low
        (16, 30),   # mid-low
        (30, 45),   # mid-high
        (45, 64),   # high frequency
    ]

    # Total energy for normalization
    total_energy = np.sum(magnitude ** 2) + 1e-8

    for r1, r2 in bands:
        mask = (r >= r1) & (r < r2)
        region = magnitude[mask]

        # Extract multiple statistics per band
        features.extend([
            np.mean(region),                         # average strength
            np.std(region),                          # variation
            np.sum(region ** 2),                     # energy
            np.percentile(region, 75) - np.percentile(region, 25),  # IQR
        ])

    # ─────────────────────────────────────────
    # 2. High-Frequency Energy Ratio
    # Fake images usually suppress high frequencies
    # ─────────────────────────────────────────
    hf_mask = r >= 30
    lf_mask = r < 30

    hf_energy = np.sum(magnitude[hf_mask] ** 2)
    lf_energy = np.sum(magnitude[lf_mask] ** 2)

    # Ratio features (very important for classification)
    features.append(hf_energy / (lf_energy + 1e-8))
    features.append(hf_energy / (total_energy + 1e-8))

    # ─────────────────────────────────────────
    # 3. Global Magnitude Features
    # ─────────────────────────────────────────
    features.extend([
        magnitude.mean(),
        magnitude.std(),
        magnitude.max(),
        np.percentile(magnitude, 95),
        np.percentile(magnitude, 5),
    ])

    # ─────────────────────────────────────────
    # 4. Phase Features
    # GANs often have smoother / unnatural phase
    # ─────────────────────────────────────────
    features.extend([
        np.std(phase),
        np.mean(np.abs(phase)),
    ])

    # Phase gradient → measures spatial irregularity
    dphase_y = np.diff(phase, axis=0)
    dphase_x = np.diff(phase, axis=1)

    features.extend([
        np.mean(np.abs(dphase_y)),
        np.std(dphase_y),
        np.mean(np.abs(dphase_x)),
        np.std(dphase_x),
    ])

    # ─────────────────────────────────────────
    # 5. Radial Profile (40 bins)
    # Captures how frequency changes with radius
    # ─────────────────────────────────────────
    r_int = r.astype(np.int32)
    max_bins = 40

    # Compute mean and variance efficiently using bincount
    radial_sum   = np.bincount(r_int.ravel(), magnitude.ravel(), minlength=max_bins)
    radial_sum2  = np.bincount(r_int.ravel(), (magnitude ** 2).ravel(), minlength=max_bins)
    radial_count = np.bincount(r_int.ravel(), minlength=max_bins)

    count = radial_count[:max_bins] + 1e-8
    radial_mean = radial_sum[:max_bins] / count

    # Variance formula: E[x^2] - (E[x])^2
    radial_var = radial_sum2[:max_bins] / count - radial_mean ** 2
    radial_std = np.sqrt(np.maximum(radial_var, 0))

    # Add 80 features (40 mean + 40 std)
    features.extend(radial_mean)
    features.extend(radial_std)

    # ─────────────────────────────────────────
    # 6. Angular Features (16 sectors)
    # Detects directional artifacts in GAN images
    # ─────────────────────────────────────────
    angle = np.arctan2((y_grid - ch), (x_grid - cw))
    sectors = np.linspace(-np.pi, np.pi, 17)

    for i in range(16):
        mask = (angle >= sectors[i]) & (angle < sectors[i + 1])
        region = magnitude[mask]

        features.append(np.mean(region))
        features.append(np.std(region))

    # Return final feature vector
    return np.array(features, dtype=np.float32)


# =====================================
# 2. Dataset Processing
# =====================================
def process_dataset(dataset_path):
    """
    Reads dataset folder:
        dataset/
            real/
            fake/

    Returns:
        X → features
        y → labels (0 = real, 1 = fake)
    """
    X, y = [], []

    # Loop over real and fake folders
    for label, value in [("real", 0), ("fake", 1)]:
        folder = os.path.join(dataset_path, label)

        print(f"\n📂 Processing: {folder}")

        if not os.path.isdir(folder):
            print("❌ Folder not found!")
            continue

        files = os.listdir(folder)
        print(f"📊 Total files: {len(files)}")

        processed, skipped = 0, 0

        # Iterate through images
        for file in tqdm(files, desc=label, ncols=100):

            # Skip non-image files
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, file)
            img = cv2.imread(path)

            # Skip unreadable images
            if img is None:
                skipped += 1
                continue

            try:
                # Extract FFT features
                feat = extract_fft_features(img)

                # Store feature + label
                X.append(feat)
                y.append(value)
                processed += 1

            except Exception:
                skipped += 1

        print(f"✅ Processed: {processed}, ❌ Skipped: {skipped}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# =====================================
# 3. Save Features
# =====================================
def save_features(dataset_path, output_file):
    """
    Runs feature extraction and saves as .pkl file
    """
    print(f"\n🚀 Processing dataset: {dataset_path}")

    X, y = process_dataset(dataset_path)

    # Check if dataset is empty
    if X.size == 0:
        print("⚠️ No data found!")
        return

    # Save features + labels together
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    print(f"💾 Saved: {output_file}")
    print(f"📐 Shape: {X.shape}")
    print(f"🧮 Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")


# =====================================
# 4. Main Execution
# =====================================
if __name__ == "__main__":

    # Base dataset path
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../real-vs-fake")
    )

    # Create output folder
    os.makedirs("features", exist_ok=True)

    print("\n🔥 Starting strong FFT-only pipeline...\n")

    # Process train, validation, and test sets
    for split in ["train", "valid", "test"]:
        save_features(
            os.path.join(base_path, split),
            f"features/{split}_fft_features.pkl"
        )

    print("\n🎉 All datasets processed successfully!")