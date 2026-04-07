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
    Improvements over base version:
      - Hann windowing (reduces spectral leakage)
      - 5 finer frequency bands (was 3)
      - Radial mean + std profile (was mean only)
      - High-freq energy ratio (powerful single feature)
      - Phase gradient features (phase regularity in GANs)
      - 16 angular sectors (was 8)
    """

    # Convert to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0
    h, w = image.shape
    ch, cw = h // 2, w // 2

    # ─────────────────────────────────────
    # Hann Window (reduces spectral leakage)
    # ─────────────────────────────────────
    hann = np.outer(np.hanning(h), np.hanning(w))
    image = image * hann

    # FFT
    fshift = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(fshift))
    phase = np.angle(fshift)

    features = []

    # Radial distance map (reused across steps)
    y_grid, x_grid = np.ogrid[:h, :w]
    r = np.hypot(x_grid - cw, y_grid - ch)

    # ─────────────────────────────────────────
    # 1. Frequency Bands — 5 finer bands
    #    (was 3: 0-10, 10-30, 30-64)
    # ─────────────────────────────────────────
    bands = [
        (0,  8),    # DC / very low
        (8,  16),   # low
        (16, 30),   # mid-low
        (30, 45),   # mid-high
        (45, 64),   # high
    ]

    total_energy = np.sum(magnitude ** 2) + 1e-8

    for r1, r2 in bands:
        mask = (r >= r1) & (r < r2)
        region = magnitude[mask]

        features.extend([
            np.mean(region),
            np.std(region),
            np.sum(region ** 2),
            np.percentile(region, 75) - np.percentile(region, 25),  # IQR
        ])

    # ─────────────────────────────────────────
    # 2. High-Frequency Energy Ratio
    #    Real images have more natural HF energy;
    #    GAN images tend to suppress it.
    # ─────────────────────────────────────────
    hf_mask = r >= 30
    lf_mask = r < 30

    hf_energy = np.sum(magnitude[hf_mask] ** 2)
    lf_energy = np.sum(magnitude[lf_mask] ** 2)

    features.append(hf_energy / (lf_energy + 1e-8))     # HF ratio
    features.append(hf_energy / (total_energy + 1e-8))  # HF fraction

    # ─────────────────────────────────────────
    # 3. Global Features
    # ─────────────────────────────────────────
    features.extend([
        magnitude.mean(),
        magnitude.std(),
        magnitude.max(),
        np.percentile(magnitude, 95),
        np.percentile(magnitude, 5),
    ])

    # ─────────────────────────────────────────
    # 4. Phase Features + Phase Gradient
    #    GAN outputs have unnaturally regular phase
    # ─────────────────────────────────────────
    features.extend([
        np.std(phase),
        np.mean(np.abs(phase)),
    ])

    # Phase gradient (measures spatial regularity of phase)
    dphase_y = np.diff(phase, axis=0)
    dphase_x = np.diff(phase, axis=1)

    features.extend([
        np.mean(np.abs(dphase_y)),
        np.std(dphase_y),
        np.mean(np.abs(dphase_x)),
        np.std(dphase_x),
    ])

    # ─────────────────────────────────────────
    # 5. Radial Profile — mean + std (40 bins)
    #    std catches how "uniform" each ring is
    # ─────────────────────────────────────────
    r_int = r.astype(np.int32)
    max_bins = 40

    radial_sum   = np.bincount(r_int.ravel(), magnitude.ravel(),          minlength=max_bins)
    radial_sum2  = np.bincount(r_int.ravel(), (magnitude ** 2).ravel(),   minlength=max_bins)
    radial_count = np.bincount(r_int.ravel(),                             minlength=max_bins)

    count = radial_count[:max_bins] + 1e-8
    radial_mean = radial_sum[:max_bins]  / count
    radial_var  = radial_sum2[:max_bins] / count - radial_mean ** 2
    radial_std  = np.sqrt(np.maximum(radial_var, 0))

    features.extend(radial_mean)   # 40 features
    features.extend(radial_std)    # 40 features

    # ─────────────────────────────────────────
    # 6. Angular (Directional) Features — 16 sectors
    #    GAN artifacts often show directional asymmetry
    # ─────────────────────────────────────────
    angle = np.arctan2((y_grid - ch), (x_grid - cw))
    sectors = np.linspace(-np.pi, np.pi, 17)  # 16 sectors

    for i in range(16):
        mask = (angle >= sectors[i]) & (angle < sectors[i + 1])
        region = magnitude[mask]
        features.append(np.mean(region))
        features.append(np.std(region))

    return np.array(features, dtype=np.float32)


# =====================================
# 2. Dataset Processing
# =====================================
def process_dataset(dataset_path):
    X, y = [], []

    for label, value in [("real", 0), ("fake", 1)]:
        folder = os.path.join(dataset_path, label)

        print(f"\n📂 Processing: {folder}")

        if not os.path.isdir(folder):
            print("❌ Folder not found!")
            continue

        files = os.listdir(folder)
        print(f"📊 Total files: {len(files)}")

        processed, skipped = 0, 0

        for file in tqdm(files, desc=label, ncols=100):

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, file)
            img = cv2.imread(path)

            if img is None:
                skipped += 1
                continue

            try:
                feat = extract_fft_features(img)
                X.append(feat)
                y.append(value)
                processed += 1
            except Exception as e:
                skipped += 1

        print(f"✅ Processed: {processed}, ❌ Skipped: {skipped}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# =====================================
# 3. Save Features
# =====================================
def save_features(dataset_path, output_file):
    print(f"\n🚀 Processing dataset: {dataset_path}")

    X, y = process_dataset(dataset_path)

    if X.size == 0:
        print("⚠️ No data found!")
        return

    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)

    print(f"💾 Saved: {output_file}")
    print(f"📐 Shape: {X.shape}")
    print(f"🧮 Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")


# =====================================
# 4. Main Execution
# =====================================
if __name__ == "__main__":

    base_path = "real-vs-fake"
    os.makedirs("features", exist_ok=True)

    print("\n🔥 Starting strong FFT-only pipeline...\n")

    for split in ["train", "valid", "test"]:
        save_features(
            os.path.join(base_path, split),
            f"features/{split}_fft_features.pkl"
        )

    print("\n🎉 All datasets processed successfully!")