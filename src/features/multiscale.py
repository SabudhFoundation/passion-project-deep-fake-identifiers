import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# =========================
# GLOBAL CONFIG
# =========================
IMG_SIZE = 128
PATCH_GRID = 2

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi/2]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

LBP_RADII = [1, 2]


# =========================
# PATCH SPLIT
# =========================
def split_into_patches(image, grid_size=PATCH_GRID):
    h, w = image.shape
    ph, pw = h // grid_size, w // grid_size
    return [
        image[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
        for i in range(grid_size)
        for j in range(grid_size)
    ]


# =========================
# GLCM FEATURES (OPTIMIZED)
# =========================
def extract_glcm_features(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    patches = split_into_patches(image)

    features = []

    for patch in patches:
        glcm = graycomatrix(
            patch,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            symmetric=True,
            normed=True
        )

        for prop in GLCM_PROPS:
            vals = graycoprops(glcm, prop)
            features.append(np.mean(vals))
            features.append(np.std(vals))

    return np.array(features, dtype=np.float32)


# =========================
# LBP FEATURES (OPTIMIZED)
# =========================
def extract_lbp_features(image):
    patches = split_into_patches(image)
    features = []

    for patch in patches:
        for r in LBP_RADII:
            n_points = 8 * r

            lbp = local_binary_pattern(
                patch, n_points, r, method='uniform'
            )

            hist, _ = np.histogram(
                lbp.ravel(),
                bins=n_points + 2,
                range=(0, n_points + 2)
            )

            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-6)

            features.extend(hist)

    return np.array(features, dtype=np.float32)


# =========================
# FFT FEATURES (SIMPLIFIED)
# =========================
def extract_fft_features(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    h, w = image.shape
    ch, cw = h // 2, w // 2

    hann = np.outer(np.hanning(h), np.hanning(w))
    image = image * hann

    fshift = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(fshift))

    features = []

    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cw, y - ch)

    bands = [(0, 8), (8, 16), (16, 30), (30, 45), (45, 64)]

    for r1, r2 in bands:
        mask = (r >= r1) & (r < r2)
        region = magnitude[mask]

        if region.size == 0:
            features.extend([0, 0, 0])
            continue

        features.extend([
            np.mean(region),
            np.std(region),
            np.sum(region**2)
        ])

    return np.array(features, dtype=np.float32)


# =========================
# COMBINED FEATURES
# =========================
def extract_all_features(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    feat = np.concatenate([
        extract_glcm_features(image),
        extract_lbp_features(image),
        extract_fft_features(image)
    ])

    # Normalize (VERY IMPORTANT)
    feat = feat / (np.linalg.norm(feat) + 1e-8)

    return feat


# =========================
# MULTIPROCESS HELPER
# =========================
def process_image(args):
    path, label = args

    img = cv2.imread(path)
    if img is None:
        return None

    try:
        feat = extract_all_features(img)
        return feat, label
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
        return None


# =========================
# DATASET PROCESSING
# =========================
def process_dataset(dataset_path, output_file):
    print(f"\n📂 Processing: {dataset_path}")

    all_tasks = []

    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = os.path.join(dataset_path, label_name)

        if not os.path.isdir(folder):
            print(f"❌ Missing folder: {folder}")
            continue

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        print(f"📊 {label_name}: {len(files)} images")

        for f in files:
            all_tasks.append((f, label_val))

    print(f"🚀 Total images: {len(all_tasks)}")

    # -------- MULTIPROCESSING --------
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_image, all_tasks),
            total=len(all_tasks),
            ncols=100
        ))

    # -------- STREAM SAVE --------
    valid_count = 0

    with open(output_file, "wb") as f:
        for res in results:
            if res is None:
                continue

            feat, label = res
            pickle.dump((feat, label), f)
            valid_count += 1

    print(f"💾 Saved: {output_file}")
    print(f"✅ Valid samples: {valid_count}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../real-vs-fake")
    )

    os.makedirs("combined_features", exist_ok=True)

    print("\n🔥 Starting optimized feature pipeline...\n")

    for split in ["train", "valid", "test"]:
        process_dataset(
            os.path.join(base_path, split),
            f"combined_features/{split}_features.pkl"
        )

    print("\n🎉 All datasets processed successfully!")