import cv2
import numpy as np


def extract_fft_features(image):
    """
    Strong FFT feature extractor for deepfake detection
    """

    # ===============================
    # 1. Convert to grayscale
    # ===============================
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ===============================
    # 2. Resize + normalize
    # ===============================
    image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0

    h, w = image.shape
    ch, cw = h // 2, w // 2

    # ===============================
    # 3. Apply Hann window (IMPORTANT)
    # ===============================
    hann = np.outer(np.hanning(h), np.hanning(w))
    image = image * hann

    # ===============================
    # 4. FFT
    # ===============================
    fshift = np.fft.fftshift(np.fft.fft2(image))

    magnitude = np.log1p(np.abs(fshift))
    phase = np.angle(fshift)

    features = []

    # ===============================
    # 5. Radius map
    # ===============================
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cw, y - ch)

    # ===============================
    # 6. Frequency bands
    # ===============================
    bands = [
        (0, 8),
        (8, 16),
        (16, 30),
        (30, 45),
        (45, 64)
    ]

    for r1, r2 in bands:
        mask = (r >= r1) & (r < r2)
        region = magnitude[mask]

        features.extend([
            np.mean(region),
            np.std(region),
            np.sum(region ** 2)
        ])

    # ===============================
    # 7. High-frequency ratio
    # ===============================
    hf = magnitude[r >= 30]
    lf = magnitude[r < 30]

    features.append(np.sum(hf**2) / (np.sum(lf**2) + 1e-8))

    # ===============================
    # 8. Global stats
    # ===============================
    features.extend([
        magnitude.mean(),
        magnitude.std()
    ])

    # ===============================
    # 9. Phase features
    # ===============================
    features.extend([
        np.std(phase),
        np.mean(np.abs(phase))
    ])

    return np.array(features, dtype=np.float32)