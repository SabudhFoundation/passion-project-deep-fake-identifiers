from scipy.stats import skew, kurtosis
import cv2
import numpy as np


def extract_fft_features(
    image,
    img_size=128,
    bands=None,
    hf_threshold=30
):

    if image is None:
        return None

    if bands is None:
        bands = [
            (0, 8),
            (8, 16),
            (16, 30),
            (30, 45),
            (45, 64),
        ]

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    image = cv2.resize(
        image,
        (img_size, img_size)
    )

    image = image.astype(
        np.float32
    ) / 255.0

    h, w = image.shape

    cy, cx = h // 2, w // 2

    hann = np.outer(
        np.hanning(h),
        np.hanning(w)
    )

    image *= hann

    fft = np.fft.fft2(image)

    fft = np.fft.fftshift(fft)

    magnitude = np.log1p(
        np.abs(fft)
    )

    phase = np.angle(fft)

    y, x = np.ogrid[:h, :w]

    radius = np.hypot(
        x - cx,
        y - cy
    )

    features = []

    for r1, r2 in bands:

        region = magnitude[
            (radius >= r1) &
            (radius < r2)
        ]

        if region.size == 0:

            features.extend(
                [0] * 7
            )

            continue

        features.extend([

            np.mean(region),

            np.std(region),

            np.sum(
                region ** 2
            ),

            np.max(region),

            np.min(region),

            skew(region),

            kurtosis(region)
        ])

    hf = magnitude[
        radius >= hf_threshold
    ]

    lf = magnitude[
        radius < hf_threshold
    ]

    features.extend([

        np.sum(hf ** 2)
        /
        (np.sum(lf ** 2) + 1e-8),

        magnitude.mean(),

        magnitude.std(),

        np.std(phase),

        np.mean(
            np.abs(phase)
        )
    ])

    return np.asarray(
        features,
        dtype=np.float32
    )