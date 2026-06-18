from skimage.feature import local_binary_pattern
import numpy as np
import cv2


def extract_lbp_features(
    image,
    img_size=128,
    patch_grid=2,
    radii=(1, 2)
):

    if image is None:
        return None

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    image = cv2.resize(
        image,
        (img_size, img_size)
    )

    h, w = image.shape

    ph = h // patch_grid
    pw = w // patch_grid

    features = []

    # --------------------------
    # Patch-level LBP
    # --------------------------
    for i in range(patch_grid):

        for j in range(patch_grid):

            patch = image[
                i * ph:(i + 1) * ph,
                j * pw:(j + 1) * pw
            ]

            for radius in radii:

                points = 8 * radius

                lbp = local_binary_pattern(
                    patch,
                    points,
                    radius,
                    method="uniform"
                )

                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=points + 2,
                    range=(0, points + 2)
                )

                hist = hist.astype(
                    np.float32
                )

                hist /= (
                    hist.sum() + 1e-8
                )

                features.extend(hist)

    # --------------------------
    # Global LBP Histogram
    # --------------------------
    global_lbp = local_binary_pattern(
        image,
        8,
        1,
        method="uniform"
    )

    global_hist, _ = np.histogram(
        global_lbp.ravel(),
        bins=10,
        range=(0, 10)
    )

    global_hist = global_hist.astype(
        np.float32
    )

    global_hist /= (
        global_hist.sum() + 1e-8
    )

    features.extend(global_hist)

    return np.asarray(
        features,
        dtype=np.float32
    )