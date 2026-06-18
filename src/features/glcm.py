from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2


def extract_glcm_features(
    image,
    img_size=128,
    patch_grid=2
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

    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    glcm_props = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation"
    ]

    glcm_angles = [
        0,
        np.pi / 4,
        np.pi / 2
    ]

    h, w = image.shape

    ph = h // patch_grid
    pw = w // patch_grid

    features = []

    for i in range(patch_grid):

        for j in range(patch_grid):

            patch = image[
                i * ph:(i + 1) * ph,
                j * pw:(j + 1) * pw
            ]

            glcm = graycomatrix(
                patch,
                distances=[1],
                angles=glcm_angles,
                symmetric=True,
                normed=True
            )

            for prop in glcm_props:

                vals = graycoprops(
                    glcm,
                    prop
                )

                features.extend([

                    np.mean(vals),

                    np.std(vals),

                    np.max(vals),

                    np.min(vals)
                ])

    return np.asarray(
        features,
        dtype=np.float32
    )