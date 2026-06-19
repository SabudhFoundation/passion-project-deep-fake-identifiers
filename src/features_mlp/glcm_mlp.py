import cv2
import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops
)


class GLCMExtractorMLP:

    def extract(self, image):

        image = cv2.resize(
            image,
            (96, 96),
            interpolation=cv2.INTER_AREA
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = (gray >> 2).astype(
            np.uint8
        )

        glcm = graycomatrix(
            gray,
            distances=[1, 2, 4],
            angles=[
                0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4
            ],
            levels=64,
            symmetric=True,
            normed=True
        )

        props = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM"
        ]

        features = []

        for prop in props:

            values = graycoprops(
                glcm,
                prop
            )

            features.extend(
                values.ravel()
            )

        return np.asarray(
            features,
            dtype=np.float32
        )