import cv2
import numpy as np

from skimage.feature import (
    local_binary_pattern
)


class LBPExtractorMLP:

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

        features = []

        configs = [
            (8, 1),
            (16, 2)
        ]

        for P, R in configs:

            lbp = local_binary_pattern(
                gray,
                P=P,
                R=R,
                method="uniform"
            )

            hist, _ = np.histogram(
                lbp.ravel(),
                bins=32,
                range=(0, 32)
            )

            hist = hist.astype(
                np.float32
            )

            hist /= (
                hist.sum() + 1e-8
            )

            features.extend(
                hist
            )

        return np.asarray(
            features,
            dtype=np.float32
        )