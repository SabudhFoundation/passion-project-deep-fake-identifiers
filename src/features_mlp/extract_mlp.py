import cv2
import numpy as np

from scipy.stats import (
    skew,
    kurtosis
)


class ExtraExtractorMLP:

    def extract(self, image):

        image = cv2.resize(
            image,
            (96, 96)
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        features = []

        # -------------------
        # COLOR FEATURES (12)
        # -------------------

        for channel in cv2.split(image):

            features.extend([

                channel.mean(),
                channel.std(),

                skew(
                    channel.ravel()
                ),

                kurtosis(
                    channel.ravel()
                )

            ])

        # -------------------
        # EDGE FEATURES (3)
        # -------------------

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        features.extend([

            edges.mean(),
            edges.std(),

            np.count_nonzero(
                edges
            ) / edges.size

        ])

        # -------------------
        # GLOBAL STATS (4)
        # -------------------

        features.extend([

            gray.mean(),
            gray.std(),

            skew(
                gray.ravel()
            ),

            kurtosis(
                gray.ravel()
            )

        ])

        return np.asarray(
            features,
            dtype=np.float32
        )