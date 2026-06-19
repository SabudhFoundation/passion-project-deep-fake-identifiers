import cv2
import numpy as np

from scipy.stats import (
    skew,
    kurtosis
)


class FFTExtractorMLP:

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

        fft = np.fft.fft2(
            gray
        )

        fft = np.fft.fftshift(
            fft
        )

        mag = np.log1p(
            np.abs(fft)
        )

        h, w = mag.shape

        center = mag[
            h // 2 - 16:h // 2 + 16,
            w // 2 - 16:w // 2 + 16
        ]

        features = [

            mag.mean(),
            mag.std(),
            mag.max(),
            mag.min(),

            center.mean(),
            center.std(),

            skew(
                mag.ravel()
            ),

            kurtosis(
                mag.ravel()
            )
        ]

        percentiles = np.percentile(
            mag,
            np.linspace(
                2,
                100,
                40
            )
        )

        features.extend(
            percentiles
        )

        cy = h // 2
        cx = w // 2

        Y, X = np.ogrid[
            :h,
            :w
        ]

        radius = np.sqrt(
            (X - cx) ** 2 +
            (Y - cy) ** 2
        )

        bands = np.linspace(
            0,
            radius.max(),
            17
        )

        for i in range(16):

            mask = (
                (radius >= bands[i]) &
                (radius < bands[i + 1])
            )

            values = mag[
                mask
            ]

            features.extend([

                values.mean(),
                values.std()

            ])

        return np.asarray(
            features,
            dtype=np.float32
        )