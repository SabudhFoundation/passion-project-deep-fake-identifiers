import os
import cv2
import pickle
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from scipy.stats import skew, kurtosis

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)


class FeatureExtractor:

    def __init__(
        self,
        img_size=128,
        patch_grid=2,
        lbp_radii=(1, 2)
    ):
        self.img_size = img_size
        self.patch_grid = patch_grid
        self.lbp_radii = lbp_radii

        self.glcm_props = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation"
        ]

        self.glcm_angles = [
            0,
            np.pi / 4,
            np.pi / 2
        ]

    # =========================
    # PATCHES
    # =========================
    def split_into_patches(self, image):

        h, w = image.shape

        ph = h // self.patch_grid
        pw = w // self.patch_grid

        patches = []

        for i in range(self.patch_grid):
            for j in range(self.patch_grid):

                patch = image[
                    i * ph:(i + 1) * ph,
                    j * pw:(j + 1) * pw
                ]

                patches.append(patch)

        return patches

    # =========================
    # GLCM (80 Features)
    # =========================
    def extract_glcm(self, image):

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        patches = self.split_into_patches(
            image
        )

        features = []

        for patch in patches:

            glcm = graycomatrix(
                patch,
                distances=[1],
                angles=self.glcm_angles,
                symmetric=True,
                normed=True
            )

            for prop in self.glcm_props:

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

    # =========================
    # LBP (122 Features)
    # =========================
    def extract_lbp(self, image):

        patches = self.split_into_patches(
            image
        )

        features = []

        for patch in patches:

            for radius in self.lbp_radii:

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

        features.extend(
            global_hist
        )

        return np.asarray(
            features,
            dtype=np.float32
        )

    # =========================
    # FFT (40 Features)
    # =========================
    def extract_fft(self, image):

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

        bands = [
            (0, 8),
            (8, 16),
            (16, 30),
            (30, 45),
            (45, 64)
        ]

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

                np.sum(region ** 2),

                np.max(region),

                np.min(region),

                skew(region),

                kurtosis(region)
            ])

        hf = magnitude[
            radius >= 30
        ]

        lf = magnitude[
            radius < 30
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

    # =========================
    # MAIN
    # =========================
    def extract(
        self,
        image,
        use_glcm=True,
        use_lbp=True,
        use_fft=True
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
            (
                self.img_size,
                self.img_size
            )
        )

        features = []

        if use_glcm:
            features.append(
                self.extract_glcm(image)
            )

        if use_lbp:
            features.append(
                self.extract_lbp(image)
            )

        if use_fft:
            features.append(
                self.extract_fft(image)
            )

        return np.concatenate(
            features
        ).astype(np.float32)


# =========================
# MULTIPROCESS HELPER
# =========================
def process_image(args):

    path, label, extractor = args

    img = cv2.imread(path)

    if img is None:
        return None

    try:

        feat = extractor.extract(img)

        return feat, label

    except Exception:
        return None


# =========================
# DATASET PROCESSING
# =========================
def process_dataset(
    dataset_path,
    extractor
):

    tasks = []

    for label_name, label_val in [

        ("real", 0),
        ("fake", 1)

    ]:

        folder = os.path.join(
            dataset_path,
            label_name
        )

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                tasks.append(

                    (
                        os.path.join(
                            folder,
                            file
                        ),
                        label_val,
                        extractor
                    )
                )

    with Pool(cpu_count()) as pool:

        results = list(

            tqdm(
                pool.imap(
                    process_image,
                    tasks
                ),
                total=len(tasks)
            )
        )

    X = []
    y = []

    for result in results:

        if result is None:
            continue

        feat, label = result

        X.append(feat)
        y.append(label)

    return (

        np.asarray(
            X,
            dtype=np.float32
        ),

        np.asarray(
            y,
            dtype=np.int32
        )
    )


# =========================
# SAVE FEATURES
# =========================
def save_features(
    dataset_path,
    output_file,
    extractor
):

    X, y = process_dataset(
        dataset_path,
        extractor
    )

    with open(
        output_file,
        "wb"
    ) as f:

        pickle.dump(
            (X, y),
            f
        )

    print(
        f"Saved {output_file}"
    )

    print(
        f"Shape: {X.shape}"
    )