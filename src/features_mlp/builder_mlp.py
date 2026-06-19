import numpy as np

from src.features_mlp.glcm_mlp import GLCMExtractorMLP
from src.features_mlp.lbp_mlp import LBPExtractorMLP
from src.features_mlp.fft_mlp import FFTExtractorMLP
from src.features_mlp.extract_mlp import ExtraExtractorMLP


class FeatureBuilderMLP:

    def __init__(
        self,
        use_lbp=True,
        use_glcm=True,
        use_fft=True
    ):

        self.use_lbp = use_lbp
        self.use_glcm = use_glcm
        self.use_fft = use_fft

        self.glcm = GLCMExtractorMLP()
        self.lbp = LBPExtractorMLP()
        self.fft = FFTExtractorMLP()

        # Color + Edge + Global Stats
        self.extra = ExtraExtractorMLP()

    def extract_features(
        self,
        image
    ):

        features = []

        if self.use_glcm:

            features.append(
                self.glcm.extract(
                    image
                )
            )

        if self.use_lbp:

            features.append(
                self.lbp.extract(
                    image
                )
            )

        if self.use_fft:

            features.append(
                self.fft.extract(
                    image
                )
            )

        # Add extra features only for combined model
        if (
            self.use_glcm
            and self.use_lbp
            and self.use_fft
        ):

            features.append(
                self.extra.extract(
                    image
                )
            )

        return np.concatenate(
            [
                np.asarray(
                    feat,
                    dtype=np.float32
                ).flatten()
                for feat in features
            ]
        )