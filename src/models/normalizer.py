import numpy as np
import pickle


class FeatureNormalizer:
    """
    Feature Normalizer for LBP + GLCM + FFT

    Key Design:
    - Each feature normalized independently
    - FFT has special preprocessing (energy + log)
    - Supports feature weighting (optional)
    """

    def __init__(
        self,
        lbp_dim,
        glcm_dim,
        use_log_fft=True,
        use_energy_norm=True,
        fft_weight=1.0,
        glcm_weight=1.0,
        lbp_scale=True
    ):
        self.lbp_dim = lbp_dim
        self.glcm_dim = glcm_dim

        self.use_log_fft = use_log_fft
        self.use_energy_norm = use_energy_norm

        self.fft_weight = fft_weight
        self.glcm_weight = glcm_weight
        self.lbp_scale = lbp_scale

        # stores normalization parameters
        self.params = {}

    # =====================================
    # FEATURE SPLITTING
    # =====================================
    def split_features(self, X):
        """
        Split input into:
        [LBP | GLCM | FFT]
        """
        lbp = X[:, :self.lbp_dim]

        glcm_end = self.lbp_dim + self.glcm_dim
        glcm = X[:, self.lbp_dim:glcm_end]

        fft = X[:, glcm_end:]

        return lbp, glcm, fft

    # =====================================
    # LBP
    # =====================================
    def fit_lbp(self, lbp):
        """
        Compute statistics for LBP normalization
        """
        if not self.lbp_scale:
            return

        self.params["lbp_mean"] = np.mean(lbp, axis=0)
        self.params["lbp_std"] = np.std(lbp, axis=0) + 1e-8

    def transform_lbp(self, lbp):
        """
        Apply LBP normalization
        """
        if not self.lbp_scale:
            return lbp

        return (lbp - self.params["lbp_mean"]) / self.params["lbp_std"]

    # =====================================
    # GLCM
    # =====================================
    def fit_glcm(self, glcm):
        """
        Compute statistics for GLCM
        """
        self.params["glcm_mean"] = np.mean(glcm, axis=0)
        self.params["glcm_std"] = np.std(glcm, axis=0) + 1e-8

    def transform_glcm(self, glcm):
        """
        Normalize GLCM + apply weighting
        """
        glcm = (glcm - self.params["glcm_mean"]) / self.params["glcm_std"]
        return glcm * self.glcm_weight

    # =====================================
    # FFT PROCESSING
    # =====================================
    def process_fft(self, fft):
        """
        FFT preprocessing:
        1. Normalize energy per sample
        2. Log scaling (stabilizes variance)
        """

        # Energy normalization
        if self.use_energy_norm:
            energy = np.sum(np.abs(fft), axis=1, keepdims=True) + 1e-8
            fft = fft / energy

        # Log scaling
        if self.use_log_fft:
            fft = np.log1p(np.abs(fft))

        return fft

    # =====================================
    # FFT
    # =====================================
    def fit_fft(self, fft):
        """
        Compute statistics AFTER preprocessing
        """
        fft = self.process_fft(fft)

        self.params["fft_mean"] = np.mean(fft, axis=0)
        self.params["fft_std"] = np.std(fft, axis=0) + 1e-8

    def transform_fft(self, fft):
        """
        Normalize FFT + apply weighting
        """
        fft = self.process_fft(fft)
        fft = (fft - self.params["fft_mean"]) / self.params["fft_std"]

        return fft * self.fft_weight

    # =====================================
    # MAIN FIT
    # =====================================
    def fit(self, X):
        """
        Fit all feature groups independently
        """
        lbp, glcm, fft = self.split_features(X)

        self.fit_lbp(lbp)
        self.fit_glcm(glcm)
        self.fit_fft(fft)

    # =====================================
    # MAIN TRANSFORM
    # =====================================
    def transform(self, X):
        """
        Apply normalization to full feature vector
        """
        lbp, glcm, fft = self.split_features(X)

        lbp = self.transform_lbp(lbp)
        glcm = self.transform_glcm(glcm)
        fft = self.transform_fft(fft)

        return np.concatenate([lbp, glcm, fft], axis=1)

    # =====================================
    # FIT + TRANSFORM
    # =====================================
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, path):
        """
        Save normalizer (for inference consistency)
        """
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        """
        Load saved normalizer
        """
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))