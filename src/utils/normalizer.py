import numpy as np
import pickle


class FeatureNormalizer:
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
        """
        Modular Feature Normalizer

        Features:
        - Separate normalization for LBP, GLCM, FFT
        - Can use individually or combined
        - Designed for experimentation and debugging

        Parameters:
        - lbp_dim: number of LBP features
        - glcm_dim: number of GLCM features
        - fft_weight: controls FFT importance
        - glcm_weight: controls GLCM importance
        """

        self.lbp_dim = lbp_dim
        self.glcm_dim = glcm_dim

        self.use_log_fft = use_log_fft
        self.use_energy_norm = use_energy_norm

        self.fft_weight = fft_weight
        self.glcm_weight = glcm_weight
        self.lbp_scale = lbp_scale

        # stores mean/std for each feature
        self.params = {}

    # =====================================
    # SPLIT FEATURES
    # =====================================
    def split_features(self, X):
        """
        Split combined feature vector into:
        - LBP
        - GLCM
        - FFT
        """
        lbp = X[:, :self.lbp_dim]
        glcm = X[:, self.lbp_dim:self.lbp_dim + self.glcm_dim]
        fft = X[:, self.lbp_dim + self.glcm_dim:]
        return lbp, glcm, fft

    # =====================================
    # LBP NORMALIZATION
    # =====================================
    def normalize_lbp_fit(self, lbp):
        """
        Compute mean and std for LBP
        """
        if self.lbp_scale:
            self.params["lbp_mean"] = np.mean(lbp, axis=0)
            self.params["lbp_std"] = np.std(lbp, axis=0) + 1e-8

    def normalize_lbp_transform(self, lbp):
        """
        Apply normalization to LBP
        """
        if self.lbp_scale:
            lbp = (lbp - self.params["lbp_mean"]) / self.params["lbp_std"]
        return lbp

    # =====================================
    # GLCM NORMALIZATION
    # =====================================
    def normalize_glcm_fit(self, glcm):
        """
        Compute mean and std for GLCM
        """
        self.params["glcm_mean"] = np.mean(glcm, axis=0)
        self.params["glcm_std"] = np.std(glcm, axis=0) + 1e-8

    def normalize_glcm_transform(self, glcm):
        """
        Normalize and apply weight scaling
        """
        glcm = (glcm - self.params["glcm_mean"]) / self.params["glcm_std"]
        return glcm * self.glcm_weight

    # =====================================
    # FFT PREPROCESSING
    # =====================================
    def process_fft(self, fft):
        """
        Apply preprocessing on FFT:
        1. Energy normalization
        2. Log scaling (reduces extreme values)
        """
        if self.use_energy_norm:
            fft = fft / (np.sum(np.abs(fft), axis=1, keepdims=True) + 1e-8)

        if self.use_log_fft:
            fft = np.log1p(np.abs(fft))

        return fft

    # =====================================
    # FFT NORMALIZATION
    # =====================================
    def normalize_fft_fit(self, fft):
        """
        Compute mean/std AFTER preprocessing
        """
        fft = self.process_fft(fft)
        self.params["fft_mean"] = np.mean(fft, axis=0)
        self.params["fft_std"] = np.std(fft, axis=0) + 1e-8

    def normalize_fft_transform(self, fft):
        """
        Normalize FFT + apply weight scaling
        """
        fft = self.process_fft(fft)
        fft = (fft - self.params["fft_mean"]) / self.params["fft_std"]
        return fft * self.fft_weight

    # =====================================
    # FIT ALL FEATURES
    # =====================================
    def fit(self, X):
        """
        Fit all feature normalizers together
        """
        lbp, glcm, fft = self.split_features(X)

        self.normalize_lbp_fit(lbp)
        self.normalize_glcm_fit(glcm)
        self.normalize_fft_fit(fft)

    # =====================================
    # TRANSFORM ALL FEATURES
    # =====================================
    def transform(self, X):
        """
        Apply normalization to all features
        """
        lbp, glcm, fft = self.split_features(X)

        lbp = self.normalize_lbp_transform(lbp)
        glcm = self.normalize_glcm_transform(glcm)
        fft = self.normalize_fft_transform(fft)

        return np.concatenate([lbp, glcm, fft], axis=1)

    # =====================================
    # FIT + TRANSFORM
    # =====================================
    def fit_transform(self, X):
        """
        Convenience function
        """
        self.fit(X)
        return self.transform(X)

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, path):
        """
        Save full normalizer state
        """
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        """
        Load normalizer state
        """
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))