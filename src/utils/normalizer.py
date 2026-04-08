import numpy as np
import pickle


class FeatureNormalizer:
    def __init__(self, method="standard", use_log=True, use_energy_norm=False):
        """
        Initializes the normalizer.

        Parameters:
        - method:
            'minmax'   → scales features to [0, 1]
            'standard' → zero mean, unit variance (most common)
            'robust'   → uses median & IQR (good for outliers)

        - use_log:
            Apply log transform (important for FFT since values are skewed)

        - use_energy_norm:
            Normalize each sample independently (helps FFT consistency)
        """
        self.method = method
        self.use_log = use_log
        self.use_energy_norm = use_energy_norm

        # Stores computed statistics (mean, std, etc.)
        self.params = {}

    # =====================================
    # PREPROCESS (FFT-SAFE PIPELINE)
    # =====================================
    def _preprocess(self, X):
        """
        Preprocessing before normalization.

        Steps:
        1. Energy normalization (per sample)
        2. Safe log transform
        """

        # ✅ Normalize each sample by its total energy
        # Helps remove brightness/intensity differences
        if self.use_energy_norm:
            X = X / (np.sum(np.abs(X), axis=1, keepdims=True) + 1e-8)

        # ✅ Log transform (stabilizes large FFT values)
        if self.use_log:
            X = np.abs(X)  # FFT magnitude should be positive

            # Replace NaN/Inf values with safe numbers
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=0.0)

            # log(1 + x) prevents log(0) issue
            X = np.log1p(X)

        return X

    # =====================================
    # FIT
    # =====================================
    def fit(self, X):
        """
        Learns normalization parameters from data.

        Stores:
        - min/max for minmax scaling
        - mean/std for standard scaling
        - median/IQR for robust scaling
        """
        X = self._preprocess(X)

        if self.method == "minmax":
            # Store feature-wise min and max
            self.params["min"] = np.min(X, axis=0)
            self.params["max"] = np.max(X, axis=0)

        elif self.method == "standard":
            # Store mean and std deviation
            self.params["mean"] = np.mean(X, axis=0)
            self.params["std"] = np.std(X, axis=0) + 1e-8  # avoid division by zero

        elif self.method == "robust":
            # Store median and interquartile range (IQR)
            self.params["median"] = np.median(X, axis=0)
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            self.params["iqr"] = (q3 - q1) + 1e-8

        else:
            raise ValueError("Invalid method")

    # =====================================
    # TRANSFORM
    # =====================================
    def transform(self, X):
        """
        Applies normalization using learned parameters.
        """
        X = self._preprocess(X)

        if self.method == "minmax":
            # Scale to [0, 1]
            return (X - self.params["min"]) / (
                self.params["max"] - self.params["min"] + 1e-8
            )

        elif self.method == "standard":
            # Standardization (zero mean, unit variance)
            return (X - self.params["mean"]) / self.params["std"]

        elif self.method == "robust":
            # Robust scaling (handles outliers better)
            return (X - self.params["median"]) / self.params["iqr"]

        else:
            raise ValueError("Invalid method")

    # =====================================
    # FIT + TRANSFORM
    # =====================================
    def fit_transform(self, X):
        """
        Convenience function:
        First learns parameters, then applies transformation.
        """
        self.fit(X)
        return self.transform(X)

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, filepath):
        """
        Saves normalization settings to file.
        Useful for applying same normalization during inference.
        """
        with open(filepath, "wb") as f:
            pickle.dump({
                "method": self.method,
                "use_log": self.use_log,
                "use_energy_norm": self.use_energy_norm,
                "params": self.params
            }, f)

        print(f"💾 Normalizer saved → {filepath}")

    def load(self, filepath):
        """
        Loads normalization settings from file.
        Ensures consistency between training and testing.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

            self.method = data["method"]
            self.use_log = data["use_log"]
            self.use_energy_norm = data.get("use_energy_norm", False)
            self.params = data["params"]

        print(f"📂 Normalizer loaded ← {filepath}")