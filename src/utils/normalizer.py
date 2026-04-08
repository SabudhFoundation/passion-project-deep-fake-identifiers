import numpy as np
import pickle


class FeatureNormalizer:
    def __init__(self, method="standard", use_log=True, use_energy_norm=False):
        """
        method: 'minmax', 'standard', 'robust'
        use_log: apply safe log transform (recommended for FFT)
        use_energy_norm: normalize each sample (very useful for FFT)
        """
        self.method = method
        self.use_log = use_log
        self.use_energy_norm = use_energy_norm
        self.params = {}

    # =====================================
    # PREPROCESS (FFT-SAFE PIPELINE)
    # =====================================
    def _preprocess(self, X):
        # ✅ Energy normalization (per sample)
        if self.use_energy_norm:
            X = X / (np.sum(np.abs(X), axis=1, keepdims=True) + 1e-8)

        # ✅ Safe log transform
        if self.use_log:
            X = np.abs(X)
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=0.0)
            X = np.log1p(X)

        return X

    # =====================================
    # FIT
    # =====================================
    def fit(self, X):
        X = self._preprocess(X)

        if self.method == "minmax":
            self.params["min"] = np.min(X, axis=0)
            self.params["max"] = np.max(X, axis=0)

        elif self.method == "standard":
            self.params["mean"] = np.mean(X, axis=0)
            self.params["std"] = np.std(X, axis=0) + 1e-8

        elif self.method == "robust":
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
        X = self._preprocess(X)

        if self.method == "minmax":
            return (X - self.params["min"]) / (self.params["max"] - self.params["min"] + 1e-8)

        elif self.method == "standard":
            return (X - self.params["mean"]) / self.params["std"]

        elif self.method == "robust":
            return (X - self.params["median"]) / self.params["iqr"]

        else:
            raise ValueError("Invalid method")

    # =====================================
    # FIT + TRANSFORM
    # =====================================
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({
                "method": self.method,
                "use_log": self.use_log,
                "use_energy_norm": self.use_energy_norm,
                "params": self.params
            }, f)
        print(f"💾 Normalizer saved → {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.method = data["method"]
            self.use_log = data["use_log"]
            self.use_energy_norm = data.get("use_energy_norm", False)
            self.params = data["params"]
        print(f"📂 Normalizer loaded ← {filepath}")