"""
Unified prediction interface for deepfake detection.

Usage:
    from src.inference.predict import DeepfakePredictor
    predictor = DeepfakePredictor(model_type="svm_features")
    result = predictor.predict("path/to/image.jpg")
    print(result)  # {"label": "fake", "confidence": 0.87}
"""

import os
import pickle
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATHS = {
    "svm_fft": os.path.join(PROJECT_ROOT, "models", "svm", "svm_fft.pkl"),
    "svm_features": os.path.join(PROJECT_ROOT, "models", "svm", "svm_features.pkl"),
}

NORMALIZER_PATHS = {
    "svm_features": os.path.join(PROJECT_ROOT, "models", "svm", "normalizer_features.pkl"),
}

LABEL_MAP = {0: "real", 1: "fake"}


class DeepfakePredictor:
    def __init__(self, model_type: str = "svm_features"):
        if model_type not in MODEL_PATHS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODEL_PATHS)}")

        self.model_type = model_type

        with open(MODEL_PATHS[model_type], "rb") as f:
            self.model = pickle.load(f)

        self.normalizer = None
        if model_type in NORMALIZER_PATHS:
            from src.models.normalizer import FeatureNormalizer
            self.normalizer = FeatureNormalizer.load(NORMALIZER_PATHS[model_type])

    def extract_features(self, image_path: str) -> np.ndarray:
        from src.features.builder import FeatureBuilder
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        builder = FeatureBuilder(use_lbp=True, use_glcm=True, use_fft=True)
        return builder.extract_features(img)

    def predict(self, image_path: str) -> dict:
        features = self.extract_features(image_path)
        features = features.reshape(1, -1)

        if self.normalizer is not None:
            features = self.normalizer.transform(features)

        pred = self.model.predict(features)[0]
        label = LABEL_MAP.get(int(pred), str(pred))

        result = {"label": label, "prediction": int(pred)}

        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(features)[0]
            result["confidence"] = float(max(prob))

        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.predict <image_path> [model_type]")
        sys.exit(1)
    image_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "svm_features"
    predictor = DeepfakePredictor(model_type=model_type)
    result = predictor.predict(image_path)
    print(result)
