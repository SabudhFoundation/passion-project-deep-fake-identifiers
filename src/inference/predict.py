"""
Unified prediction interface for deepfake detection.

Supports all 12 experiment methods:
    Classical ML : lbp_svm, fft_svm, combined_svm
                   glcm_mlp, lbp_mlp, fft_mlp, combined_mlp
    CNN (Keras)  : resnet50, inceptionv3, efficientnet
    Dual Channel : dual_channel_inception, dual_channel_resnet

Usage:
    from src.inference.predict import DeepfakePredictor
    predictor = DeepfakePredictor()
    result = predictor.predict("path/to/image.jpg", method="combined_svm")
    # {"label": "fake", "confidence": 0.87, "prediction": 1}
"""

import os
import pickle
import numpy as np
import cv2
import sklearn
import sys
import hashlib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LABEL_MAP = {0: "real", 1: "fake"}

# (family, model_path_rel, extra_rel, feature_flags)
# extra_rel: normalizer path for SVM, backbone name for dual channel, None otherwise
# feature_flags: dict of {use_lbp, use_glcm, use_fft} for classical methods
METHOD_CONFIG = {
    "lbp_svm": (
        "svm",
        "models/svm/svm_lbp.pkl",
        None,
        {"use_lbp": True, "use_glcm": False, "use_fft": False},
    ),
    "fft_svm": (
        "svm",
        "models/svm/svm_fft.pkl",
        None,
        {"use_lbp": False, "use_glcm": False, "use_fft": True},
    ),
    "glcm_svm": (
        "svm",
        "models/svm/glcm_svm.pkl",
        None,
        {"use_lbp": False, "use_glcm": True, "use_fft": False},
    ),
    "combined_svm": (
        "svm",
        "models/svm/svm_features.pkl",
        None,
        {"use_lbp": True, "use_glcm": True, "use_fft": True},
    ),
    "glcm_mlp": (
        "mlp",
        "models/mlp/glcm.pkl",
        None,
        {"use_lbp": False, "use_glcm": True, "use_fft": False},
    ),
    "lbp_mlp": (
        "mlp",
        "models/mlp/lbp.pkl",
        None,
        {"use_lbp": True, "use_glcm": False, "use_fft": False},
    ),
    "fft_mlp": (
        "mlp",
        "models/mlp/fft.pkl",
        None,
        {"use_lbp": False, "use_glcm": False, "use_fft": True},
    ),
    "combined_mlp": (
        "mlp",
        "models/mlp/combined.pkl",
        None,
        {"use_lbp": True, "use_glcm": True, "use_fft": True},
    ),
    "inceptionv3": (
        "cnn_inception",
        "models/cnn/inceptionv3_finetuned.keras",
        None,
        None,
    ),
    "efficientnet": (
        "efficientnet_512",
        "models/cnn/efficientnet_classifier_512.pth",
        None,
        None,
    ),
    "dual_cnn": (
        "dual_cnn",
        "models/dual_channel/dual_cnn.pth",
        None,
        None,
    ),
}

VALID_METHODS = list(METHOD_CONFIG.keys())


class ModelNotTrainedError(Exception):
    """Raised when weight file for a method is not yet available."""


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class DeepfakePredictor:
    """Lazy-loading predictor for all deepfake detection methods."""

    def __init__(self):
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _abs(self, rel: str) -> str:
        return os.path.join(PROJECT_ROOT, rel)

    def _require_file(self, path: str, method: str) -> None:
        if not os.path.exists(path):
            raise ModelNotTrainedError(
                f"Model '{method}' weights not found at: {path}"
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_svm(
        self,
        method: str,
        model_path: str,
        normalizer_path,
        feature_flags,
    ):
        self._require_file(
            model_path,
            method
        )

        import joblib

        try:
            data = joblib.load(
                model_path
            )
        except Exception:
            with open(
                model_path,
                "rb"
            ) as f:
                data = pickle.load(f)

        if (
            isinstance(data, dict)
            and "model" in data
        ):
            pipeline = data["model"]

            threshold = float(
                data.get(
                    "threshold",
                    0.0
                )
            )

        else:
            pipeline = data
            threshold = 0.0

        print(
            f"{method} loaded:",
            type(pipeline)
        )

        if hasattr(
            pipeline,
            "n_features_in_"
        ):
            print(
                "Expected features:",
                pipeline.n_features_in_
            )

        elif hasattr(
            pipeline,
            "named_steps"
        ):
            if (
                "scaler"
                in pipeline.named_steps
            ):
                print(
                    "Expected features:",
                    pipeline.named_steps[
                        "scaler"
                    ].n_features_in_
                )

        return {
            "pipeline": pipeline,
            "threshold": threshold,
            "normalizer": None,
            "feature_flags": feature_flags,
        }

    def _load_mlp(self, method: str, model_path: str, feature_flags):
        self._require_file(model_path, method)
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        return {
            "model": payload["model"],
            "scaler": payload["scaler"],
            "feature_flags": feature_flags,
        }

    def _load_cnn(self, method: str, model_path: str, family: str):
        self._require_file(model_path, method)
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for CNN methods. Install: pip install tensorflow"
            )
        model = tf.keras.models.load_model(model_path)
        return {"model": model, "family": family}
    
    def _load_efficientnet_512(self, method: str):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for EfficientNet. "
                "Install: pip install torch torchvision"
            )

        from src.models.efficientnet512 import EfficientNet512

        model_path = self._abs(
            "models/cnn/efficientnet_classifier_512.pth"
        )

        info_path = self._abs(
            "models/cnn/encoder_info.pth"
        )

        self._require_file(model_path, method)
        self._require_file(info_path, method)

        model = EfficientNet512()

        model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu"
            )
        )

        model.eval()

        info = torch.load(
            info_path,
            map_location="cpu"
        )

        return {
            "model": model,
            "info": info
        }

    def _load_dual(self, method: str, model_path: str, backbone: str):
        self._require_file(model_path, method)
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for dual-channel methods. Install: pip install torch torchvision"
            )
        from src.models.dual_channel.detector import DualChannelDetector
        model = DualChannelDetector(backbone=backbone)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return {"model": model, "backbone": backbone}
    
    def _load_dual_cnn(self, method: str):
        import torch

        from src.models.dual_cnn import ImprovedDualStreamCNN

        model_path = self._abs(
            "models/dual_channel/dual_cnn.pth"
        )

        self._require_file(model_path, method)

        model = ImprovedDualStreamCNN()

        model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu"
            )
        )

        model.eval()

        return {"model": model}

    # ------------------------------------------------------------------
    # Cache-aware model getter
    # ------------------------------------------------------------------
    def _get_model(self, method: str) -> dict:
        if method in self._cache:
            return self._cache[method]

        family, model_rel, extra_rel, feature_flags = METHOD_CONFIG[method]
        model_path = self._abs(model_rel)

        if family == "svm":
            normalizer_path = self._abs(extra_rel) if extra_rel else None
            obj = self._load_svm(
                method,
                model_path,
                normalizer_path,
                feature_flags,
            )

        elif family == "mlp":
            obj = self._load_mlp(
                method,
                model_path,
                feature_flags,
            )

        elif family in ("cnn_resnet", "cnn_inception"):
            obj = self._load_cnn(
                method,
                model_path,
                family,
            )

        elif family == "efficientnet_512":
            obj = self._load_efficientnet_512(method)

        elif family == "dual":
            obj = self._load_dual(
                method,
                model_path,
                extra_rel,
            )
        elif family == "dual_cnn":
            obj = self._load_dual_cnn(method)
        

        else:
            raise ValueError(f"Unknown method family: {family}")

        self._cache[method] = obj
        return obj

    # ------------------------------------------------------------------
    # Feature extraction (classical methods)
    # ------------------------------------------------------------------
    def _extract_classical_features(self, image_path: str, feature_flags: dict) -> np.ndarray:
        from src.features.builder import FeatureBuilder
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        builder = FeatureBuilder(**feature_flags)
        return builder.extract_features(img)
    
    def _extract_mlp_features(
        self,
        image_path: str,
        feature_flags: dict
    ) -> np.ndarray:

        from src.features_mlp.builder_mlp import (
            FeatureBuilderMLP
        )

        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        builder = FeatureBuilderMLP(
            **feature_flags
        )

        return builder.extract_features(
            img
        )

    # ------------------------------------------------------------------
    # Predictors per family
    # ------------------------------------------------------------------
    def _predict_svm(self, image_path: str, obj: dict) -> dict:
        features = self._extract_classical_features(
            image_path,
            obj["feature_flags"]
        )

        print("Feature shape:", features.shape)
        pipeline = obj["pipeline"]

        if hasattr(pipeline, "named_steps"):

            print(
                "Pipeline steps:",
                pipeline.named_steps.keys()
            )

            if "scaler" in pipeline.named_steps:

                print(
                    "Expected features:",
                    pipeline.named_steps[
                        "scaler"
                    ].n_features_in_
                )
        print("Feature flags:", obj["feature_flags"])

        features = features.reshape(1, -1)

        if obj["normalizer"] is not None:
            features = obj["normalizer"].transform(features)

        print("Feature shape:", features.shape)

        pipeline = obj["pipeline"]

        if hasattr(pipeline, "named_steps"):
            print("Pipeline steps:", pipeline.named_steps.keys())

            if "pca" in pipeline.named_steps:
                print(
                    "PCA expects:",
                    pipeline.named_steps["pca"].n_features_in_
                )

        score = float(obj["pipeline"].decision_function(features)[0])
        pred = int(score > obj["threshold"])
        prob_fake = _sigmoid(score)
        confidence = prob_fake if pred == 1 else 1.0 - prob_fake

        return {
            "label": LABEL_MAP[pred],
            "confidence": confidence,
            "prediction": pred
        }
    def _predict_mlp(self, image_path: str, obj: dict) -> dict:
        features = self._extract_mlp_features(
            image_path,
            obj["feature_flags"]
        )
        features = obj["scaler"].transform(features.reshape(1, -1))
        proba = obj["model"].predict_proba(features)[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[pred])
        return {"label": LABEL_MAP[pred], "confidence": confidence, "prediction": pred}
    
    def _predict_inception(self, image_path: str, obj: dict) -> dict:

        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (224, 224)
        ).astype(np.float32)

        img /= 255.0

        img_batch = np.expand_dims(
            img,
            0
        )

        prob_real = float(
            obj["model"].predict(
                img_batch,
                verbose=0
            )[0][0]
        )

        pred = 0 if prob_real > 0.5 else 1

        confidence = (
            prob_real
            if pred == 0
            else 1.0 - prob_real
        )

        return {
            "label": LABEL_MAP[pred],
            "confidence": confidence,
            "prediction": pred,
        }

    def _predict_cnn(self, image_path: str, obj: dict) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32)

        img = img / 255.0

        img_batch = np.expand_dims(img, 0)

        prob_fake = float(
            obj["model"].predict(img_batch, verbose=0)[0][0]
        )

        pred = int(prob_fake > 0.5)
        confidence = prob_fake if pred == 1 else 1.0 - prob_fake

        return {
            "label": LABEL_MAP[pred],
            "confidence": confidence,
            "prediction": pred,
        }
    
    def _predict_efficientnet_512(self, image_path: str, obj: dict) -> dict:
        import torch
        from PIL import Image
        from torchvision import transforms

        img_size = obj["info"].get("image_size", 224)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        model = obj["model"]
        model.eval()

        with torch.no_grad():
            logits = model(tensor)
            prob_fake = torch.sigmoid(logits).item()

        pred = int(prob_fake > 0.5)
        confidence = prob_fake if pred == 1 else 1.0 - prob_fake

        return {
            "label": LABEL_MAP[pred],
            "confidence": confidence,
            "prediction": pred,
        }

    def _predict_dual(self, image_path: str, obj: dict) -> dict:
        import torch
        from torchvision import transforms
        from PIL import Image

        backbone = obj["backbone"]
        img_size = 299 if backbone == "inception" else 224

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            prob_fake = float(obj["model"](tensor).item())

        pred = int(prob_fake > 0.5)
        confidence = prob_fake if pred == 1 else 1.0 - prob_fake
        return {"label": LABEL_MAP[pred], "confidence": confidence, "prediction": pred}
    
    def _create_fft_image(self, image_path):
        import cv2
        import numpy as np

        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        img = cv2.resize(img, (128, 128))

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        f = np.fft.fft2(gray)

        fshift = np.fft.fftshift(f)

        magnitude = np.log(
            np.abs(fshift) + 1
        )

        magnitude = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        magnitude = magnitude.astype(np.uint8)

        magnitude = cv2.cvtColor(
            magnitude,
            cv2.COLOR_GRAY2RGB
        )

        return magnitude
    
    def _predict_dual_cnn(self, image_path: str, obj: dict) -> dict:
        import torch
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        # RGB IMAGE
        rgb_image = Image.open(
            image_path
        ).convert("RGB")

        # FFT IMAGE
        fft_image = self._create_fft_image(
            image_path
        )

        fft_image = Image.fromarray(
            fft_image
        )

        rgb_tensor = transform(
            rgb_image
        ).unsqueeze(0)

        fft_tensor = transform(
            fft_image
        ).unsqueeze(0)

        model = obj["model"]

        model.eval()

        with torch.no_grad():

            logits = model(
                rgb_tensor,
                fft_tensor
            )

            prob_fake = torch.sigmoid(
                logits
            ).item()

        pred = int(prob_fake > 0.5)

        confidence = (
            prob_fake
            if pred == 1
            else 1.0 - prob_fake
        )

        return {
            "label": LABEL_MAP[pred],
            "confidence": confidence,
            "prediction": pred,
        }


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, image_path: str, method: str) -> dict:
        """
        Run inference on a single image with the specified method.

        Returns:
            dict with keys: label ("real"/"fake"), confidence (0-1), prediction (0/1)

        Raises:
            ValueError: unknown method
            ModelNotTrainedError: weight file missing
            FileNotFoundError: image not readable
        """
        if method not in METHOD_CONFIG:
            raise ValueError(
                f"Unknown method '{method}'. Valid methods: {VALID_METHODS}"
            )

        obj = self._get_model(method)
        family = METHOD_CONFIG[method][0]

        if family == "svm":
            return self._predict_svm(image_path, obj)

        if family == "mlp":
            return self._predict_mlp(image_path, obj)

        if family in ("cnn_resnet"):
            return self._predict_cnn(image_path, obj)
        
        if method == "inceptionv3":
            return self._predict_inception(
                image_path,
                obj
            )

        if family == "efficientnet_512":
            return self._predict_efficientnet_512(image_path, obj)

        if family == "dual":
            return self._predict_dual(image_path, obj)
        if family == "dual_cnn":
            return self._predict_dual_cnn(image_path, obj)

        raise ValueError(f"Unhandled family: {family}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(f"Usage: python -m src.inference.predict <image_path> <method>")
        print(f"Methods: {VALID_METHODS}")
        sys.exit(1)

    predictor = DeepfakePredictor()
    result = predictor.predict(sys.argv[1], sys.argv[2])
    print(result)
