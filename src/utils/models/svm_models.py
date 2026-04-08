import pickle
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, classification_report


class SVMClassifier:
    def __init__(self,
                 model_type="linear",
                 C=1.0,
                 gamma=0.01,
                 probability=False,
                 class_weight=None,
                 use_pca=False,
                 pca_components=100,
                 rbf_components=8000):
        """
        Initializes the classifier.

        Parameters:
        - model_type:
            'linear'      → Linear SVM (fast, scalable)
            'rbf'         → Kernel SVM (powerful but slow)
            'sgd'         → Linear SVM using SGD (very fast for large data)
            'rbf_approx'  → RBF kernel approximation using random features

        - C:
            Regularization parameter (higher = less regularization)

        - gamma:
            Kernel width (used in RBF)

        - probability:
            Enable probability outputs (only for SVC, slower)

        - class_weight:
            Handles imbalance (e.g., "balanced")

        - use_pca:
            Whether to reduce feature dimensionality

        - pca_components:
            Number of principal components

        - rbf_components:
            Number of random Fourier features (for RBF approximation)
        """

        self.model_type = model_type

        # Pipeline steps (PCA → Model)
        steps = []

        # =====================================
        # OPTIONAL PCA (Dimensionality Reduction)
        # =====================================
        if use_pca:
            # Reduces feature size → faster training + less noise
            steps.append(PCA(n_components=pca_components))

        # =====================================
        # MODEL SELECTION
        # =====================================

        # 🔹 Exact RBF Kernel SVM (slow but accurate)
        if model_type == "rbf":
            steps.append(
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    probability=probability,
                    class_weight=class_weight,
                    cache_size=1000   # larger cache for speed
                )
            )

        # 🔹 Linear SVM (fast and strong baseline)
        elif model_type == "linear":
            steps.append(
                LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    max_iter=2000,
                    tol=1e-3
                )
            )

        # 🔹 SGD-based linear classifier (scales to huge datasets)
        elif model_type == "sgd":
            steps.append(
                SGDClassifier(
                    loss="hinge",       # same as linear SVM
                    alpha=1e-4,         # regularization strength
                    max_iter=2000,
                    class_weight=class_weight
                )
            )

        # 🔹 RBF Approximation (FAST + NON-LINEAR 🔥)
        elif model_type == "rbf_approx":
            # Step 1: Map data to higher dimension using random Fourier features
            steps.append(RBFSampler(gamma=gamma, n_components=rbf_components))

            # Step 2: Train linear classifier on transformed features
            steps.append(
                SGDClassifier(
                    loss="hinge",
                    max_iter=1000,
                    class_weight=class_weight
                )
            )

        else:
            raise ValueError("Invalid model_type")

        # Create pipeline (applies steps sequentially)
        self.model = make_pipeline(*steps)

    # =====================================
    # TRAIN
    # =====================================
    def train(self, X, y):
        """
        Trains the model on given data.
        """
        print(f"\n🚀 Training {self.model_type.upper()} model...")
        start = time.time()

        # Fit pipeline (applies PCA → Model training)
        self.model.fit(X, y)

        print(f"✅ Training complete (Time: {time.time() - start:.2f} sec)")

    # =====================================
    # PREDICT
    # =====================================
    def predict(self, X):
        """
        Predict labels for given input.
        """
        return self.model.predict(X)

    # =====================================
    # EVALUATE
    # =====================================
    def evaluate(self, X, y):
        """
        Evaluates model performance.
        Outputs:
        - Accuracy
        - Precision, Recall, F1-score
        """
        y_pred = self.predict(X)

        # Compute accuracy
        acc = accuracy_score(y, y_pred)

        print(f"\n🎯 Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y, y_pred))

        return acc

    # =====================================
    # SAVE
    # =====================================
    def save(self, path):
        """
        Saves trained model to disk.
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"💾 Model saved → {path}")

    # =====================================
    # LOAD
    # =====================================
    def load(self, path):
        """
        Loads model from disk.
        """
        with open(path, "rb") as f:
            self.model = pickle.load(f)

        print(f"📂 Model loaded ← {path}")