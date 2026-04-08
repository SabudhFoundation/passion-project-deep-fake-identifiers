import pickle
import time
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler


class SVMClassifier:
    def __init__(self,
                 model_type="linear",
                 C=1.0,
                 gamma=0.01,
                 probability=False,
                 class_weight=None,
                 use_pca=False,
                 pca_components=100,
                 rbf_components=2000):

        self.model_type = model_type

        # =====================================
        # 🔥 PREPROCESSING
        # =====================================
        log_transform = FunctionTransformer(np.log1p)
        scaler = StandardScaler()

        steps = [log_transform, scaler]

        # =====================================
        # 🔥 OPTIONAL PCA (VERY IMPORTANT)
        # =====================================
        if use_pca:
            steps.append(PCA(n_components=pca_components))

        # =====================================
        # MODEL SELECTION
        # =====================================
        if model_type == "rbf":
            base_model = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=probability,
                class_weight=class_weight,
                cache_size=1000
            )
            steps.append(base_model)

        elif model_type == "linear":
            base_model = LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=2000,
                tol=1e-3
            )
            steps.append(base_model)

        elif model_type == "sgd":
            base_model = SGDClassifier(
                loss="hinge",
                alpha=1 / C,
                max_iter=1000,
                class_weight=class_weight
            )
            steps.append(base_model)

        # 🔥 NONLINEAR (BEST FOR YOU)
        elif model_type == "rbf_approx":
            steps.append(RBFSampler(gamma=gamma, n_components=rbf_components))

            base_model = SGDClassifier(
                loss="hinge",
                max_iter=1000,
                class_weight=class_weight
            )
            steps.append(base_model)

        else:
            raise ValueError("Invalid model_type")

        # =====================================
        # FINAL PIPELINE
        # =====================================
        self.model = make_pipeline(*steps)

    # =====================================
    # TRAIN
    # =====================================
    def train(self, X_train, y_train):
        print(f"\n🚀 Training {self.model_type.upper()} model...")
        start = time.time()

        self.model.fit(X_train, y_train)

        print(f"✅ Training complete (Time: {time.time() - start:.2f} sec)")

    # =====================================
    # PREDICT
    # =====================================
    def predict(self, X):
        return self.model.predict(X)

    # =====================================
    # PROBABILITY (only for true RBF)
    # =====================================
    def predict_proba(self, X):
        if self.model_type != "rbf":
            raise ValueError("⚠️ Probability only available for RBF SVC")
        return self.model.predict_proba(X)

    # =====================================
    # EVALUATE
    # =====================================
    def evaluate(self, X, y, verbose=True):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)

        if verbose:
            print(f"\n🎯 Accuracy: {acc:.4f}")
            print("\n📊 Classification Report:")
            print(classification_report(y, y_pred))

        return acc

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"💾 Model saved → {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        print(f"📂 Model loaded ← {filepath}")