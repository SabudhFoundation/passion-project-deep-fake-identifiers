import pickle
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


class SVMClassifier:
    def __init__(
        self,
        model_type="sgd",
        C=1.0,
        gamma=0.01,
        class_weight="balanced",
        use_pca=False,
        pca_components=80,
        rbf_components=1200,
        random_state=42
    ):
        """
        Modular SVM Classifier

        Features:
        - Separate builders for PCA, RBF, Classifier
        - Flexible pipeline creation
        - Easy experimentation

        model_type:
        - 'sgd'        → fast, scalable
        - 'linear'     → strong baseline
        - 'rbf_approx'→ non-linear (best accuracy)
        """

        self.model_type = model_type
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.rbf_components = rbf_components
        self.random_state = random_state

        # Final pipeline
        self.model = self.build_pipeline()

    # =====================================
    # PCA BUILDER
    # =====================================
    def build_pca(self):
        """
        Optional dimensionality reduction
        """
        return PCA(
            n_components=self.pca_components,
            svd_solver="randomized",
            random_state=self.random_state
        )

    # =====================================
    # RBF FEATURE MAPPER
    # =====================================
    def build_rbf(self):
        """
        Approximate RBF kernel using random features
        """
        return RBFSampler(
            gamma=self.gamma,
            n_components=self.rbf_components,
            random_state=self.random_state
        )

    # =====================================
    # CLASSIFIER BUILDER
    # =====================================
    def build_classifier(self):
        """
        Choose classifier based on model_type
        """

        # 🔥 FAST + LARGE DATA
        if self.model_type == "sgd":
            return SGDClassifier(
                loss="hinge",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-3,
                class_weight=self.class_weight,
                early_stopping=True,
                validation_fraction=0.1,
                n_jobs=-1,
                random_state=self.random_state
            )

        # ⚡ STRONG LINEAR MODEL
        elif self.model_type == "linear":
            return LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=3000,
                tol=1e-3
            )

        # 🔥 NON-LINEAR (BEST FOR YOUR PROJECT)
        elif self.model_type == "rbf_approx":
            return SGDClassifier(
                loss="hinge",
                alpha=3e-5,
                max_iter=4000,
                tol=1e-4,
                class_weight=self.class_weight,
                learning_rate="optimal",
                early_stopping=False,
                n_jobs=-1,
                random_state=self.random_state
            )

        else:
            raise ValueError("❌ Use: 'sgd', 'linear', or 'rbf_approx'")

    # =====================================
    # PIPELINE BUILDER
    # =====================================
    def build_pipeline(self):
        """
        Combine all components into pipeline
        """
        steps = []

        # ---- PCA ----
        if self.use_pca:
            steps.append(("pca", self.build_pca()))

        # ---- RBF (only for non-linear) ----
        if self.model_type == "rbf_approx":
            steps.append(("rbf", self.build_rbf()))

        # ---- Classifier ----
        steps.append(("clf", self.build_classifier()))

        return Pipeline(steps)

    # =====================================
    # TRAIN
    # =====================================
    def train(self, X, y):
        print(f"\n🚀 Training {self.model_type.upper()} model...")
        start = time.time()

        self.model.fit(X, y)

        print(f"✅ Done in {time.time() - start:.2f} sec")

    # =====================================
    # PREDICT
    # =====================================
    def predict(self, X):
        return self.model.predict(X)

    # =====================================
    # DECISION SCORES
    # =====================================
    def decision_scores(self, X):
        """
        Used for ROC-AUC
        """
        clf = self.model.named_steps["clf"]
        return clf.decision_function(X)

    # =====================================
    # EVALUATE
    # =====================================
    def evaluate(self, X, y):
        y_pred = self.predict(X)

        acc = accuracy_score(y, y_pred)

        print(f"\n🎯 Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y, y_pred))

        # ---- ROC-AUC ----
        try:
            scores = self.decision_scores(X)
            auc = roc_auc_score(y, scores)
            print(f"\n📈 ROC-AUC: {auc:.4f}")
        except Exception:
            print("⚠️ ROC-AUC not available")

        return acc

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)