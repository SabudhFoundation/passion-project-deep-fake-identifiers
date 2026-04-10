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
        rbf_components=4000,
        random_state=42
    ):
        """
        Enhanced SVM Classifier

        New:
        - Threshold tuning (boosts accuracy significantly)
        - Better evaluation (uses decision scores)

        model_type:
        - 'sgd'
        - 'linear'
        - 'rbf_approx'
        """

        self.model_type = model_type
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.rbf_components = rbf_components
        self.random_state = random_state

        self.best_threshold = 0.0  # 🔥 NEW

        self.model = self.build_pipeline()

    # =====================================
    # BUILDERS
    # =====================================
    def build_pca(self):
        return PCA(
            n_components=self.pca_components,
            svd_solver="randomized",
            random_state=self.random_state
        )

    def build_rbf(self):
        return RBFSampler(
            gamma=self.gamma,
            n_components=self.rbf_components,
            random_state=self.random_state
        )

    def build_classifier(self):

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

        elif self.model_type == "linear":
            return LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=3000,
                tol=1e-3
            )

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
            raise ValueError("❌ Invalid model_type")

    def build_pipeline(self):
        steps = []

        if self.use_pca:
            steps.append(("pca", self.build_pca()))

        if self.model_type == "rbf_approx":
            steps.append(("rbf", self.build_rbf()))

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
    # DECISION SCORES
    # =====================================
    def decision_scores(self, X):
        return self.model.decision_function(X)

    # =====================================
    # THRESHOLD TUNING
    # =====================================
    def tune_threshold(self, X, y):
        """
        Find best threshold using validation data
        """
        scores = self.decision_scores(X)

        thresholds = np.linspace(scores.min(), scores.max(), 200)

        best_acc = 0
        best_t = 0

        for t in thresholds:
            y_pred = (scores > t).astype(int)
            acc = accuracy_score(y, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_t = t

        self.best_threshold = best_t

        print(f"🔥 Best Threshold: {best_t:.4f}")
        print(f"🔥 Tuned Accuracy: {best_acc:.4f}")

        return best_t, best_acc

    # =====================================
    # PREDICT (WITH THRESHOLD)
    # =====================================
    def predict(self, X):
        scores = self.decision_scores(X)
        return (scores > self.best_threshold).astype(int)

    # =====================================
    # EVALUATE
    # =====================================
    def evaluate(self, X, y, use_threshold=True):
        if use_threshold:
            y_pred = self.predict(X)
        else:
            y_pred = self.model.predict(X)

        acc = accuracy_score(y, y_pred)

        print(f"\n🎯 Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y, y_pred))

        try:
            scores = self.decision_scores(X)
            auc = roc_auc_score(y, scores)
            print(f"\n📈 ROC-AUC: {auc:.4f}")
        except:
            print("⚠️ ROC-AUC not available")

        return acc

    # =====================================
    # SAVE / LOAD
    # =====================================
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.best_threshold
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.best_threshold = data.get("threshold", 0.0)