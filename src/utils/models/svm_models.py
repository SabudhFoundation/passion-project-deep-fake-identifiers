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

        self.model_type = model_type

        steps = []

        # OPTIONAL PCA
        if use_pca:
            steps.append(PCA(n_components=pca_components))

        # MODEL SELECTION
        if model_type == "rbf":
            steps.append(
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    probability=probability,
                    class_weight=class_weight,
                    cache_size=1000
                )
            )

        elif model_type == "linear":
            steps.append(
                LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    max_iter=2000,
                    tol=1e-3
                )
            )

        elif model_type == "sgd":
            steps.append(
                SGDClassifier(
                    loss="hinge",
                    alpha = 1e-4,
                    max_iter=2000,
                    class_weight=class_weight
                )
            )

        elif model_type == "rbf_approx":
            steps.append(RBFSampler(gamma=gamma, n_components=rbf_components))
            steps.append(
                SGDClassifier(
                    loss="hinge",
                    max_iter=1000,
                    class_weight=class_weight
                )
            )

        else:
            raise ValueError("Invalid model_type")

        self.model = make_pipeline(*steps)

    # TRAIN
    def train(self, X, y):
        print(f"\n🚀 Training {self.model_type.upper()} model...")
        start = time.time()
        self.model.fit(X, y)
        print(f"✅ Training complete (Time: {time.time() - start:.2f} sec)")

    # PREDICT
    def predict(self, X):
        return self.model.predict(X)

    # EVALUATE
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)

        print(f"\n🎯 Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y, y_pred))

        return acc

    # SAVE
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"💾 Model saved → {path}")

    # LOAD
    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print(f"📂 Model loaded ← {path}")