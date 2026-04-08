import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class SVMClassifier:
    def __init__(self, kernel="rbf", C=1.0, gamma="scale", probability=True):
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability
        )

    # =====================================
    # 1. TRAIN
    # =====================================
    def train(self, X_train, y_train):
        print("🚀 Training SVM...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete")

    # =====================================
    # 2. PREDICT
    # =====================================
    def predict(self, X):
        return self.model.predict(X)

    # =====================================
    # 3. PREDICT PROBA
    # =====================================
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # =====================================
    # 4. EVALUATE
    # =====================================
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)

        print(f"🎯 Accuracy: {acc:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y, y_pred))

        return acc

    # =====================================
    # 5. SAVE / LOAD
    # =====================================
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"💾 Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        print(f"📂 Model loaded from {filepath}")