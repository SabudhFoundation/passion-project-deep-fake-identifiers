import pickle
import json
import time
import os
import numpy as np

from normalizer import FeatureNormalizer
from models.svm_models import SVMClassifier


# =====================================
# PATH SETUP
# =====================================
BASE_DIR = os.path.dirname(__file__)
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================
# 1. LOAD FEATURES
# =====================================
print("📂 Loading features...")

with open(os.path.join(FEATURE_DIR, "train_fft_features.pkl"), "rb") as f:
    X_train, y_train = pickle.load(f)

with open(os.path.join(FEATURE_DIR, "valid_fft_features.pkl"), "rb") as f:
    X_valid, y_valid = pickle.load(f)

with open(os.path.join(FEATURE_DIR, "test_fft_features.pkl"), "rb") as f:
    X_test, y_test = pickle.load(f)

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")


# =====================================
# 🔥 2. NORMALIZATION (FINAL VERSION)
# =====================================
print("\n⚙️ Applying preprocessing (energy + log + standard)...")

norm = FeatureNormalizer(
    method="standard",
    use_log=True,
    use_energy_norm=True   # 🔥 BIG BOOST
)

X_train = norm.fit_transform(X_train)
X_valid = norm.transform(X_valid)
X_test  = norm.transform(X_test)

norm.save(os.path.join(FEATURE_DIR, "normalizer.pkl"))

print("✅ Preprocessing complete")


# =====================================
# 3. HYPERPARAMETER TUNING
# =====================================
print("\n🚀 Starting RBF Approx tuning...\n")

gamma_values = [0.02, 0.01, 0.005]   # 🔥 refined search

best_acc = 0
best_gamma = None


for i, gamma in enumerate(gamma_values, 1):

    print(f"\n🔍 [{i}/{len(gamma_values)}] Testing gamma={gamma}")

    start_time = time.time()

    svm = SVMClassifier(
        model_type="rbf_approx",
        gamma=gamma,
        rbf_components=4000,   # 🔥 more power
        use_pca=True,
        pca_components=100,
        class_weight="balanced"
    )

    svm.train(X_train, y_train)

    acc = svm.evaluate(X_valid, y_valid)

    print(f"⏱ Time: {time.time() - start_time:.2f} sec")

    if acc > best_acc:
        best_acc = acc
        best_gamma = gamma


# =====================================
# 4. BEST PARAMETERS
# =====================================
print("\n🏆 Best Parameter Found:")
print(f"gamma = {best_gamma}")
print(f"Validation Accuracy = {best_acc:.4f}")


# =====================================
# 5. FINAL TRAINING
# =====================================
print("\n🚀 Training final model on FULL data...")

X_full = np.vstack((X_train, X_valid))
y_full = np.hstack((y_train, y_valid))

start_time = time.time()

svm = SVMClassifier(
    model_type="rbf_approx",
    gamma=best_gamma,
    rbf_components=4000,
    use_pca=True,
    pca_components=100,
    class_weight="balanced"
)

svm.train(X_full, y_full)

print(f"⏱ Final training time: {time.time() - start_time:.2f} sec")


# =====================================
# 6. FINAL TEST
# =====================================
print("\n🎯 Final Test Performance:")
svm.evaluate(X_test, y_test)


# =====================================
# 7. SAVE MODEL + PARAMS
# =====================================
svm.save(os.path.join(MODEL_DIR, "svm_fft.pkl"))

with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
    json.dump({"gamma": best_gamma}, f)

with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({0: "real", 1: "fake"}, f)

print("\n🎉 Pipeline complete!")