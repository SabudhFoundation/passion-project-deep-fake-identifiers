import pickle
import json
import time
import os
import numpy as np

from normalizer import FeatureNormalizer
from models.svm_models import SVMClassifier
from sklearn.metrics import roc_auc_score


# =====================================
# CONFIG
# =====================================
np.random.seed(42)

BASE_DIR = os.path.dirname(__file__)
FEATURE_DIR = os.path.join(BASE_DIR, "combined_features")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================
# LOAD FEATURES (STREAMING)
# =====================================
def load_stream(path):
    """
    Load features stored as sequential pickle dumps
    (memory-efficient for large datasets)
    """
    features, labels = [], []

    with open(path, "rb") as f:
        while True:
            try:
                feat, label = pickle.load(f)
                features.append(feat)
                labels.append(label)
            except EOFError:
                break

    return np.array(features), np.array(labels)


# =====================================
# SAVE FEATURES (STREAMING)
# =====================================
def save_stream(X, y, path):
    """
    Save features sequentially (avoids large memory spikes)
    """
    with open(path, "wb") as f:
        for i in range(len(X)):
            pickle.dump((X[i], y[i]), f)


print("📂 Loading features...")

X_train, y_train = load_stream(os.path.join(FEATURE_DIR, "train_features.pkl"))
X_valid, y_valid = load_stream(os.path.join(FEATURE_DIR, "valid_features.pkl"))
X_test,  y_test  = load_stream(os.path.join(FEATURE_DIR, "test_features.pkl"))

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")


# =====================================
# FEATURE CONFIG
# =====================================
# Defines how feature vector is split
LBP_DIM = 112
GLCM_DIM = 40


# =====================================
# NORMALIZATION
# =====================================
print("\n⚙️ Applying normalization...")

norm = FeatureNormalizer(
    lbp_dim=LBP_DIM,
    glcm_dim=GLCM_DIM,

    # FFT preprocessing
    use_log_fft=True,
    use_energy_norm=True,

    # 🔥 Feature balancing (VERY IMPORTANT)
    fft_weight=1.0,      # reduce noise influence
    glcm_weight=1.0,     # emphasize texture
    lbp_scale=True       # normalize histogram safely
)

# Fit ONLY on training data
norm.fit(X_train)

# Apply to all splits
X_train = norm.transform(X_train)
X_valid = norm.transform(X_valid)
X_test  = norm.transform(X_test)


# =====================================
# SAVE NORMALIZED FEATURES
# =====================================
save_stream(X_train, y_train, os.path.join(FEATURE_DIR, "train_norm.pkl"))
save_stream(X_valid, y_valid, os.path.join(FEATURE_DIR, "valid_norm.pkl"))
save_stream(X_test,  y_test,  os.path.join(FEATURE_DIR, "test_norm.pkl"))

print("💾 Normalized features saved")

# Save normalizer for inference
norm.save(os.path.join(MODEL_DIR, "normalizer_features.pkl"))

print("✅ Normalization complete")


# =====================================
# HYPERPARAMETER TUNING
# =====================================
print("\n🚀 Hyperparameter tuning...\n")

param_grid = [
    {"gamma": 0.01},
    {"gamma": 0.005},
    {"gamma": 0.002},
]

best_acc = 0
best_config = None
results = []

for i, params in enumerate(param_grid, 1):

    gamma = params["gamma"]
    print(f"\n🔍 [{i}/{len(param_grid)}] gamma={gamma}")

    start = time.time()

    svm = SVMClassifier(
        model_type="rbf_approx",
        gamma=gamma,
        rbf_components=1500,
        use_pca=False,
        class_weight="balanced",
        random_state=42
    )

    svm.train(X_train, y_train)

    acc = svm.evaluate(X_valid, y_valid)

    # ROC-AUC (important for deepfake detection)
    try:
        scores = svm.decision_scores(X_valid)
        auc = roc_auc_score(y_valid, scores)
        print(f"📈 ROC-AUC: {auc:.4f}")
    except:
        auc = None

    duration = time.time() - start

    results.append({
        "gamma": gamma,
        "accuracy": acc,
        "auc": auc,
        "time": duration
    })

    if acc > best_acc:
        best_acc = acc
        best_config = params


# =====================================
# SAVE TUNING RESULTS
# =====================================
with open(os.path.join(MODEL_DIR, "tuning_results.json"), "w") as f:
    json.dump(results, f, indent=4)


print("\n🏆 Best Parameters:")
print(best_config)
print(f"Validation Accuracy = {best_acc:.4f}")


# =====================================
# FINAL TRAINING (TRAIN + VALID)
# =====================================
print("\n🚀 Training final model...")

X_full = np.vstack((X_train, X_valid))
y_full = np.hstack((y_train, y_valid))

svm = SVMClassifier(
    model_type="rbf_approx",
    gamma=best_config["gamma"],
    rbf_components=1500,
    use_pca=False,
    class_weight="balanced",
    random_state=42
)

svm.train(X_full, y_full)


# =====================================
# FINAL TEST
# =====================================
print("\n🎯 Final Test Performance:")

svm.evaluate(X_test, y_test)


# =====================================
# SAVE MODEL + METADATA
# =====================================
svm.save(os.path.join(MODEL_DIR, "svm_features.pkl"))

with open(os.path.join(MODEL_DIR, "best_params_features.json"), "w") as f:
    json.dump(best_config, f, indent=4)

with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({0: "real", 1: "fake"}, f)

print("\n🎉 Pipeline complete!")