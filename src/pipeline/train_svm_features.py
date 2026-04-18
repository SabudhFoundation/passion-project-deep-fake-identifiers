import pickle
import json
import os
import numpy as np
from scipy.stats import skew, kurtosis

from utils.normalizer import FeatureNormalizer
from pipeline.models.svm_models import SVMClassifier


# =====================================
# LOAD FEATURES
# =====================================
def load_stream(path):
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
# ADD STAT FEATURES
# =====================================
def add_stat_features(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    skewness = skew(X, axis=1).reshape(-1, 1)
    kurt = kurtosis(X, axis=1).reshape(-1, 1)

    return np.hstack([X, mean, std, skewness, kurt])


# =====================================
# MAIN PIPELINE FUNCTION
# =====================================
def run_hybrid_svm_pipeline(feature_dir, model_dir):

    print("📂 Loading combined features...")

    X_train, y_train = load_stream(os.path.join(feature_dir, "train_features.pkl"))
    X_valid, y_valid = load_stream(os.path.join(feature_dir, "valid_features.pkl"))
    X_test,  y_test  = load_stream(os.path.join(feature_dir, "test_features.pkl"))

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # =========================
    # ADD STAT FEATURES
    # =========================
    print("\n⚙️ Adding statistical features...")

    X_train = add_stat_features(X_train)
    X_valid = add_stat_features(X_valid)
    X_test  = add_stat_features(X_test)

    print(f"New feature dim: {X_train.shape[1]}")

    # =========================
    # NORMALIZATION
    # =========================
    print("\n⚙️ Applying normalization...")

    norm = FeatureNormalizer(
        lbp_dim=112,
        glcm_dim=40,
        use_log_fft=True,
        use_energy_norm=True,
        fft_weight=1.0,
        glcm_weight=1.0,
        lbp_scale=True
    )

    norm.fit(X_train)

    X_train = norm.transform(X_train)
    X_valid = norm.transform(X_valid)
    X_test  = norm.transform(X_test)

    os.makedirs(model_dir, exist_ok=True)
    norm.save(os.path.join(model_dir, "normalizer_features.pkl"))

    print("✅ Normalization complete")

    # =========================
    # HYPERPARAMETER TUNING
    # =========================
    print("\n🚀 Hyperparameter tuning...\n")

    gamma_values = [0.005, 0.003, 0.002, 0.0015]

    best_acc = 0
    best_gamma = None
    best_threshold = 0

    for gamma in gamma_values:
        print(f"Testing gamma={gamma}")

        svm = SVMClassifier(
            model_type="rbf_approx",
            gamma=gamma,
            rbf_components=4000,
            use_pca=True,
            pca_components=120,
            class_weight="balanced",
            random_state=42
        )

        svm.train(X_train, y_train)

        threshold, acc = svm.tune_threshold(X_valid, y_valid)

        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
            best_threshold = threshold

    print("\n🏆 Best Parameters:")
    print(f"Gamma: {best_gamma}")
    print(f"Validation Accuracy: {best_acc:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")

    # =========================
    # FINAL TRAINING
    # =========================
    print("\n🚀 Training final model...")

    X_full = np.vstack((X_train, X_valid))
    y_full = np.hstack((y_train, y_valid))

    svm = SVMClassifier(
        model_type="rbf_approx",
        gamma=best_gamma,
        rbf_components=4000,
        use_pca=True,
        pca_components=120,
        class_weight="balanced",
        random_state=42
    )

    svm.train(X_full, y_full)
    svm.best_threshold = best_threshold

    # =========================
    # FINAL TEST
    # =========================
    print("\n🎯 Final Test Performance:")
    svm.evaluate(X_test, y_test)

    # =========================
    # SAVE MODEL
    # =========================
    svm.save(os.path.join(model_dir, "svm_hybrid.pkl"))

    with open(os.path.join(model_dir, "best_params_hybrid.json"), "w") as f:
        json.dump({
            "gamma": best_gamma,
            "threshold": float(best_threshold)
        }, f, indent=4)

    print("\n🎉 Hybrid pipeline complete!")

    return {
        "gamma": best_gamma,
        "val_accuracy": best_acc,
        "threshold": best_threshold
    }