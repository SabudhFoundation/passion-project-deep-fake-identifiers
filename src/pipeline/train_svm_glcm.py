import pickle
import os
import numpy as np

from utils.normalizer import FeatureNormalizer
from pipeline.models.svm_models import SVMClassifier


# =====================================
# LOAD PRECOMPUTED GLCM FEATURES
# =====================================
def load_glcm_features(feature_dir):

    with open(os.path.join(feature_dir, "train_glcm_features.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)

    with open(os.path.join(feature_dir, "valid_glcm_features.pkl"), "rb") as f:
        X_valid, y_valid = pickle.load(f)

    with open(os.path.join(feature_dir, "test_glcm_features.pkl"), "rb") as f:
        X_test, y_test = pickle.load(f)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# =====================================
# MAIN PIPELINE
# =====================================
def run_glcm_svm_pipeline(feature_dir, model_dir):

    print("Loading GLCM features...")

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_glcm_features(feature_dir)

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # =========================
    # NORMALIZATION
    # =========================
    print("\nApplying normalization...")

    norm = FeatureNormalizer(
        lbp_dim=0,
        glcm_dim=X_train.shape[1],
        use_log_fft=False,
        use_energy_norm=False,
        glcm_weight=1.0
    )

    norm.fit(X_train)

    X_train = norm.transform(X_train)
    X_valid = norm.transform(X_valid)
    X_test  = norm.transform(X_test)

    os.makedirs(model_dir, exist_ok=True)
    norm.save(os.path.join(model_dir, "normalizer_glcm.pkl"))

    print("Normalization complete")

    # =========================
    # HYPERPARAMETER TUNING
    # =========================
    print("\nStarting RBF Approx tuning...\n")

    gamma_values = [0.02, 0.01, 0.005]

    best_acc = 0
    best_gamma = None

    for gamma in gamma_values:
        print(f"Testing gamma={gamma}")

        svm = SVMClassifier(
            model_type="rbf_approx",
            gamma=gamma,
            rbf_components=4000,
            use_pca=True,
            pca_components=100,
            class_weight="balanced"
        )

        svm.train(X_train, y_train)
        acc = svm.evaluate(X_valid, y_valid)

        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma

    print(f"\nBest gamma: {best_gamma}")
    print(f"Validation Accuracy: {best_acc:.4f}")

    # =========================
    # FINAL TRAINING
    # =========================
    print("\nTraining final model...")

    X_full = np.vstack((X_train, X_valid))
    y_full = np.hstack((y_train, y_valid))

    svm = SVMClassifier(
        model_type="rbf_approx",
        gamma=best_gamma,
        rbf_components=4000,
        use_pca=True,
        pca_components=100,
        class_weight="balanced"
    )

    svm.train(X_full, y_full)

    # =========================
    # FINAL TEST
    # =========================
    print("\nFinal Test Performance:")
    svm.evaluate(X_test, y_test)

    # =========================
    # SAVE MODEL
    # =========================
    svm.save(os.path.join(model_dir, "svm_glcm.pkl"))

    print("\nGLCM pipeline complete!")

    return {
        "gamma": best_gamma,
        "val_accuracy": best_acc
    }