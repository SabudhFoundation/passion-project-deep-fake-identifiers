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
# Define base directory (current script location)
BASE_DIR = os.path.dirname(__file__)

# Folder containing extracted FFT features
FEATURE_DIR = os.path.join(BASE_DIR, "features")

# Folder to save trained models
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================
# 1. LOAD FEATURES
# =====================================
print("📂 Loading features...")

# Load training features and labels
with open(os.path.join(FEATURE_DIR, "train_fft_features.pkl"), "rb") as f:
    X_train, y_train = pickle.load(f)

# Load validation features and labels
with open(os.path.join(FEATURE_DIR, "valid_fft_features.pkl"), "rb") as f:
    X_valid, y_valid = pickle.load(f)

# Load test features and labels
with open(os.path.join(FEATURE_DIR, "test_fft_features.pkl"), "rb") as f:
    X_test, y_test = pickle.load(f)

# Print dataset shapes for verification
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")


# =====================================
# 2. NORMALIZATION (FFT-SPECIFIC)
# =====================================
print("\n⚙️ Applying preprocessing (energy + log + standard)...")

# Initialize normalizer
norm = FeatureNormalizer(
    method="standard",     # Standard scaling (best for SVM)
    use_log=True,          # Log transform for skewed FFT values
    use_energy_norm=True   # Normalize per sample (important for FFT)
)

# Fit on training data and transform all splits
X_train = norm.fit_transform(X_train)
X_valid = norm.transform(X_valid)
X_test  = norm.transform(X_test)

# Save normalizer for inference consistency
norm.save(os.path.join(FEATURE_DIR, "normalizer.pkl"))

print("✅ Preprocessing complete")


# =====================================
# 3. HYPERPARAMETER TUNING
# =====================================
print("\n🚀 Starting RBF Approx tuning...\n")

# Candidate gamma values (kernel width)
gamma_values = [0.02, 0.01, 0.005]

best_acc = 0
best_gamma = None

# Iterate over all gamma values
for i, gamma in enumerate(gamma_values, 1):

    print(f"\n🔍 [{i}/{len(gamma_values)}] Testing gamma={gamma}")

    start_time = time.time()

    # Initialize SVM with RBF approximation
    svm = SVMClassifier(
        model_type="rbf_approx",   # Faster than exact RBF kernel
        gamma=gamma,               # Kernel parameter
        rbf_components=4000,       # Number of random features (higher = better approx)
        use_pca=True,              # Reduce dimensionality
        pca_components=100,        # Keep top 100 components
        class_weight="balanced"    # Handle class imbalance
    )

    # Train model
    svm.train(X_train, y_train)

    # Evaluate on validation set
    acc = svm.evaluate(X_valid, y_valid)

    print(f"⏱ Time: {time.time() - start_time:.2f} sec")

    # Track best performing gamma
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

# Combine training + validation data for final model
X_full = np.vstack((X_train, X_valid))
y_full = np.hstack((y_train, y_valid))

start_time = time.time()

# Train final model using best hyperparameter
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

# Evaluate final model on unseen test data
svm.evaluate(X_test, y_test)


# =====================================
# 7. SAVE MODEL + PARAMS
# =====================================

# Save trained SVM model
svm.save(os.path.join(MODEL_DIR, "svm_fft.pkl"))

# Save best hyperparameters
with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
    json.dump({"gamma": best_gamma}, f)

# Save label mapping (for inference readability)
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({0: "real", 1: "fake"}, f)

print("\n🎉 Pipeline complete!")