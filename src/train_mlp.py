import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier


# ==========================================
# LOAD FEATURES (STREAM FORMAT)
# ==========================================
def load_features(file_path):
    X, y = [], []

    with open(file_path, "rb") as f:
        try:
            while True:
                feat, label = pickle.load(f)
                X.append(feat)
                y.append(label)
        except EOFError:
            pass

    # 🔥 MEMORY OPTIMIZATION
    return np.array(X, dtype=np.float32), np.array(y)


# ==========================================
# TRAIN + EVALUATE + SAVE
# ==========================================
def train_model(X, y, name):

    print(f"\n🚀 Training {name}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 🔥 FEATURE SCALING (CRITICAL)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🔥 OPTIMIZED MLP FOR LARGE DATA
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        batch_size=256,          # 🔥 faster for large data
        max_iter=80,             # 🔥 reduced training time
        early_stopping=True,
        verbose=True,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 {name} Results")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model + scaler
    save_path = f"src/models/{name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler
        }, f)

    print(f"💾 Saved: {save_path}")


# ==========================================
# MAIN
# ==========================================
def main():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    FEATURE_DIR = os.path.join(
        BASE_DIR,
        "src",
        "utils",
        "combined_features"
    )

    print("📂 Loading features...")

    X_train, y_train = load_features(os.path.join(FEATURE_DIR, "train_features.pkl"))
    X_valid, y_valid = load_features(os.path.join(FEATURE_DIR, "valid_features.pkl"))

    # Combine datasets
    X = np.vstack((X_train, X_valid))
    y = np.hstack((y_train, y_valid))

    print("Final dataset shape:", X.shape)

    # ==========================================
    # 🔥 FEATURE SPLITTING
    # ==========================================
    # Adjust if needed based on your feature design
    glcm_dim = 24
    lbp_dim = 128
    fft_dim = X.shape[1] - (glcm_dim + lbp_dim)

    glcm = X[:, :glcm_dim]
    lbp = X[:, glcm_dim:glcm_dim + lbp_dim]
    fft = X[:, glcm_dim + lbp_dim:]

    print("\nFeature Shapes:")
    print("GLCM:", glcm.shape)
    print("LBP:", lbp.shape)
    print("FFT:", fft.shape)

    # ==========================================
    # TRAIN INDIVIDUAL MODELS
    # ==========================================
    train_model(glcm, y, "glcm_mlp")
    train_model(lbp, y, "lbp_mlp")
    train_model(fft, y, "fft_mlp")

    # ==========================================
    # TRAIN COMBINED MODEL
    # ==========================================
    train_model(X, y, "combined_mlp")

    print("\n✅ ALL MODELS TRAINED & SAVED!")


# ==========================================
if __name__ == "__main__":
    main()