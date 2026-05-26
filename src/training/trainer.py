import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_and_save(X, y, model, name):

    if len(X) == 0:
        print(f"❌ Skipping {name} (no data)")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs("src/models", exist_ok=True)

    with open(f"src/models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"💾 Saved: src/models/{name}.pkl")