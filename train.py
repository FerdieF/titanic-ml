# train.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from titanic_ml.pipeline import build_model, get_features_target


DATA_PATH = "data/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Split feature & target
    X, y = get_features_target(df)

    # 3. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Build model
    model = build_model(random_state=42)

    # 5. Train
    model.fit(X_train, y_train)

    # 6. Evaluate on validation set
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))

    # 7. Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
