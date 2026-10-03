"""Entraîne, compare sur validation, évalue sur test et sauvegarde les artefacts."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_processing import NUMERIC_COLUMNS, load_and_clean_data

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def evaluate(name, model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y, pred):.4f} | Precision: {precision_score(y, pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y, pred, zero_division=0):.4f} | F1: {f1_score(y, pred, zero_division=0):.4f} | ROC-AUC: {roc_auc_score(y, proba):.4f}")
    print("Matrice de confusion:\n", confusion_matrix(y, pred))
    return roc_auc_score(y, proba)


def main():
    X_train, X_test, y_train, y_test = load_and_clean_data()
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    scaler = StandardScaler()
    X_fit, X_val = X_fit.copy(), X_val.copy()
    X_fit[NUMERIC_COLUMNS] = scaler.fit_transform(X_fit[NUMERIC_COLUMNS])
    X_val[NUMERIC_COLUMNS] = scaler.transform(X_val[NUMERIC_COLUMNS])
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                                  eval_metric="logloss", n_jobs=1),
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(X_fit, y_fit)
        scores[name] = evaluate(f"Validation — {name}", model, X_val, y_val)

    selected_name = max(scores, key=scores.get)
    print(f"\nModèle retenu selon ROC-AUC validation : {selected_name}")
    # Réentraînement sur toutes les données d'entraînement, sans toucher au test.
    final_scaler = StandardScaler()
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[NUMERIC_COLUMNS] = final_scaler.fit_transform(X_train_scaled[NUMERIC_COLUMNS])
    X_test_scaled[NUMERIC_COLUMNS] = final_scaler.transform(X_test_scaled[NUMERIC_COLUMNS])
    model = candidates[selected_name].__class__(**candidates[selected_name].get_params())
    model.fit(X_train_scaled, y_train)
    evaluate(f"Test final — {selected_name}", model, X_test_scaled, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "best_model.joblib")
    joblib.dump(final_scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(list(X_train.columns), MODEL_DIR / "model_columns.joblib")
    joblib.dump(selected_name, MODEL_DIR / "model_name.joblib")
    print(f"Artefacts sauvegardés dans {MODEL_DIR}")


if __name__ == "__main__":
    main()
