"""Chargement, nettoyage et encodage communs aux scripts du projet."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_and_clean_data(file_path=DATA_PATH):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
    )
    y = df.pop("Churn").map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("La colonne Churn contient des valeurs autres que Yes/No.")

    # Séparer avant de calculer la médiane évite d'utiliser les données de test.
    raw_train, raw_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    median_total = raw_train["TotalCharges"].median()
    raw_train["TotalCharges"] = raw_train["TotalCharges"].fillna(median_total)
    raw_test["TotalCharges"] = raw_test["TotalCharges"].fillna(median_total)

    # Un encodage commun garde le même schéma de colonnes dans train et test.
    combined = pd.get_dummies(pd.concat([raw_train, raw_test]), drop_first=True).fillna(0)
    X_train = combined.iloc[: len(raw_train)].copy()
    X_test = combined.iloc[len(raw_train) :].copy()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_clean_data()
    print(f"Données prêtes : {len(X_train)} lignes d'entraînement, {len(X_test)} de test.")
