"""Entraînement minimal : à faire évoluer pendant la formation."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age"]


def main() -> None:
    df = pd.read_csv("data/raw/diabetes.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["outcome"], test_size=0.2, random_state=42, stratify=df["outcome"]
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ])
    pipe.fit(x_train, y_train)
    print(f"accuracy test : {pipe.score(x_test, y_test):.3f}")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/model.joblib")


if __name__ == "__main__":
    main()