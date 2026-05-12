from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backend.model_loader import NOTEBOOK_METRICS
from backend.preprocessing import DummiesAligner, HeartDiseaseCleaner

ORIGINAL_DATASET = ROOT_DIR / "heart_disease_uci.csv"
PREFERRED_DATASET = ROOT_DIR / "modified_dataset_with_imputations.csv"
ARTIFACT_PATH = ROOT_DIR / "backend" / "artifacts" / "heart_model.joblib"
METADATA_PATH = ROOT_DIR / "backend" / "artifacts" / "metadata.json"

TARGET = "num"
DROP_COLUMNS = ["id", TARGET]


def load_training_data() -> tuple[pd.DataFrame, pd.Series, str]:
    dataset_path = PREFERRED_DATASET if PREFERRED_DATASET.exists() else ORIGINAL_DATASET
    df = pd.read_csv(dataset_path)
    y = (df[TARGET] > 0).astype(int)
    X = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])
    return X, y, dataset_path.name


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("cleaner", HeartDiseaseCleaner()),
            ("encoder", DummiesAligner()),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(weights="distance")),
        ]
    )


def main() -> None:
    X, y, dataset_name = load_training_data()
    pipeline = build_pipeline(X)

    search = GridSearchCV(
        pipeline,
        {"classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]},
        cv=3,
        scoring="recall",
        n_jobs=1,
    )

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(search, X, y, cv=outer_cv, n_jobs=1)

    cv_metrics = {
        "dataset_used_for_artifact": dataset_name,
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "recall_disease": round(float(recall_score(y, y_pred)), 4),
        "classification_report": classification_report(
            y,
            y_pred,
            target_names=["no disease", "disease"],
            zero_division=0,
            output_dict=True,
        ),
    }

    search.fit(X, y)
    metadata = {
        "task": "Binary classification: disease vs no disease",
        "target_definition": "num > 0",
        "dataset_used_for_artifact": dataset_name,
        "best_params": search.best_params_,
        "cross_validation_metrics_from_current_artifact": cv_metrics,
        "reported_notebook_metrics": NOTEBOOK_METRICS,
        "feature_names": X.columns.tolist(),
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": search.best_estimator_, "metadata": metadata}, ARTIFACT_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model artifact to {ARTIFACT_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(json.dumps(cv_metrics, indent=2))


if __name__ == "__main__":
    main()
