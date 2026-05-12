from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "heart_model.joblib"

NOTEBOOK_METRICS = {
    "source": "Model Training.ipynb",
    "task": "Binary classification: disease vs no disease",
    "dataset": "Imputed Ill Conditions Dataset",
    "model": "KNN tuned",
    "accuracy": 0.84,
    "recall_disease": 0.86,
    "precision_disease": 0.83,
    "f1_disease": 0.85,
    "best_k": 13,
    "support": 767,
}

EDA_SUMMARY = {
    "source": "EDA .ipynb",
    "dataset": "heart_disease_uci.csv",
    "original_shape": {"rows": 920, "columns": 16},
    "target": "num: 0 = no disease; 1-4 = disease stages",
    "main_steps": [
        "Initial inspection of clinical variables and target distribution.",
        "Physiological cleaning: trestbps == 0 and chol == 0 treated as missing values.",
        "Rows with 4 or more missing values removed because they add little signal and many gaps.",
        "Low-missingness numeric fields such as trestbps and oldpeak imputed with the mean.",
        "fbs imputed by preserving the observed frequency distribution.",
        "chol imputed with the mean after predictive imputers failed to outperform the baseline.",
        "ca dropped in the imputed ill-conditions dataset because missingness was too high.",
        "slope collapsed into slope_bin: flat vs non-flat, then predictively imputed.",
        "thal collapsed into thal_bin: defect vs no defect, then predictively imputed.",
        "Final processed datasets exported for model comparison.",
    ],
    "processed_datasets": [
        "modified_dataset_without_imputations.csv",
        "modified_dataset_with_imputations.csv",
        "modified_dataset_with_imputations_no_ill_conditions.csv",
    ],
}

TRAINING_SUMMARY = {
    "source": "Model Training.ipynb",
    "selected_demo_model": "KNN tuned on Imputed Ill Conditions Dataset",
    "selection_reason": (
        "For the binary disease/no disease task, this combination achieved the best "
        "balanced practical result in the notebook: accuracy around 0.84 and disease "
        "recall around 0.86."
    ),
    "models_compared": [
        "Dummy Classifier",
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "Naive Bayes",
        "SVM RBF",
        "XGBoost",
    ],
    "binary_results_highlights": [
        "No NaNs dataset: Logistic Regression reached accuracy around 0.84 and disease recall around 0.80.",
        "Imputed Ill Conditions dataset: KNN reached accuracy around 0.84 and disease recall around 0.86.",
        "Imputed No Ill Conditions dataset: KNN reached accuracy around 0.83 and disease recall around 0.84.",
    ],
    "multiclass_note": (
        "Direct prediction of classes 0-4 performed poorly on the original distributions "
        "because stages 2-4, especially stage 4, are underrepresented. Balanced experiments "
        "improved metrics but are treated as experimental in this demo."
    ),
}

DISCLAIMER = (
    "This application does not provide medical diagnosis. It is an academic "
    "demonstration based on a machine-learning model trained on historical "
    "public data. Always consult a qualified healthcare professional."
)


def load_artifact() -> dict[str, Any] | None:
    if not ARTIFACT_PATH.exists():
        return None
    return joblib.load(ARTIFACT_PATH)
