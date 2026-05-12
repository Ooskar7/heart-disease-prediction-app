from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.model_loader import (
    DISCLAIMER,
    EDA_SUMMARY,
    NOTEBOOK_METRICS,
    TRAINING_SUMMARY,
    load_artifact,
)
from backend.schemas import PredictionInput, PredictionResponse


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="Heart Disease ML Demo",
    description="Academic demo for binary heart-disease prediction.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    artifact = load_artifact()
    return {"status": "ok", "model_loaded": artifact is not None}


@app.get("/metadata")
def metadata():
    artifact = load_artifact()
    return {
        "disclaimer": DISCLAIMER,
        "reported_notebook_metrics": NOTEBOOK_METRICS,
        "eda_summary": EDA_SUMMARY,
        "training_summary": TRAINING_SUMMARY,
        "artifact_available": artifact is not None,
        "artifact_metadata": None if artifact is None else artifact.get("metadata", {}),
        "input_features": list(PredictionInput.model_fields.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    artifact = load_artifact()
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifact is not trained yet. Run scripts/train_model.py after "
                "installing dependencies to generate backend/artifacts/heart_model.joblib."
            ),
        )

    model = artifact["model"]
    raw_input = payload.model_dump()
    raw_input["slope_bin"] = "flat" if raw_input.get("slope") == "flat" else "non-flat"
    raw_input["thal_bin"] = "no defect" if raw_input.get("thal") == "normal" else "defect"
    row = pd.DataFrame([raw_input])
    prediction = int(model.predict(row)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])

    label = "disease" if prediction == 1 else "no disease"
    risk_text = (
        "The model estimates possible presence of heart disease."
        if prediction == 1
        else "The model estimates no heart disease."
    )

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability_disease=probability,
        risk_text=risk_text,
        disclaimer=DISCLAIMER,
    )
