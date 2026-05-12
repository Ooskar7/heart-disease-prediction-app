from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: Literal["Male", "Female"]
    dataset: Literal["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    trestbps: float = Field(ge=0, le=300)
    chol: float = Field(ge=0, le=700)
    fbs: bool
    restecg: Literal["normal", "st-t abnormality", "lv hypertrophy"]
    thalch: float = Field(ge=0, le=250)
    exang: bool
    oldpeak: float = Field(ge=-10, le=10)
    slope: Literal["upsloping", "flat", "downsloping"]
    ca: float | None = Field(default=None, ge=0, le=4)
    thal: Literal["normal", "fixed defect", "reversable defect"] | None = None


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_disease: float | None
    risk_text: str
    disclaimer: str
