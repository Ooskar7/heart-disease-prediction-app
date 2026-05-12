from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HeartDiseaseCleaner(BaseEstimator, TransformerMixin):
    """Apply project-specific clinical cleaning before sklearn preprocessing."""

    def __init__(self, zero_as_missing: Iterable[str] = ("trestbps", "chol")):
        self.zero_as_missing = tuple(zero_as_missing)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = X.copy()
        for column in self.zero_as_missing:
            if column in cleaned.columns:
                cleaned[column] = cleaned[column].replace(0, np.nan)
        return cleaned


class DummiesAligner(BaseEstimator, TransformerMixin):
    """Mirror notebook encoding with pandas.get_dummies(drop_first=True)."""

    def fit(self, X: pd.DataFrame, y=None):
        encoded = pd.get_dummies(X, drop_first=True)
        self.columns_ = encoded.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.get_dummies(X, drop_first=True)
        return encoded.reindex(columns=self.columns_, fill_value=0)
