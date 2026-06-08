"""Small pure-NumPy ridge estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeEffectEstimator:
    alpha: float = 1.0
    intercept_: float = 0.0
    coef_: np.ndarray | None = None
    x_mean_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeEffectEstimator":
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if len(y) != x.shape[0]:
            raise ValueError("y length must match x rows")
        self.x_mean_ = x.mean(axis=0)
        x_centered = x - self.x_mean_
        self.intercept_ = float(y.mean())
        y_centered = y - self.intercept_
        xtx = x_centered.T @ x_centered
        penalty = self.alpha * np.eye(x.shape[1])
        self.coef_ = np.linalg.pinv(xtx + penalty) @ x_centered.T @ y_centered
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.x_mean_ is None:
            raise RuntimeError("Estimator is not fitted")
        return self.intercept_ + (x - self.x_mean_) @ self.coef_

    def coefficients(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Estimator is not fitted")
        return self.coef_.copy()
