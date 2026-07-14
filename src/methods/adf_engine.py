"""
Fast ADF regression engine (numpy backend).

Used by SADF. Returns the coefficient and t-statistic from
Δy_t = α + δ·y_{t-1} + Σ φ_j Δy_{t-j} + ε_t.
"""
from __future__ import annotations
import numpy as np


def _build_regressor(dy: np.ndarray, ylag: np.ndarray, L: int) -> np.ndarray | None:
    """Design matrix [1, y_{t-1}, Δy_{t-1}, …, Δy_{t-L}] for the ADF regression."""
    n = len(dy)
    if L >= n - 2:
        return None
    rows = n - L
    X = np.empty((rows, 2 + L))
    X[:, 0] = 1.0
    X[:, 1] = ylag[L:]
    for j in range(1, L + 1):
        X[:, 1 + j] = dy[L - j: n - j]
    return X


def _ols_fast(X: np.ndarray, y: np.ndarray) -> dict | None:
    """OLS via lstsq, plus the standard errors / t-stats / log-likelihood
    needed for coefficient inference and IC-based lag selection."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    if dof <= 0:
        return None
    sigma2 = (resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    se     = np.sqrt(np.diag(cov))
    t_stat = beta / np.where(se > 0, se, np.nan)
    ll     = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return {"beta": beta, "t_stat": t_stat, "se": se, "sigma2": sigma2, "ll": ll}


def adf_regression(y, max_lags: int = 0, lag_method: str = "fixed") -> dict | None:
    """ADF regression with IC-based or fixed lag selection."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 10:
        return None
    dy   = np.diff(y)
    ylag = y[:-1]

    if lag_method == "fixed":
        L = int(max_lags)
    else:
        best_ic = np.inf; L = 0
        for k in range(int(max_lags) + 1):
            X_try = _build_regressor(dy, ylag, k)
            if X_try is None: continue
            res = _ols_fast(X_try, dy[k:])
            if res is None: continue
            penalty = np.log(len(dy[k:])) if lag_method == "bic" else 2.0
            ic = -2.0 * res["ll"] + penalty * X_try.shape[1]
            if ic < best_ic:
                best_ic = ic; L = k

    X = _build_regressor(dy, ylag, L)
    if X is None:
        return None
    yreg = dy[L:]
    if X.shape[0] < X.shape[1] + 2:
        return None
    res = _ols_fast(X, yreg)
    if res is None:
        return None
    tau = X.shape[0]
    return {
        "delta_hat": res["beta"][1],
        "t_stat":    res["t_stat"][1],
        "coef_stat": tau * res["beta"][1],
        "n": tau, "lags": L,
    }
