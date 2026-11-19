"""Yield-curve utilities: interpolation, Nelson-Siegel loadings, forwards."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def curve_arrays(curve: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (maturities, yields) arrays from a {maturity: yield} dict."""
    if not curve:
        raise ValueError("Curve is empty")
    mats = np.array(sorted(float(k) for k in curve), dtype=float)
    ylds = np.array([float(curve[m]) for m in mats], dtype=float)
    return mats, ylds


def interp_curve(curve: Mapping, tenors) -> np.ndarray:
    """Linear interpolation in maturity, flat beyond the ends."""
    mats, ylds = curve_arrays(curve)
    return np.interp(np.asarray(tenors, dtype=float), mats, ylds)


def ns_loadings(tenors, lam: float) -> np.ndarray:
    """Nelson-Siegel factor loadings, shape (len(tenors), 3): level, slope, curvature."""
    tau = np.asarray(tenors, dtype=float)
    x = lam * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(x > 1e-10, (1.0 - np.exp(-x)) / x, 1.0)
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(tau), slope, curv])


def fit_nelson_siegel(curve: Mapping, lam: float) -> np.ndarray:
    """Least-squares Nelson-Siegel factors (level, slope, curvature) for a curve."""
    mats, ylds = curve_arrays(curve)
    if mats.size < 3:
        raise ValueError("Need at least 3 maturities to fit Nelson-Siegel")
    beta, *_ = np.linalg.lstsq(ns_loadings(mats, lam), ylds, rcond=None)
    return beta


def forward_curve(curve: Mapping, tenors, horizon_years: float) -> np.ndarray:
    """Approximate h-year-forward yields for each tenor.

    Treats par yields as continuously compounded zero rates, which is close
    enough for a curve-path anchor:  F(t) = [(t+h) y(t+h) - h y(h)] / t.
    """
    tenors = np.asarray(tenors, dtype=float)
    h = float(horizon_years)
    y_h = interp_curve(curve, [h])[0]
    y_th = interp_curve(curve, tenors + h)
    return ((tenors + h) * y_th - h * y_h) / tenors


def passthrough_weights(tenors, tau: float) -> np.ndarray:
    """How much of a persistent fed-funds surprise reaches each tenor (1 at 0, -> 0 long)."""
    x = np.asarray(tenors, dtype=float) / float(tau)
    return np.where(x > 1e-10, (1.0 - np.exp(-x)) / np.maximum(x, 1e-10), 1.0)


def interp_weights(tenors, target: float) -> np.ndarray:
    """Weights w such that values @ w linearly interpolates the tenor grid at ``target``."""
    tenors = np.asarray(tenors, dtype=float)
    w = np.zeros_like(tenors)
    if target <= tenors[0]:
        w[0] = 1.0
    elif target >= tenors[-1]:
        w[-1] = 1.0
    else:
        j = int(np.searchsorted(tenors, target))
        lo, hi = tenors[j - 1], tenors[j]
        a = (target - lo) / (hi - lo)
        w[j - 1], w[j] = 1.0 - a, a
    return w


def nearest_correlation(c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues so a user-edited correlation matrix stays valid."""
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    if vals.min() >= eps:
        return c
    vals = np.clip(vals, eps, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed
