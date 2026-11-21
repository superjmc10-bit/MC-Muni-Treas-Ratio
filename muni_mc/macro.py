"""Macro block: recession regime, inflation, unemployment, Fed, MOVE, HY, Treasuries.

Nothing in here depends on the muni market, so the whole block is simulated
first and handed to the muni block (``model.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import tenor_values
from .curves import (
    forward_curve,
    interp_curve,
    nearest_correlation,
    ns_loadings,
    passthrough_weights,
)

DT = 1.0 / 12.0
MACRO_SHOCKS = [
    "inflation",
    "unemployment",
    "move",
    "hy",
    "level",
    "slope",
    "curvature",
    "term_premium",
]


def step_dates(as_of: str, horizon_months: int) -> list[pd.Timestamp]:
    """Monthly time grid starting at the as-of date."""
    start = pd.Timestamp(as_of)
    return [start + pd.DateOffset(months=i) for i in range(horizon_months + 1)]


def _split_pair(key: str) -> tuple[str, str]:
    parts = str(key).split("_")
    for i in range(1, len(parts)):
        a, b = "_".join(parts[:i]), "_".join(parts[i:])
        if a in MACRO_SHOCKS and b in MACRO_SHOCKS:
            return a, b
    raise ValueError(
        f"Correlation key '{key}' must be <name>_<name> with names from {MACRO_SHOCKS}"
    )


def build_shock_correlation(cfg: dict) -> np.ndarray:
    n = len(MACRO_SHOCKS)
    corr = np.eye(n)
    pairs = dict(cfg["macro"].get("shock_correlations", {}) or {})
    pairs.update(cfg["treasury"].get("factor_corr", {}) or {})
    for key, val in pairs.items():
        a, b = _split_pair(key)
        i, j = MACRO_SHOCKS.index(a), MACRO_SHOCKS.index(b)
        if i == j:
            continue
        v = float(val)
        if not -1.0 < v < 1.0:
            raise ValueError(f"Correlation {key}={v} must lie strictly between -1 and 1")
        corr[i, j] = corr[j, i] = v
    return nearest_correlation(corr)


def market_ff_path(cfg: dict, dates: list[pd.Timestamp]) -> np.ndarray:
    """Expected fed-funds mid-rate at each step (step function between meetings)."""
    fed = cfg["macro"]["fed"]
    start = float(fed["start_mid"])
    shift = float(fed.get("market_path_shift_bp", 0.0)) / 100.0
    path = sorted(
        (pd.Period(str(k), freq="M"), float(v)) for k, v in (fed.get("market_path") or {}).items()
    )
    out = np.empty(len(dates))
    out[0] = start
    for t in range(1, len(dates)):
        p = pd.Period(dates[t], freq="M")
        val = start
        for per, v in path:
            if per <= p:
                val = v
        out[t] = val + shift
    return out


def treasury_base_path(cfg: dict, tenors: np.ndarray, horizon_months: int) -> np.ndarray:
    """Expected Treasury curve at each step, shape (H+1, K)."""
    tr = cfg["treasury"]
    y0 = interp_curve(tr["curve_start"], tenors)
    mode = tr.get("path_mode", "unchanged")
    if mode == "unchanged":
        y_h = y0
    elif mode == "forwards":
        y_h = forward_curve(tr["curve_start"], tenors, horizon_months / 12.0)
    elif mode == "shifts":
        y_h = y0 + tenor_values(tr["shifts_bp"], tenors, "treasury.shifts_bp") / 100.0
    elif mode == "custom":
        y_h = interp_curve(tr["curve_target"], tenors)
    else:  # pragma: no cover - validated earlier
        raise ValueError(mode)
    frac = np.arange(horizon_months + 1, dtype=float) / horizon_months
    return y0[None, :] + frac[:, None] * (y_h - y0)[None, :]


def _ou_baseline(start: float, target: float, kappa: float, steps: int) -> np.ndarray:
    decay = (1.0 - kappa * DT) ** np.arange(steps + 1)
    return target + (start - target) * decay


def log_ou_baseline(start: float, target: float, kappa: float, steps: int) -> np.ndarray:
    """Shock-free path of a log mean-reverting variable (MOVE, HY OAS)."""
    return np.exp(_ou_baseline(np.log(start), np.log(target), kappa, steps))


def simulate_macro(
    cfg: dict, rng: np.random.Generator, tenors: np.ndarray, dates: list[pd.Timestamp]
) -> dict[str, np.ndarray]:
    n = int(cfg["simulation"]["n_paths"])
    h = len(dates) - 1
    k = len(tenors)
    sq = np.sqrt(DT)
    m, tr = cfg["macro"], cfg["treasury"]
    inf_c, un_c, rec_c, fed = m["inflation"], m["unemployment"], m["recession"], m["fed"]
    mv_c, hy_c, tp_c = m["rate_vol"], m["hy_oas"], tr["term_premium"]

    chol = np.linalg.cholesky(build_shock_correlation(cfg))
    eps = rng.standard_normal((h, n, len(MACRO_SHOCKS))) @ chol.T
    ix = {name: i for i, name in enumerate(MACRO_SHOCKS)}

    infl = np.empty((n, h + 1))
    unemp = np.empty((n, h + 1))
    rec = np.zeros((n, h + 1), dtype=np.int8)
    rec_entry = np.zeros((n, h + 1), dtype=bool)
    ff = np.empty((n, h + 1))
    move = np.empty((n, h + 1))
    hy = np.empty((n, h + 1))
    tp = np.empty((n, h + 1))
    treasury = np.empty((n, h + 1, k))

    infl[:, 0] = inf_c["start"]
    unemp[:, 0] = un_c["start"]
    rec[:, 0] = 1 if rec_c.get("start_in_recession") else 0
    ff[:, 0] = fed["start_mid"]
    move[:, 0] = mv_c["start"]
    hy[:, 0] = hy_c["start"]

    infl_base = _ou_baseline(inf_c["start"], inf_c["target_12m"], inf_c["mean_reversion"], h)
    unemp_base = _ou_baseline(un_c["start"], un_c["target_12m"], un_c["mean_reversion"], h)
    ff_mkt = market_ff_path(cfg, dates)
    base_curve = treasury_base_path(cfg, tenors, h)
    load = ns_loadings(tenors, float(tr["ns_lambda"]))
    w_ff = passthrough_weights(tenors, float(tr["ff_passthrough_tau"]))
    tp_base = np.linspace(0.0, float(tp_c.get("change_bp", 0.0)) / 100.0, h + 1)

    fv = tr["factor_vol_bp"]
    factor_vol = np.array([fv["level"], fv["slope"], fv["curvature"]], dtype=float) / 100.0
    kz = float(tr["factor_mean_reversion"])
    k_tp, s_tp = float(tp_c["mean_reversion"]), float(tp_c["vol_bp"]) / 100.0

    p12 = float(rec_c["prob_12m"])
    hazard = 1.0 - (1.0 - p12) ** (1.0 / 12.0)
    p_exit = 1.0 / max(float(rec_c["avg_duration_months"]), 1.0)
    hy_base = log_ou_baseline(hy_c["start"], hy_c["long_run"], hy_c["mean_reversion"], h)
    meeting_months = {int(x) for x in fed["meeting_months"]}
    cut_speed = float(fed["recession_cut_per_month_bp"]) / 100.0
    max_cut = float(fed["recession_max_cut_bp"]) / 100.0
    recovery = float(fed.get("recession_cut_recovery", 0.05))
    max_move = float(fed["max_move_bp"]) / 100.0

    z = np.zeros((n, 3))
    tp_dev = np.zeros(n)
    rec_cut = np.zeros(n)
    pi_dev_prev = np.zeros(n)
    treasury[:, 0, :] = base_curve[0]

    for t in range(h):
        e = eps[t]
        # --- recession regime (hazard tilts with policy tightness and credit stress)
        tilt = np.exp(
            rec_c["ff_sensitivity"] * (ff[:, t] - ff_mkt[t])
            + rec_c["hy_sensitivity"] * (hy[:, t] - hy_base[t]) / 100.0
        )
        p_enter = np.clip(hazard * tilt, 0.0, 0.9)
        draw = rng.random(n)
        prev = rec[:, t]
        enter = (prev == 0) & (draw < p_enter)
        leave = (prev == 1) & (draw < p_exit)
        now = np.where(enter, 1, np.where(leave, 0, prev)).astype(np.int8)
        rec[:, t + 1] = now
        rec_entry[:, t + 1] = enter

        # --- inflation and unemployment (mean-reverting, recession drift)
        infl[:, t + 1] = (
            infl[:, t]
            + inf_c["mean_reversion"] * (inf_c["target_12m"] - infl[:, t]) * DT
            + inf_c["recession_drift"] * now * DT
            + inf_c["vol"] * sq * e[:, ix["inflation"]]
        )
        unemp[:, t + 1] = (
            unemp[:, t]
            + un_c["mean_reversion"] * (un_c["target_12m"] - unemp[:, t]) * DT
            + un_c["recession_drift"] * now * DT
            + un_c["vol"] * sq * e[:, ix["unemployment"]]
        )
        pi_dev = infl[:, t + 1] - infl_base[t + 1]
        u_dev = unemp[:, t + 1] - unemp_base[t + 1]

        # --- Fed: market path + reaction to surprises, 25bp steps at meetings
        rec_cut = np.where(
            now == 1, np.minimum(rec_cut + cut_speed, max_cut), rec_cut * (1.0 - recovery)
        )
        desired = (
            ff_mkt[t + 1]
            + fed["inflation_reaction"] * pi_dev
            - fed["unemployment_reaction"] * u_dev
            - rec_cut
        )
        cap = np.where(now == 1, 1.0, max_move)
        gap = np.clip(desired - ff[:, t], -cap, cap)
        q = gap / 0.25
        lo = np.floor(q)
        step = (lo + (rng.random(n) < (q - lo))) * 0.25  # unbiased stochastic rounding
        meeting = dates[t + 1].month in meeting_months
        emergency = (now == 1) & (gap <= -0.5)
        act = emergency | meeting
        ff[:, t + 1] = np.where(act, np.maximum(ff[:, t] + step, fed["floor"]), ff[:, t])

        # --- rate volatility (MOVE) and risk appetite (HY OAS), log mean-reverting
        lm = np.log(move[:, t])
        move[:, t + 1] = np.exp(
            lm
            + mv_c["mean_reversion"] * (np.log(mv_c["long_run"]) - lm) * DT
            + mv_c["vol"] * sq * e[:, ix["move"]]
            + mv_c["recession_jump"] * enter
        )
        lh = np.log(hy[:, t])
        target = np.where(now == 1, np.log(hy_c["recession_level"]), np.log(hy_c["long_run"]))
        hy[:, t + 1] = np.exp(
            lh + hy_c["mean_reversion"] * (target - lh) * DT + hy_c["vol"] * sq * e[:, ix["hy"]]
        )

        # --- Treasury curve: expected path + NS shocks + Fed surprise + term premium
        vol_scale = (move[:, t] / mv_c["reference"])[:, None]
        shocks = e[:, [ix["level"], ix["slope"], ix["curvature"]]]
        z = z - kz * z * DT + vol_scale * factor_vol * sq * shocks
        z[:, 0] += tr["inflation_to_level"] * (pi_dev - pi_dev_prev)
        z[:, 0] += tr["recession_level_shock_bp"] / 100.0 * enter
        pi_dev_prev = pi_dev
        tp_dev = tp_dev - k_tp * tp_dev * DT + s_tp * vol_scale[:, 0] * sq * e[:, ix["term_premium"]]
        tp[:, t + 1] = tp_base[t + 1] + tp_dev
        ff_dev = ff[:, t + 1] - ff_mkt[t + 1]
        treasury[:, t + 1, :] = np.maximum(
            base_curve[t + 1][None, :]
            + z @ load.T
            + ff_dev[:, None] * w_ff[None, :]
            + tp[:, t + 1][:, None] * (1.0 - w_ff)[None, :],
            0.05,
        )
    tp[:, 0] = 0.0

    # Centre the simulated curve on the expected path so the user's curve view
    # is the mean (or median) outcome; macro shocks then only shape the spread.
    center = str(tr.get("center", "mean")).lower()
    if center in {"mean", "median"}:
        stat = treasury.mean(axis=0) if center == "mean" else np.median(treasury, axis=0)
        treasury = np.maximum(treasury - (stat - base_curve)[None, :, :], 0.05)
    elif center != "none":
        raise ValueError("treasury.center must be mean, median or none")

    return {
        "inflation": infl,
        "unemployment": unemp,
        "recession": rec,
        "recession_entry": rec_entry,
        "fed_funds": ff,
        "fed_funds_market": ff_mkt,
        "move": move,
        "hy_oas": hy,
        "term_premium": tp,
        "treasury": treasury,
        "treasury_base": base_curve,
        "hy_base": hy_base,
        "move_base": log_ou_baseline(mv_c["start"], mv_c["long_run"], mv_c["mean_reversion"], h),
    }
