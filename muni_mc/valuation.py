"""Cheap vs expensive: reference statistics, classification and summary tables.

Sign convention used everywhere: a HIGHER ratio means munis are CHEAP (they
yield more relative to Treasuries); a LOWER ratio means munis are EXPENSIVE.

Three lenses are reported:

* history   z = (ratio - historical mean) / historical sd
* model     z = (ratio - model fair value) / historical sd
* composite weighted blend of the two (weights in ``valuation``)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, tenor_values

if TYPE_CHECKING:  # pragma: no cover
    from .model import SimulationResult


# --------------------------------------------------------------------------- #
#  reference statistics
# --------------------------------------------------------------------------- #
def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


def stats_from_history(
    path: str, tenors: np.ndarray, lookback_years: float, as_of: str
) -> dict:
    """Mean / sd of ratios (and average UST yields) from a history CSV."""
    df = pd.read_csv(_resolve(path))
    if "date" not in df.columns:
        raise ValueError("history CSV needs a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    end = pd.Timestamp(as_of)
    start = end - pd.DateOffset(years=float(lookback_years))
    df = df[(df["date"] > start) & (df["date"] <= end)]
    if len(df) < 10:
        raise ValueError(f"history CSV has only {len(df)} rows inside the lookback window")

    def collect(prefix: str) -> tuple[np.ndarray, list[str]]:
        cols, mats = [], []
        for c in df.columns:
            mt = re.fullmatch(rf"{prefix}_(\d+(?:\.\d+)?)", str(c))
            if mt:
                cols.append(c)
                mats.append(float(mt.group(1)))
        order = np.argsort(mats)
        return np.array(mats)[order], [cols[i] for i in order]

    r_mats, r_cols = collect("ratio")
    if not r_cols:
        raise ValueError("history CSV needs columns named ratio_<tenor>, e.g. ratio_10")
    ratios = df[r_cols].astype(float)
    if np.nanmedian(ratios.values) > 2.0:  # given in percent
        ratios = ratios / 100.0
    mean = np.interp(tenors, r_mats, ratios.mean().values)
    sd = np.interp(tenors, r_mats, ratios.std(ddof=1).values)

    avg_t = None
    u_mats, u_cols = collect("ust")
    if u_cols:
        avg_t = np.interp(tenors, u_mats, df[u_cols].astype(float).mean().values)
    window = f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}"
    return {"mean": mean, "sd": sd, "avg_treasury": avg_t, "n_obs": len(df), "window": window}


def reference_stats(cfg: dict, tenors: np.ndarray) -> dict:
    val = cfg["valuation"]
    ref = val["reference"]
    out = {
        "mean": tenor_values(ref["mean"], tenors, "valuation.reference.mean"),
        "sd": tenor_values(ref["sd"], tenors, "valuation.reference.sd"),
        "avg_treasury": tenor_values(ref["avg_treasury"], tenors, "valuation.reference.avg_treasury"),
        "source": "config (valuation.reference)",
    }
    if val.get("history_csv"):
        h = stats_from_history(val["history_csv"], tenors, val["lookback_years"], cfg["meta"]["as_of"])
        out["mean"], out["sd"] = h["mean"], h["sd"]
        if h["avg_treasury"] is not None:
            out["avg_treasury"] = h["avg_treasury"]
        out["source"] = f"{val['history_csv']} ({h['n_obs']} obs, {h['window']})"
    return out


# --------------------------------------------------------------------------- #
#  classification
# --------------------------------------------------------------------------- #
def classify(z, bands) -> np.ndarray:
    """Zone index for each z (0 = most expensive ... len(bands) = cheapest)."""
    return np.digitize(np.asarray(z, dtype=float), np.asarray(bands, dtype=float))


def label_for(z: float, cfg: dict) -> str:
    return cfg["valuation"]["labels"][int(classify(z, cfg["valuation"]["bands_z"]))]


def z_scores(res: "SimulationResult", basis: str) -> np.ndarray:
    """(N, H+1, K) z-scores under 'history', 'model' or 'composite'."""
    mean, sd = res.reference["mean"], res.reference["sd"]
    z_hist = (res.ratio - mean) / sd
    if basis == "history":
        return z_hist
    z_model = (res.ratio - res.fair_value) / sd
    if basis == "model":
        return z_model
    if basis == "composite":
        v = res.cfg["valuation"]
        wh, wm = float(v["weight_history"]), float(v["weight_model"])
        return (wh * z_hist + wm * z_model) / (wh + wm)
    raise ValueError(f"basis must be history|model|composite, got {basis}")


# --------------------------------------------------------------------------- #
#  tables
# --------------------------------------------------------------------------- #
def _tenor_label(t: float) -> str:
    return f"{t:g}Y"


def valuation_today(res: "SimulationResult") -> pd.DataFrame:
    """Where each tenor sits today under every lens."""
    cfg = res.cfg
    rows = []
    zh = z_scores(res, "history")[0, 0]
    zm = z_scores(res, "model")[0, 0]
    zc = z_scores(res, "composite")[0, 0]
    for i, t in enumerate(res.tenors):
        rows.append(
            {
                "Tenor": _tenor_label(t),
                "UST %": res.treasury[0, 0, i],
                "MMD AAA %": res.muni[0, 0, i],
                "Ratio %": 100 * res.ratio[0, 0, i],
                "Hist mean %": 100 * res.reference["mean"][i],
                "Hist sd (pts)": 100 * res.reference["sd"][i],
                "z vs history": zh[i],
                "History view": label_for(zh[i], cfg),
                "Model fair value %": 100 * res.fair_value[0, 0, i],
                "Gap to FV (pts)": 100 * (res.ratio[0, 0, i] - res.fair_value[0, 0, i]),
                "z vs model": zm[i],
                "Model view": label_for(zm[i], cfg),
                "Composite z": zc[i],
                "Signal": label_for(zc[i], cfg),
            }
        )
    return pd.DataFrame(rows)


def forecast_summary(res: "SimulationResult") -> pd.DataFrame:
    """Ratio distribution by tenor at each report month."""
    rows = []
    for mth in res.report_months:
        r = res.ratio[:, mth, :]
        zc = z_scores(res, "composite")[:, mth, :]
        zone = classify(zc, res.cfg["valuation"]["bands_z"])
        mid = len(res.cfg["valuation"]["bands_z"]) // 2  # index of "Fair"
        for i, t in enumerate(res.tenors):
            q = np.percentile(r[:, i], [5, 25, 50, 75, 95])
            rows.append(
                {
                    "Month": mth,
                    "Date": res.dates[mth].strftime("%Y-%m-%d"),
                    "Tenor": _tenor_label(t),
                    "Start %": 100 * res.ratio[0, 0, i],
                    "Mean %": 100 * r[:, i].mean(),
                    "P5 %": 100 * q[0],
                    "P25 %": 100 * q[1],
                    "Median %": 100 * q[2],
                    "P75 %": 100 * q[3],
                    "P95 %": 100 * q[4],
                    "Exp. change (pts)": 100 * (r[:, i].mean() - res.ratio[0, 0, i]),
                    "P(ratio higher)": (r[:, i] > res.ratio[0, 0, i]).mean(),
                    "P(cheap or better)": (zone[:, i] > mid).mean(),
                    "P(expensive or worse)": (zone[:, i] < mid).mean(),
                    "Median FV %": 100 * np.median(res.fair_value[:, mth, i]),
                    "Median UST %": np.median(res.treasury[:, mth, i]),
                    "Median MMD %": np.median(res.muni[:, mth, i]),
                }
            )
    return pd.DataFrame(rows)


def zone_probabilities(res: "SimulationResult", month: int, basis: str = "composite") -> pd.DataFrame:
    labels = res.cfg["valuation"]["labels"]
    zone = classify(z_scores(res, basis)[:, month, :], res.cfg["valuation"]["bands_z"])
    data = {lab: (zone == j).mean(axis=0) for j, lab in enumerate(labels)}
    df = pd.DataFrame(data, index=[_tenor_label(t) for t in res.tenors])
    df.index.name = "Tenor"
    return df


def tax_breakeven_table(res: "SimulationResult") -> pd.DataFrame:
    """Who is muni ownership cheap for? Ratio vs (1 - bracket) today and at the horizon."""
    rows = []
    h = res.horizon
    for name, rate in res.cfg["tax"]["brackets"].items():
        be = 1.0 - float(rate)
        row = {"Bracket": name, "Breakeven ratio %": 100 * be}
        for i, t in enumerate(res.tenors):
            tl = _tenor_label(t)
            row[f"{tl} cushion today (pts)"] = 100 * (res.ratio[0, 0, i] - be)
            row[f"{tl} P(above BE @ {h}m)"] = (res.ratio[:, h, i] > be).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def attribution(res: "SimulationResult") -> pd.DataFrame:
    """Bridge from today's ratio to the expected horizon ratio, by driver.

    E[R_H] - R_0 = sum_j (E[FV_j,H] - FV_j,0) + (E[gap_H] - gap_0)
    where gap = R - FV. The identity is exact (expectations are additive).
    """
    from .model import FV_COMPONENTS

    h = res.horizon
    rows = []
    for i, t in enumerate(res.tenors):
        row = {"Tenor": _tenor_label(t), "Start ratio %": 100 * res.ratio[0, 0, i]}
        for key, label in FV_COMPONENTS:
            c = res.fv_components[key]
            row[label] = 100 * (c[h, i] - c[0, i])
        gap0 = res.ratio[0, 0, i] - res.fair_value[0, 0, i]
        gap_h = (res.ratio[:, h, i] - res.fair_value[:, h, i]).mean()
        row["Rich/cheap gap closing"] = 100 * (gap_h - gap0)
        row["Expected ratio %"] = 100 * res.ratio[:, h, i].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def fair_value_breakdown(res: "SimulationResult", month: int = 0) -> pd.DataFrame:
    """Level of each fair-value component (pts) at a given month (path mean)."""
    from .model import FV_COMPONENTS

    data = {
        label: 100 * res.fv_components[key][month] for key, label in FV_COMPONENTS
    }
    df = pd.DataFrame(data, index=[_tenor_label(t) for t in res.tenors])
    df["Fair value %"] = df.sum(axis=1)
    df.index.name = "Tenor"
    return df


def event_summary(res: "SimulationResult") -> pd.DataFrame:
    rows = [
        ("Recession at any point", res.events["recession"].mean()),
        ("Liquidity / outflow event", res.events["liquidity"].mean()),
        ("Top tax rate changed", res.events["tax_change"].mean()),
        ("Tax-exemption threat", res.events["exemption_threat"].mean()),
    ]
    return pd.DataFrame(rows, columns=["Event within horizon", "Probability"])
