"""Loading, layering and validating model configuration.

A run's configuration is built in three layers:

1. ``config/default.yaml`` (or a file passed with ``--config``)
2. zero or more scenario files from ``config/scenarios`` (deep-merged in order)
3. ``--set key.path=value`` overrides

Dicts whose keys are all numbers (curves, per-tenor parameters) are *replaced*
by a scenario rather than merged, so a scenario that supplies a curve supplies
the whole curve.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
SCENARIO_DIR = PROJECT_ROOT / "config" / "scenarios"


# --------------------------------------------------------------------------- #
#  key normalisation and merging
# --------------------------------------------------------------------------- #
def _as_number(key: Any) -> Any:
    """Return ``float(key)`` when the key is numeric (10, '10', '0.25'), else the key."""
    if isinstance(key, bool):
        return key
    if isinstance(key, (int, float)):
        return float(key)
    if isinstance(key, str) and re.fullmatch(r"[+-]?\d+(\.\d+)?", key.strip()):
        return float(key)
    return key


def normalize_keys(obj: Any) -> Any:
    """Recursively turn numeric dict keys into floats so 10, 10.0 and '10' agree."""
    if isinstance(obj, Mapping):
        return {_as_number(k): normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_keys(v) for v in obj]
    return obj


def _is_numeric_keyed(d: Mapping) -> bool:
    return bool(d) and all(isinstance(k, float) for k in d)


def deep_merge(base: dict, override: Mapping) -> dict:
    """Merge ``override`` into a copy of ``base``.

    Nested dicts merge key by key, except numeric-keyed dicts (curves), which
    the override replaces outright.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if (
            isinstance(v, Mapping)
            and isinstance(out.get(k), Mapping)
            and not _is_numeric_keyed(v)
            and not _is_numeric_keyed(out[k])
        ):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------- #
#  scenarios
# --------------------------------------------------------------------------- #
def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return normalize_keys(data)


def load_scenario(name_or_path: str | Path) -> dict:
    """Load a scenario by name (``bear_steepener``) or by file path."""
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return _read_yaml(p)
    candidate = SCENARIO_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return _read_yaml(candidate)
    raise FileNotFoundError(
        f"Unknown scenario '{name_or_path}'. Available: {', '.join(list_scenarios())}"
    )


# --------------------------------------------------------------------------- #
#  --set overrides
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"\[([^\]]+)\]|([^.\[\]]+)")


def _split_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for bracket, plain in _TOKEN.findall(path):
        tokens.append(_as_number(bracket if bracket else plain))
    if not tokens:
        raise ValueError(f"Empty override path: '{path}'")
    return tokens


def apply_override(cfg: dict, assignment: str) -> dict:
    """Apply ``a.b.c=value`` (value parsed as YAML) to ``cfg`` in place.

    Use brackets for keys that contain a dot: ``treasury.curve_start[0.25]=4.1``.
    """
    if "=" not in assignment:
        raise ValueError(f"Override must look like key.path=value, got '{assignment}'")
    path, raw = assignment.split("=", 1)
    keys = _split_path(path.strip())
    value = normalize_keys(yaml.safe_load(raw.strip()))
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value
    return cfg


# --------------------------------------------------------------------------- #
#  public loader
# --------------------------------------------------------------------------- #
def load_config(
    path: str | Path | None = None,
    scenarios: Iterable[str | Path] = (),
    overrides: Iterable[str] = (),
    updates: Mapping | None = None,
) -> dict:
    """Build a validated configuration.

    ``updates`` is a nested dict merged last (used by the Streamlit app).
    """
    cfg = _read_yaml(Path(path) if path else DEFAULT_CONFIG)
    scenario_names = []
    for sc in scenarios:
        layer = load_scenario(sc)
        cfg = deep_merge(cfg, layer)
        scenario_names.append(str(sc))
    if updates:
        cfg = deep_merge(cfg, normalize_keys(dict(updates)))
    for ov in overrides:
        apply_override(cfg, ov)
    cfg = normalize_keys(cfg)
    cfg.setdefault("meta", {})["scenarios_applied"] = scenario_names
    validate(cfg)
    return cfg


# --------------------------------------------------------------------------- #
#  helpers used throughout the model
# --------------------------------------------------------------------------- #
def tenor_values(param: Any, tenors: np.ndarray, name: str = "parameter") -> np.ndarray:
    """Evaluate a scalar or a {tenor: value} dict at ``tenors``.

    Dicts are linearly interpolated in maturity with flat extrapolation, so
    per-tenor settings never need to list every tenor.
    """
    tenors = np.asarray(tenors, dtype=float)
    if param is None:
        raise ValueError(f"{name} is missing")
    if isinstance(param, Mapping):
        if not param:
            return np.zeros_like(tenors)
        keys = np.array(sorted(float(k) for k in param), dtype=float)
        vals = np.array([float(param[k]) for k in keys], dtype=float)
        return np.interp(tenors, keys, vals)
    return np.full_like(tenors, float(param))


def seasonal_weights(table: Mapping) -> np.ndarray:
    """Return a length-12 array (Jan..Dec) of weights that sum to 1."""
    w = np.array([float(table.get(float(m), 0.0)) for m in range(1, 13)], dtype=float)
    if w.sum() <= 0:
        raise ValueError("Seasonality table must have positive weights")
    return w / w.sum()


def _check_prob(value: Any, name: str) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate(cfg: dict) -> None:
    """Raise ``ValueError`` with a readable message on bad inputs."""
    sim = cfg["simulation"]
    if int(sim["n_paths"]) < 100:
        raise ValueError("simulation.n_paths must be at least 100")
    if not (1 <= int(sim["horizon_months"]) <= 60):
        raise ValueError("simulation.horizon_months must be between 1 and 60")
    for m in sim.get("report_months", []):
        if not (1 <= int(m) <= int(sim["horizon_months"])):
            raise ValueError(f"report_months entry {m} lies outside the horizon")

    tenors = np.asarray(cfg["tenors"], dtype=float)
    if tenors.size == 0 or np.any(tenors <= 0) or np.any(np.diff(tenors) <= 0):
        raise ValueError("tenors must be positive and strictly increasing")

    tr = cfg["treasury"]
    if not tr.get("curve_start"):
        raise ValueError("treasury.curve_start is required")
    mode = tr.get("path_mode", "unchanged")
    if mode not in {"unchanged", "forwards", "shifts", "custom"}:
        raise ValueError(f"treasury.path_mode '{mode}' must be unchanged|forwards|shifts|custom")
    if mode == "custom" and not tr.get("curve_target"):
        raise ValueError("treasury.path_mode=custom needs treasury.curve_target")
    if mode == "shifts" and not tr.get("shifts_bp"):
        raise ValueError("treasury.path_mode=shifts needs treasury.shifts_bp")

    mu = cfg["muni"]
    if not mu.get("ratio_start") and not mu.get("curve_start"):
        raise ValueError("Provide muni.curve_start (MMD yields) or muni.ratio_start")

    _check_prob(cfg["macro"]["recession"]["prob_12m"], "macro.recession.prob_12m")
    _check_prob(cfg["tax"]["change_prob_12m"], "tax.change_prob_12m")
    _check_prob(cfg["tax"]["exemption_threat_prob_12m"], "tax.exemption_threat_prob_12m")
    _check_prob(cfg["tax"]["marginal_rate"], "tax.marginal_rate")

    val = cfg["valuation"]
    bands = list(val["bands_z"])
    if any(b2 <= b1 for b1, b2 in zip(bands, bands[1:])):
        raise ValueError("valuation.bands_z must be strictly increasing")
    if len(val["labels"]) != len(bands) + 1:
        raise ValueError("valuation.labels needs exactly one more entry than bands_z")
    if np.any(tenor_values(val["reference"]["sd"], tenors, "valuation.reference.sd") <= 0):
        raise ValueError("valuation.reference.sd must be positive")
