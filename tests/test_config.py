import numpy as np
import pytest

from muni_mc.config import apply_override, deep_merge, list_scenarios, load_config, normalize_keys, tenor_values


def test_default_loads_and_validates():
    cfg = load_config()
    assert cfg["tenors"] == [2, 5, 10, 20, 30]
    assert 10.0 in cfg["treasury"]["curve_start"]  # numeric keys normalised to float


def test_override_numeric_and_bracket_keys():
    cfg = load_config(overrides=["treasury.curve_start.10=4.85", "treasury.curve_start[0.25]=3.9",
                                 "macro.recession.prob_12m=0.35"])
    assert cfg["treasury"]["curve_start"][10.0] == 4.85
    assert cfg["treasury"]["curve_start"][0.25] == 3.9
    assert cfg["macro"]["recession"]["prob_12m"] == 0.35


def test_override_parses_yaml_values():
    cfg = {"a": {}}
    apply_override(cfg, "a.b={2: 10, 30: -5}")
    assert cfg["a"]["b"] == {2.0: 10, 30.0: -5}


def test_curves_are_replaced_not_merged():
    base = normalize_keys({"c": {2: 1.0, 10: 2.0, 30: 3.0}, "x": {"y": 1, "z": 2}})
    over = normalize_keys({"c": {2: 5.0}, "x": {"y": 9}})
    out = deep_merge(base, over)
    assert out["c"] == {2.0: 5.0}
    assert out["x"] == {"y": 9, "z": 2}


def test_tenor_values_interpolates_and_extrapolates_flat():
    vals = tenor_values({2: 1.0, 10: 3.0}, np.array([1, 2, 6, 10, 30]))
    assert np.allclose(vals, [1.0, 1.0, 2.0, 3.0, 3.0])
    assert np.allclose(tenor_values(0.5, np.array([2, 5])), [0.5, 0.5])


@pytest.mark.parametrize("bad", [
    "macro.recession.prob_12m=1.5",
    "treasury.path_mode=sideways",
    "valuation.bands_z=[1, 0]",
    "tenors=[10, 5]",
])
def test_validation_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        load_config(overrides=[bad])


def test_every_scenario_file_loads():
    names = list_scenarios()
    assert {"base", "bear_steepener", "bull_steepener", "hard_landing", "tax_hike"} <= set(names)
    for n in names:
        load_config(scenarios=[n])
