import numpy as np
import pandas as pd

from conftest import run
from muni_mc import valuation as val


def test_classify_boundaries():
    bands = [-1.0, -0.5, 0.5, 1.0]
    assert list(val.classify([-2, -1, -0.75, 0, 0.5, 0.99, 1.0, 3], bands)) == [0, 1, 1, 2, 3, 3, 4, 4]


def test_today_table_matches_hand_calculation(base_result):
    t = val.valuation_today(base_result)
    row = t[t["Tenor"] == "10Y"].iloc[0]
    assert np.isclose(row["Ratio %"], 100 * 3.25 / 5.07)
    assert np.isclose(row["z vs history"], (3.25 / 5.07 - 0.70) / 0.07)
    assert row["History view"] == "Expensive"


def test_zone_probabilities_sum_to_one(base_result):
    for m in base_result.report_months:
        z = val.zone_probabilities(base_result, m)
        assert np.allclose(z.sum(axis=1), 1.0)


def test_attribution_bridge_is_exact(base_result):
    a = val.attribution(base_result)
    parts = a.drop(columns=["Tenor", "Start ratio %", "Expected ratio %"]).sum(axis=1)
    assert np.allclose(a["Start ratio %"] + parts, a["Expected ratio %"], atol=1e-8)


def test_tax_breakeven_table(base_result):
    tb = val.tax_breakeven_table(base_result)
    top = tb[tb["Bracket"] == "37% + 3.8% NIIT"].iloc[0]
    assert np.isclose(top["Breakeven ratio %"], 59.2)
    assert np.isclose(top["10Y cushion today (pts)"], 100 * 3.25 / 5.07 - 59.2)


def test_history_csv_drives_reference_stats(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-11-19", "2026-11-19")
    df = pd.DataFrame({"date": dates})
    for t, m in [(2, 60), (10, 70), (30, 90)]:
        df[f"ratio_{t}"] = m + rng.normal(0, 5, len(dates))  # percent
        df[f"ust_{t}"] = 4.0
    path = tmp_path / "hist.csv"
    df.to_csv(path, index=False)
    r = run(overrides=[f"valuation.history_csv={path}"], n=200)
    assert np.allclose(r.reference["mean"], np.interp([2, 5, 10, 20, 30], [2, 10, 30], [0.60, 0.70, 0.90]), atol=0.01)
    assert np.allclose(r.reference["sd"][[0, 2, 4]], 0.05, atol=0.005)
    assert np.allclose(r.reference["avg_treasury"], 4.0)
    assert "obs" in r.reference["source"]
