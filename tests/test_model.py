import numpy as np

from conftest import run
from muni_mc.curves import interp_curve


def test_shapes_and_start_values(base_result, small_cfg):
    r = base_result
    n, h, k = small_cfg["simulation"]["n_paths"], small_cfg["simulation"]["horizon_months"], len(small_cfg["tenors"])
    for arr in (r.ratio, r.treasury, r.muni, r.fair_value):
        assert arr.shape == (n, h + 1, k)
        assert np.isfinite(arr).all()
    # starting ratios are MMD / UST from the config
    mmd = interp_curve(small_cfg["muni"]["curve_start"], r.tenors)
    ust = interp_curve(small_cfg["treasury"]["curve_start"], r.tenors)
    assert np.allclose(r.ratio[:, 0, :], mmd / ust)
    assert np.allclose(r.muni, r.ratio * r.treasury)


def test_reproducible_with_seed():
    a, b = run(n=500), run(n=500)
    assert np.array_equal(a.ratio, b.ratio)


def test_treasury_mean_follows_expected_path():
    r = run(scenarios=["bear_steepener"])
    base = r.cfg["treasury"]
    start = interp_curve(base["curve_start"], r.tenors)
    target = start + np.interp(r.tenors, [2, 5, 10, 30], [0.20, 0.40, 0.60, 0.80])
    assert np.allclose(r.treasury[:, -1, :].mean(axis=0), target, atol=0.02)


def test_fair_value_components_sum(base_result):
    r = base_result
    total = sum(r.fv_components.values())
    assert np.allclose(total, r.fair_value.mean(axis=0), atol=1e-10)


def test_fair_value_hits_history_at_reference_conditions():
    """At the reference Treasury level and neutral drivers, FV = historical mean ratio."""
    ref_t = {2: 3.9, 5: 3.9, 10: 4.05, 20: 4.4, 30: 4.3}
    ov = [
        f"treasury.curve_start={ref_t}", f"muni.curve_start={ {k: v * 0.65 for k, v in ref_t.items()} }",
        "macro.rate_vol.start=105", "macro.hy_oas.start=380", "macro.recession.prob_12m=0",
        "flows.start=1.0", "supply.annual_gross_issuance=0", "supply.annual_reinvestment=0",
        "supply.reference_net_monthly=0",
    ]
    r = run(overrides=ov, n=200)
    assert np.allclose(r.fair_value[0, 0], r.reference["mean"], atol=1e-9)


def test_no_noise_ratio_converges_to_fair_value():
    ov = [
        "muni.ratio_vol=0", "treasury.factor_vol_bp={level: 0, slope: 0, curvature: 0}",
        "treasury.term_premium.vol_bp=0", "macro.inflation.vol=0", "macro.unemployment.vol=0",
        "macro.rate_vol.vol=0", "macro.hy_oas.vol=0", "macro.recession.prob_12m=0",
        "flows.vol=0", "supply.issuance_vol=0", "liquidity_events.annual_intensity=0",
        "tax.change_prob_12m=0", "tax.exemption_threat_prob_12m=0", "simulation.horizon_months=36",
        "simulation.report_months=[12, 36]",
        f"supply.issuance_seasonality={ {m: 1 for m in range(1, 13)} }",
        f"supply.reinvestment_seasonality={ {m: 1 for m in range(1, 13)} }",
    ]
    r = run(overrides=ov, n=200)
    gap_end = np.abs(r.ratio[:, -1] - r.fair_value[:, -1]).mean(axis=0)
    assert (gap_end < 0.003).all()
    # only residual randomness left is the Fed's discrete 25bp steps
    assert (r.ratio[:, -1].std(axis=0) < 0.003).all()


def test_tax_hike_richens_ratios():
    base = run()
    hike = run(scenarios=["tax_hike"])
    diff = np.median(hike.ratio[:, -1], axis=0) - np.median(base.ratio[:, -1], axis=0)
    assert (diff < -0.01).all()          # at least 1 ratio point richer everywhere
    assert np.allclose(hike.tax_rate[:, -1], 0.408 + 0.026)


def test_recession_cheapens_front_end_and_lowers_yields():
    base = run()
    hard = run(scenarios=["hard_landing"])
    assert np.median(hard.ratio[:, -1, 0]) > np.median(base.ratio[:, -1, 0]) + 0.05
    assert np.median(hard.treasury[:, -1, 0]) < np.median(base.treasury[:, -1, 0]) - 0.5
    assert hard.events["recession"].mean() > 0.8


def test_more_supply_and_outflows_cheapen():
    base = run()
    glut = run(scenarios=["supply_glut"])
    assert (np.median(glut.ratio[:, -1], axis=0) > np.median(base.ratio[:, -1], axis=0)).all()


def test_higher_rates_lower_fair_value():
    lo = run(overrides=["treasury.path_mode=shifts", "treasury.shifts_bp={2: -100, 30: -100}"])
    hi = run(overrides=["treasury.path_mode=shifts", "treasury.shifts_bp={2: 100, 30: 100}"])
    assert (lo.fv_components["rate_level"][-1] > hi.fv_components["rate_level"][-1]).all()


def test_ratio_start_override():
    r = run(overrides=["muni.ratio_start={2: 0.60, 30: 0.90}"], n=200)
    assert np.isclose(r.ratio[0, 0, 0], 0.60) and np.isclose(r.ratio[0, 0, -1], 0.90)


def test_fed_moves_in_quarter_points(base_result):
    ff = base_result.macro["fed_funds"]
    steps = np.diff(ff, axis=1)
    assert np.allclose(np.round(steps / 0.25) * 0.25, steps)
