import numpy as np

from muni_mc.curves import fit_nelson_siegel, forward_curve, interp_curve, interp_weights, nearest_correlation, ns_loadings


def test_nelson_siegel_recovers_factors():
    lam = 0.7308
    tenors = np.array([0.25, 1, 2, 5, 10, 20, 30])
    beta = np.array([5.0, -1.0, 0.5])
    curve = dict(zip(tenors, ns_loadings(tenors, lam) @ beta))
    assert np.allclose(fit_nelson_siegel(curve, lam), beta, atol=1e-8)


def test_flat_curve_forwards_are_flat():
    curve = {1: 4.0, 5: 4.0, 30: 4.0}
    assert np.allclose(forward_curve(curve, [2, 10, 30], 1.0), 4.0)


def test_interp_weights_match_np_interp():
    tenors = np.array([2.0, 5.0, 10.0, 20.0, 30.0])
    vals = np.array([1.0, 2.0, 4.0, 6.0, 7.0])
    for target in (1.0, 7.0, 10.0, 25.0, 40.0):
        assert np.isclose(vals @ interp_weights(tenors, target), np.interp(target, tenors, vals))


def test_nearest_correlation_repairs_invalid_matrix():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = nearest_correlation(bad)
    assert np.linalg.eigvalsh(fixed).min() > 0
    assert np.allclose(np.diag(fixed), 1.0)


def test_interp_curve_flat_beyond_ends():
    assert np.allclose(interp_curve({2: 4.0, 10: 5.0}, [1, 6, 40]), [4.0, 4.5, 5.0])
