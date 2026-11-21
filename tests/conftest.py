import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muni_mc import load_config, run_simulation  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg():
    return load_config(overrides=["simulation.n_paths=3000", "simulation.seed=7"])


@pytest.fixture(scope="session")
def base_result(small_cfg):
    return run_simulation(small_cfg)


def run(scenarios=(), overrides=(), n=3000):
    cfg = load_config(scenarios=scenarios, overrides=[f"simulation.n_paths={n}", "simulation.seed=7", *overrides])
    return run_simulation(cfg)
