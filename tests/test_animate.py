import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animate import animate, coverage_history
from objective import Building, OrbitPath, SplinePath, evaluate, sort_ccw
from sim import FlownPath, SimConfig, fly

U = Building([(0, 0), (80, 0), (80, 50), (60, 50), (60, 20), (20, 20), (20, 50), (0, 50)])
WP_U = [(40, -20), (100, 25), (70, 70), (40, 38), (10, 70), (-20, 25)]


@pytest.mark.parametrize("wind", [0.0, 1.5])
def test_final_coverage_matches_evaluate(wind):
    for path in (OrbitPath(U.centroid, 60.0), SplinePath(sort_ccw(WP_U))):
        log = fly(path, SimConfig(wind_std=wind), seed=2)
        best, cov, _, _ = coverage_history(log, U)
        assert np.all(np.diff(cov) >= 0)                      # coverage only grows
        assert cov[-1] == pytest.approx(evaluate(FlownPath(log), U)["coverage_pct"])


def test_renders_gif(tmp_path):
    path = OrbitPath(U.centroid, 60.0)
    log = fly(path, SimConfig(laps=0.1))
    out = tmp_path / "a.gif"
    animate({"Orbit": log, "Orbit again": log}, U, str(out), speedup=40, fps=4)
    assert out.stat().st_size > 1000
