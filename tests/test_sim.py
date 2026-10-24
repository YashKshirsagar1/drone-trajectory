import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objective import Building, OrbitPath, SplinePath, evaluate, sort_ccw, wrap_deg
from sim import FlownPath, SimConfig, fly

RECT = Building([(0, 0), (50, 0), (50, 25), (0, 25)])
WP_U = [(40, -20), (100, 25), (70, 70), (40, 38), (10, 70), (-20, 25)]


def test_calm_orbit_tracks_plan():
    path = OrbitPath(np.array([25.0, 12.5]), 40.0)
    log = fly(path, SimConfig())
    s = log.summary()
    assert s["pos_err_max_m"] < 0.2
    assert s["yaw_err_max_deg"] < 1.0
    assert s["yaw_rate_saturated_pct"] == 0
    # one lap at 4 Hz
    assert len(log.t) == round(2 * np.pi * 40 / (2.0 * 0.25))
    planned, flown = evaluate(path, RECT), evaluate(FlownPath(log), RECT)
    for k in ("coverage_pct", "angle_dev_target_deg", "obs_distance_m", "distance_m"):
        assert flown[k] == pytest.approx(planned[k], rel=0.02, abs=0.5), k


def test_limits_respected():
    cfg = SimConfig(max_yaw_rate=10.0, wind_std=2.0)
    log = fly(SplinePath(sort_ccw(WP_U)), cfg, seed=1)
    assert np.all(np.abs(log.yaw_rate) <= cfg.max_yaw_rate + 1e-9)
    # yaw change between 4 Hz frames bounded by the rate limit
    assert np.abs(wrap_deg(np.diff(log.yaw))).max() <= cfg.max_yaw_rate * 0.25 + 1e-6
    assert log.summary()["yaw_rate_saturated_pct"] > 0
    drone_speed = np.linalg.norm(log.vel - log.wind, axis=1)
    assert drone_speed.max() <= cfg.max_speed + 1e-9


def test_wind_is_seeded_and_causes_error():
    path = SplinePath(sort_ccw(WP_U))
    cfg = SimConfig(wind_std=1.5)
    a, b, c = fly(path, cfg, seed=3), fly(path, cfg, seed=3), fly(path, cfg, seed=4)
    assert np.array_equal(a.xy, b.xy)
    assert not np.array_equal(a.xy, c.xy)
    assert a.summary()["pos_err_rms_m"] > 0.1


def test_gust_pushes_drone_then_recovers():
    path = OrbitPath(np.array([25.0, 12.5]), 40.0)
    log = fly(path, SimConfig(gusts=[(20.0, 3.0, (4.0, 0.0))]))
    e = log.pos_err
    assert e[log.t < 20].max() < 0.2
    assert e.max() > 1.0
    assert e[log.t > 40].max() < 0.2


@pytest.mark.parametrize("by", ["arclength", "time"])
def test_flownpath_contract(by):
    log = fly(SplinePath(sort_ccw(WP_U)), SimConfig(wind_std=1.0))
    xy, yaw, L = FlownPath(log, by=by).sample(123)
    assert xy.shape == (123, 2) and yaw.shape == (123,)
    assert np.all(np.abs(yaw) <= 180)
    assert L == pytest.approx(np.linalg.norm(np.diff(np.vstack([log.xy, log.xy[:1]]), axis=0),
                                             axis=1).sum(), rel=1e-6)
    assert np.allclose(xy[0], log.xy[0])
    if by == "arclength":
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        assert seg.std() / seg.mean() < 0.1


def test_flown_eval_uses_logged_frames():
    path = SplinePath(sort_ccw(WP_U))
    log = fly(path, SimConfig())
    xy, yaw, _ = FlownPath(log).sample(len(log.t))
    assert np.allclose(xy, log.xy) and np.allclose(wrap_deg(yaw - log.yaw), 0, atol=1e-9)
    m = evaluate(FlownPath(log), RECT)
    assert m["smoothness_deg_per_frame"] == pytest.approx(
        np.abs(wrap_deg(np.diff(np.append(log.yaw, log.yaw[0])))).mean())
