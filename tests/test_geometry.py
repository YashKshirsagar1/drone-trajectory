import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import BUILDINGS, pilot_candidates, pilot_waypoints
from objective import Building, SplinePath, evaluate
from matplotlib.path import Path as MplPath

U = Building(BUILDINGS["U"][1])
MULTI = Building(BUILDINGS["multi"][1])


@pytest.mark.parametrize("name", list(BUILDINGS))
def test_normals_point_outward(name):
    b = Building(BUILDINGS[name][1])
    assert np.all(b.signed_distance(b.m + 0.5 * b.n) > 0)
    assert np.all(b.signed_distance(b.m - 0.5 * b.n) < 0)


def test_clockwise_input_is_reoriented():
    cw = Building(BUILDINGS["rect"][1][::-1])
    assert np.allclose(cw.n.sum(0), 0) and np.all(cw.signed_distance(cw.m + 0.5 * cw.n) > 0)


def test_line_of_sight_blocked_across_notch():
    # from left of the U, the inner face of the right arm (x=60, facing -x) is hidden
    j = np.flatnonzero((np.isclose(U.m[:, 0], 60)) & (U.n[:, 0] < 0) & (U.m[:, 1] > 25))
    assert len(j) and not U.line_of_sight(np.array([[-20.0, 35.0]]))[0, j].any()
    # but visible from inside the notch
    assert U.line_of_sight(np.array([[40.0, 40.0]]))[0, j].all()


def test_center_ray_hits_expected_wall():
    hit, d, inc, k = U.cast_center_rays(np.array([[40.0, -20.0], [40.0, 40.0]]),
                                        np.array([[0.0, 1.0], [0.0, -1.0]]))
    assert hit.all()
    assert np.allclose(d, [20.0, 20.0]) and np.allclose(inc, 0)
    assert np.allclose(U.A[k[0], 1], 0) and np.allclose(U.A[k[1], 1], 20)


def test_multi_building_geometry():
    assert len(MULTI.rings) == 2 and len(MULTI.A) == 8
    sd = MULTI.signed_distance(np.array([[15.0, 20.0], [60.0, 20.0], [37.5, 20.0]]))
    assert sd[0] < 0 and sd[1] < 0 and sd[2] == pytest.approx(7.5)
    # building B blocks A's right wall from a viewpoint east of B
    j = np.flatnonzero(np.isclose(MULTI.m[:, 0], 30) & (MULTI.n[:, 0] > 0))
    assert not MULTI.line_of_sight(np.array([[100.0, 20.0]]))[0, j].any()


@pytest.mark.parametrize("name", list(BUILDINGS))
def test_pilot_waypoints_enclose_building(name):
    b = Building(BUILDINGS[name][1])
    assert np.all(b.signed_distance(pilot_candidates(b)) >= 12)
    rng = np.random.default_rng(0)
    for _ in range(5):
        W = pilot_waypoints(b, rng)
        assert 4 <= len(W) <= 8
        P, _, _ = SplinePath(W).sample(400)
        assert b.signed_distance(P).min() > 0
        assert MplPath(P).contains_points(b.V).all()
