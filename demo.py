"""Compare orbit, naive spline (paper) and optimized spline on two buildings."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from objective import (Building, ObjectiveConfig, SplinePath, Weights,
                       evaluate, objective, orbit_from_waypoints, sort_ccw)

# L-shaped (concave) and rectangular (convex) footprints, metres
L_SHAPE = [(0, 0), (60, 0), (60, 25), (25, 25), (25, 60), (0, 60)]
RECT = [(0, 0), (50, 0), (50, 25), (0, 25)]
U_SHAPE = [(0, 0), (80, 0), (80, 50), (60, 50), (60, 20), (20, 20), (20, 50), (0, 50)]

# "Pilot" image locations: ~20 m out, bisecting edges (paper section 4.2)
WP_L = [(30, -20), (80, 12), (45, 45), (12, 80), (-20, 30)]
WP_R = [(25, -20), (70, 12), (25, 45), (-20, 12)]
WP_U = [(40, -20), (100, 25), (70, 70), (40, 38), (10, 70), (-20, 25)]


def optimize(building, waypoints, weights, cfg, K=10, maxiter=150):
    naive = SplinePath(sort_ccw(waypoints))
    xy0, _, L0 = naive.sample(K)                    # densify pilot path to K pts
    cfg.L_ref = L0
    x0 = np.concatenate([xy0.ravel(), np.zeros(K)])
    bounds = [(None, None)] * (2 * K) + [(-60, 60)] * K
    f = lambda x: objective(SplinePath.from_vector(x, K, K), building, weights, cfg)
    t0 = time.time()
    res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=maxiter, eps=0.25))
    print(f"  optimizer: J {f(x0):.3f} -> {res.fun:.3f} "
          f"in {res.nit} iters, {time.time() - t0:.0f}s")
    return SplinePath.from_vector(res.x, K, K)


def plot(ax, building, path, title, waypoints=None):
    P, yaw, _ = path.sample(160)
    for R in building.rings:
        V = np.vstack([R, R[:1]])
        ax.fill(V[:, 0], V[:, 1], color="#ccc")
        ax.plot(V[:, 0], V[:, 1], "k-", lw=1)
    ax.plot(*np.vstack([P, P[:1]]).T, "-", color="#1f6feb", lw=1.5)
    k = slice(None, None, 8)
    ax.quiver(P[k, 0], P[k, 1], np.cos(np.radians(yaw[k])), np.sin(np.radians(yaw[k])),
              color="#d9480f", scale=18, width=0.006)
    if waypoints is not None:
        ax.plot(*np.asarray(waypoints).T, "D", ms=4, color="k")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    cfg = ObjectiveConfig()
    weights = Weights()
    rows = []
    cases = [("U-shape (concave)", U_SHAPE, WP_U),
             ("L-shape (concave)", L_SHAPE, WP_L),
             ("Rectangle (convex)", RECT, WP_R)]
    fig, axes = plt.subplots(len(cases), 3, figsize=(11, 11))
    for r, (name, verts, wps) in enumerate(cases):
        b = Building(verts)
        print(name)
        orbit = orbit_from_waypoints(b, wps)
        naive = SplinePath(sort_ccw(wps))
        opt = optimize(b, wps, weights, ObjectiveConfig())
        for c, (label, p) in enumerate([("Orbit", orbit), ("Naive spline", naive),
                                        ("Optimized", opt)]):
            m = evaluate(p, b, cfg.speed)
            rows.append((name, label, m))
            plot(axes[r, c], b, p,
                 f"{name} - {label}\ncov {m['coverage_pct']:.0f}%  "
                 f"angle {m['angle_dev_target_deg']:.0f}°  dist {m['obs_distance_m']:.0f} m",
                 wps if label == "Naive spline" else None)
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=130)

    keys = list(rows[0][2].keys())
    print("\n" + "building,method," + ",".join(keys))
    for name, label, m in rows:
        print(f"{name},{label}," + ",".join(f"{m[k]:.2f}" for k in keys))


if __name__ == "__main__":
    main()
