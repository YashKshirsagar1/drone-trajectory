"""
Experiment runner: orbit vs. naive spline vs. optimized on many buildings.

For each building type and seed, a synthetic "pilot" picks 4-8 photo
locations ~20 m out from the walls (the paper's section 4.2 rule: bisecting
edges, plus outside corners), jittered +-5 m. All three methods are planned
from the same waypoints, scored with `evaluate`, and optionally flown through
the wind simulator and scored again. Paired t-tests (optimized vs. naive,
optimized vs. orbit, paired on building x seed) mirror the paper's Tables 1-2,
split by concave vs. convex.

    python experiments.py                 # 6 seeds, all buildings, ~5 min on 6 cores
    python experiments.py --seeds 2 --buildings rect L --maxiter 40   # quick check

Outputs (results/): experiments.csv (one row per case x method x condition),
experiments_ttests.csv, experiments_cases.png (seed-0 paths per building).

Caveat: buildings and waypoints are synthetic (the paper's 28 Hurricane Irma
targets aren't available), and seeds on the same building are not independent
buildings, so p-values describe this synthetic set, not the field.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # parallelism is across cases

import contextlib
import csv
import io
import time

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import stats

from objective import (Building, ObjectiveConfig, SplinePath, Weights, evaluate,
                       objective, orbit_from_waypoints, sort_ccw)

# Footprints in metres. Courtyard = enclosed yard reachable through a 30 m
# mouth (a fully enclosed yard can't be seen from a fixed-altitude loop in 2D).
BUILDINGS = {
    "rect":      ("convex", [(0, 0), (50, 0), (50, 25), (0, 25)]),
    "hexagon":   ("convex", [(0, 0), (40, -5), (60, 20), (45, 45), (10, 45), (-10, 20)]),
    "L":         ("concave", [(0, 0), (60, 0), (60, 25), (25, 25), (25, 60), (0, 60)]),
    "U":         ("concave", [(0, 0), (80, 0), (80, 50), (60, 50), (60, 20), (20, 20),
                              (20, 50), (0, 50)]),
    "cross":     ("concave", [(20, 0), (40, 0), (40, 20), (60, 20), (60, 40), (40, 40),
                              (40, 60), (20, 60), (20, 40), (0, 40), (0, 20), (20, 20)]),
    "courtyard": ("concave", [(0, 0), (70, 0), (70, 70), (50, 70), (50, 55), (55, 55),
                              (55, 15), (15, 15), (15, 55), (20, 55), (20, 70), (0, 70)]),
    "multi":     ("concave", [[(0, 0), (30, 0), (30, 40), (0, 40)],
                              [(45, 10), (80, 10), (80, 35), (45, 35)]]),
}
METHODS = ("orbit", "naive", "optimized")
METRICS = ["coverage_pct", "angle_dev_target_deg", "angle_dev_uav_deg", "obs_distance_m",
           "visual_stability_pct", "smoothness_deg_per_frame", "max_yaw_step_deg",
           "distance_m", "time_s", "min_standoff_m"]
HIGHER_IS_BETTER = {"coverage_pct", "visual_stability_pct", "min_standoff_m"}


# ---------------------------------------------------------------------------
# Synthetic pilot
# ---------------------------------------------------------------------------
def pilot_candidates(b: Building, offset=20.0):
    """Photo spots `offset` m out: edge midpoints and outside (convex) corners."""
    C = list((b.A + b.B) / 2 + offset * b.edge_normals)
    for R in b.rings:
        prev, nxt = R - np.roll(R, 1, 0), np.roll(R, -1, 0) - R
        n_prev = np.stack([prev[:, 1], -prev[:, 0]], 1)
        n_next = np.stack([nxt[:, 1], -nxt[:, 0]], 1)
        bis = (n_prev / np.linalg.norm(n_prev, axis=1, keepdims=True)
               + n_next / np.linalg.norm(n_next, axis=1, keepdims=True))
        convex = (prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0]) > 0
        C += list(R[convex] + offset * bis[convex]
                  / np.linalg.norm(bis[convex], axis=1, keepdims=True))
    C = np.array(C)
    return C[b.signed_distance(C) >= 0.6 * offset]


def pilot_waypoints(b: Building, rng, k_range=(4, 8), jitter=5.0, tries=500):
    """4-8 jittered candidates whose CCW spline encloses the building
    without passing through it (the pilot still flies a sensible loop).

    Like a pilot circling the building: the circle around the centroid is cut
    into K sectors (random phase) and one photo spot is picked per sector.
    """
    C = pilot_candidates(b)
    ang = np.arctan2(*(C - b.centroid).T[::-1])
    for _ in range(tries):
        K = int(rng.integers(k_range[0], k_range[1] + 1))
        sector = np.floor(((ang - rng.uniform(0, 2 * np.pi)) % (2 * np.pi))
                          / (2 * np.pi / K)).astype(int)
        pick = [rng.choice(np.flatnonzero(sector == s)) for s in range(K)
                if np.any(sector == s)]
        if len(pick) < k_range[0]:
            continue
        W = sort_ccw(C[pick] + rng.uniform(-jitter, jitter, (len(pick), 2)))
        P, _, _ = SplinePath(W).sample(400)
        if b.signed_distance(P).min() > 0 and MplPath(P).contains_points(b.V).all():
            return W
    raise RuntimeError("no valid pilot waypoints found")


# ---------------------------------------------------------------------------
# One case = one building x one seed
# ---------------------------------------------------------------------------
def run_case(job):
    from demo import optimize
    from sim import FlownPath, SimConfig, fly

    name, seed, opts = job
    group, verts = BUILDINGS[name]
    b = Building(verts)
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    wps = pilot_waypoints(b, rng)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        opt = optimize(b, wps, Weights(), ObjectiveConfig(), maxiter=opts["maxiter"])
    t_opt = time.time() - t0
    paths = {"orbit": orbit_from_waypoints(b, wps), "naive": SplinePath(wps),
             "optimized": opt}
    J_cfg = ObjectiveConfig(L_ref=paths["naive"].sample(400)[2])

    rows = []
    for method, path in paths.items():
        base = dict(building=name, group=group, seed=seed, n_waypoints=len(wps),
                    method=method, J=objective(path, b, Weights(), J_cfg),
                    opt_time_s=t_opt if method == "optimized" else np.nan)
        rows.append({**base, "condition": "planned", **evaluate(path, b)})
        if opts["wind"] > 0:
            cfg = SimConfig(wind_std=opts["wind"], wind_mean=(0.5, 0.0))
            log = fly(path, cfg, seed=seed)
            s = log.summary()
            rows.append({**base, "condition": "wind", **evaluate(FlownPath(log), b),
                         "pos_err_rms_m": s["pos_err_rms_m"],
                         "yaw_rate_saturated_pct": s["yaw_rate_saturated_pct"]})
    return rows, wps, paths


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def paired_tests(rows, metrics=METRICS):
    """Optimized vs. naive and vs. orbit, paired on (building, seed)."""
    out = []
    conds = sorted({r["condition"] for r in rows}, key=lambda c: c != "planned")
    for cond in conds:
        for group in ("all", "concave", "convex"):
            sel = [r for r in rows if r["condition"] == cond
                   and (group == "all" or r["group"] == group)]
            by = {(r["building"], r["seed"], r["method"]): r for r in sel}
            keys = sorted({(r["building"], r["seed"]) for r in sel})
            for other in ("naive", "orbit"):
                for m in metrics:
                    a = np.array([by[k + ("optimized",)][m] for k in keys], float)
                    o = np.array([by[k + (other,)][m] for k in keys], float)
                    ok = np.isfinite(a) & np.isfinite(o)
                    a, o = a[ok], o[ok]
                    d = a - o
                    if len(d) < 2 or np.allclose(d, 0):
                        t = p = p_w = np.nan
                    else:
                        t, p = stats.ttest_rel(a, o)
                        p_w = stats.wilcoxon(a, o).pvalue
                    better = (d.mean() > 0) == (m in HIGHER_IS_BETTER)
                    out.append(dict(condition=cond, group=group, comparison=f"optimized-{other}",
                                    metric=m, n=len(d), mean_optimized=a.mean(),
                                    mean_other=o.mean(), mean_diff=d.mean(),
                                    sd_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
                                    t=t, p=p, p_wilcoxon=p_w,
                                    optimized_better=bool(better and p < 0.05)
                                    if np.isfinite(p) else False))
    return out


def print_tables(rows, tests):
    short = {"coverage_pct": "Coverage %", "angle_dev_target_deg": "Angle target °",
             "angle_dev_uav_deg": "Angle UAV °", "obs_distance_m": "Obs. dist m",
             "visual_stability_pct": "Stability %", "smoothness_deg_per_frame": "Smooth °/frame",
             "max_yaw_step_deg": "Max yaw step °", "distance_m": "Distance m",
             "time_s": "Time s", "min_standoff_m": "Min standoff m"}
    T = {(t["condition"], t["group"], t["comparison"], t["metric"]): t for t in tests}

    def stars(p):
        return "" if not np.isfinite(p) else "***" if p < .001 else "**" if p < .01 \
            else "*" if p < .05 else ""

    for cond in sorted({t["condition"] for t in tests}, key=lambda c: c != "planned"):
        for group in ("concave", "convex"):
            n = T[(cond, group, "optimized-naive", METRICS[0])]["n"]
            print(f"\n== {cond}, {group} (n = {n} paired cases) ==")
            print(f"{'metric':16s}{'orbit':>9s}{'naive':>9s}{'optim':>9s}"
                  f"{'Δ vs naive':>14s}{'p':>9s}{'Δ vs orbit':>14s}{'p':>9s}")
            for m in METRICS:
                tn = T[(cond, group, "optimized-naive", m)]
                to = T[(cond, group, "optimized-orbit", m)]
                print(f"{short[m]:16s}{to['mean_other']:9.2f}{tn['mean_other']:9.2f}"
                      f"{tn['mean_optimized']:9.2f}"
                      f"{tn['mean_diff']:+10.2f}{stars(tn['p']):>4s}{tn['p']:9.3g}"
                      f"{to['mean_diff']:+10.2f}{stars(to['p']):>4s}{to['p']:9.3g}")
    print("\npaired t-test: * p<.05  ** p<.01  *** p<.001 (Wilcoxon p in the CSV)")


def plot_cases(cases, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"orbit": "#2a78d6", "naive": "#eb6834", "optimized": "#1baf7a"}
    n = len(cases)
    cols = min(4, n)
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.4 * cols, 3.4 * rws + 0.4),
                             squeeze=False, layout="constrained")
    for ax, (name, b, wps, paths) in zip(axes.ravel(), cases):
        for R in b.rings:
            ax.fill(R[:, 0], R[:, 1], color="#d9d9d9", zorder=0)
            ax.plot(*np.vstack([R, R[:1]]).T, color="#555", lw=0.8)
        for m in METHODS:
            P, _, _ = paths[m].sample(400)
            ax.plot(*np.vstack([P, P[:1]]).T, color=colors[m], lw=1.6, label=m)
        ax.plot(*wps.T, "D", ms=4, color="#222", label="pilot waypoints")
        ax.set_title(f"{name} ({BUILDINGS[name][0]}), seed 0", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    import argparse
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="+", default=list(BUILDINGS), choices=list(BUILDINGS))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--wind", type=float, default=1.5,
                    help="also fly each plan in this turbulence (m/s); 0 = planned only")
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(name, s, dict(wind=args.wind, maxiter=args.maxiter))
            for name in args.buildings for s in range(args.seeds)]
    print(f"{len(jobs)} cases ({len(args.buildings)} buildings x {args.seeds} seeds) "
          f"on {args.jobs} processes")
    rows, cases = [], {}
    fields = (["building", "group", "seed", "n_waypoints", "method", "condition", "J"]
              + METRICS + ["fov_coverage_pct", "pos_err_rms_m", "yaw_rate_saturated_pct",
                           "opt_time_s"])
    t0 = time.time()
    csv_path = os.path.join(args.out, "experiments.csv")
    with open(csv_path, "w", newline="") as f, Pool(args.jobs) as pool:
        wr = csv.DictWriter(f, fields, extrasaction="ignore")
        wr.writeheader()
        for k, ((name, seed, _), (case_rows, wps, paths)) in enumerate(
                zip(jobs, pool.imap(run_case, jobs))):
            rows += case_rows
            wr.writerows(case_rows)
            f.flush()
            if seed == 0:
                cases[name] = (name, Building(BUILDINGS[name][1]), wps, paths)
            opt = next(r for r in case_rows if r["method"] == "optimized")
            print(f"  [{k + 1:2d}/{len(jobs)}] {name:9s} seed {seed}  K={len(wps)}  "
                  f"opt {opt['opt_time_s']:4.0f}s  {time.time() - t0:5.0f}s elapsed")

    tests = paired_tests(rows)
    with open(os.path.join(args.out, "experiments_ttests.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, list(tests[0]))
        wr.writeheader()
        wr.writerows(tests)
    print_tables(rows, tests)
    plot_cases([cases[n] for n in args.buildings if n in cases],
               os.path.join(args.out, "experiments_cases.png"))
    print(f"\nwrote {csv_path}, experiments_ttests.csv, experiments_cases.png "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
